# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo-tracking (``world_state_tracking_reward``) reward terms for the LIBERO DGPO path.

Port of the original ``WorldStateTrackingRewardsCfg`` term functions onto the
unified demo command stack: per-step exponential-kernel tracking of the demo's
EE pose (``ee_pose`` command view), gripper state (``gripper_state`` view), and
object/articulation poses (``source_action`` pose cache via
:class:`~.observations.ObjectTargetPoseDiff`).

All terms degrade to zeros when the demo command stack is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg

from ...utils.commands import resolve_command, resolve_command_term

from .observations import ObjectTargetPoseDiff, _to_torch, joint_pos_diff, joint_vel_diff, layout_active_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "frame_transformer_position_command_error_exp",
    "frame_transformer_orientation_command_error_exp",
    "joint_position_command_error_exp",
    "joint_velocity_command_error_exp",
    "object_position_command_error_exp",
    "object_orientation_command_error_exp",
    "articulation_joint_position_command_error_exp",
]


def _get_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor | None:
    """Fetch a demo command view by name; ``None`` when no command manager / term exists."""
    if getattr(env, "command_manager", None) is None:
        return None
    try:
        return resolve_command(env, command_name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# EE pose tracking (``ee_pose`` command view; base-frame, XYZW)
# ---------------------------------------------------------------------------


def _frame_position_error(
    env: ManagerBasedRLEnv, command_name: str, frame_name: str, target_index: int
) -> torch.Tensor | None:
    """Compute the EE position error [m] vs the demo ``ee_pose`` command.

    Distance between the frame transformer's base-frame target position and the
    command's position slice; ``(num_envs,)``, or ``None`` when the command is absent.
    """
    command = _get_command(env, command_name)
    if command is None or command.shape[-1] < 3:
        return None
    frame_sensor = env.scene[frame_name]
    curr_pos = _to_torch(frame_sensor.data.target_pos_source)[:, target_index, :3]
    return torch.norm(curr_pos - command[:, :3], dim=1)


def _frame_orientation_error(
    env: ManagerBasedRLEnv, command_name: str, frame_name: str, target_index: int
) -> torch.Tensor | None:
    """Compute the quaternion geodesic error [rad] vs the demo ``ee_pose`` command.

    Both quats are sim XYZW (demo loaded WXYZ is converted at load);
    ``(num_envs,)``, or ``None`` when the command is absent.
    """
    command = _get_command(env, command_name)
    if command is None or command.shape[-1] < 7:
        return None
    frame_sensor = env.scene[frame_name]
    curr_quat = _to_torch(frame_sensor.data.target_quat_source)[:, target_index]
    return math_utils.quat_error_magnitude(curr_quat, command[:, 3:7])


def frame_transformer_position_command_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ee_pose",
    frame_name: str = "franka_ee_frame",
    target_index: int = 0,
) -> torch.Tensor:
    """Exp kernel on the EE position error vs the demo ``ee_pose`` command."""
    dist = _frame_position_error(env, command_name, frame_name, target_index)
    if dist is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.exp(-dist / std)


def frame_transformer_orientation_command_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ee_pose",
    frame_name: str = "franka_ee_frame",
    target_index: int = 0,
) -> torch.Tensor:
    """Exp kernel on the EE orientation error vs the demo ``ee_pose`` command."""
    dist = _frame_orientation_error(env, command_name, frame_name, target_index)
    if dist is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.exp(-dist / std)


# ---------------------------------------------------------------------------
# Gripper / joint state tracking (``gripper_state`` / ``joint_state`` views)
# ---------------------------------------------------------------------------


def _gripper_action_is_replayed(env: ManagerBasedRLEnv, action_name: str = "gripper") -> bool:
    """Whether the gripper action term can substitute a demo command for the policy's bit.

    Only :class:`~...utils.gripper_curriculum.CurriculumBinaryJointPositionAction`
    carries ``source_command_name``; the plain ``BinaryJointPositionAction`` the env
    falls back to when ``GRIPPER_CURRICULUM=0`` does not, and under it the gripper is
    policy-controlled at every step.
    """
    action_manager = getattr(env.unwrapped, "action_manager", None)
    if action_manager is None:
        return False
    try:
        term = action_manager.get_term(action_name)
    except (KeyError, AttributeError):
        return False
    return getattr(getattr(term, "cfg", None), "source_command_name", None) is not None


def _gripper_reward_gate(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Per-env 1.0/0.0 gate zeroing gripper rewards while the curriculum owns the bit.

    While the gripper curriculum is active the action term replaces the policy's
    gripper command with the demonstration command, so any reward scoring "does the
    gripper match the demonstration" is a constant with respect to the policy's
    action and carries no gradient.

    Returns ``None`` when nothing is actually overriding the gripper bit (e.g.
    ``GRIPPER_CURRICULUM=0``), so the gate only applies where the reward it is
    guarding is genuinely independent of the policy's action.
    """
    if not _gripper_action_is_replayed(env):
        return None
    tracker = getattr(env.unwrapped, "task_sr_tracker", None)
    replay_mask = getattr(tracker, "last_replay_mask", None)
    if not isinstance(replay_mask, torch.Tensor):
        return None
    return (~replay_mask.to(device=env.device, dtype=torch.bool)).view(-1).float()


def joint_position_command_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    gate_on_gripper_curriculum: bool = False,
) -> torch.Tensor:
    """Exp kernel on the joint-position error vs a demo joint command view.

    Set ``gate_on_gripper_curriculum`` on gripper-joint terms so the reward is zeroed
    for envs whose gripper bit is currently driven by the demonstration; see
    :func:`_gripper_reward_gate`.
    """
    diff = joint_pos_diff(env, command_name=command_name, asset_cfg=asset_cfg)
    reward = torch.exp(-torch.norm(diff, dim=1) / std)
    if gate_on_gripper_curriculum:
        gate = _gripper_reward_gate(env)
        if gate is not None:
            reward = reward * gate
    return reward


def joint_velocity_command_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "joint_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"]),
) -> torch.Tensor:
    """Exp kernel on the joint-VELOCITY error vs the demo ``joint_velocity`` view.

    Ported from ``TrackingRewardsCfg.joint_velocity_tracking``. Unlike the pose
    tracking terms, this term's error is bounded by the demonstration's own
    speed rather than by distance from a fixed reference, so it keeps
    providing gradient even once the policy has drifted far off the
    demonstrated path. ``std`` (0.3-0.5 widens the basin, at the cost of
    parity with the original value) is the first knob to try if training
    stalls on a near-zero-action policy.
    """
    diff = joint_vel_diff(env, command_name=command_name, asset_cfg=asset_cfg)
    return torch.exp(-torch.norm(diff, dim=1) / std)


# ---------------------------------------------------------------------------
# Object / articulation tracking vs the demo pose cache (``source_action``)
# ---------------------------------------------------------------------------


def _first_true_frame(moved: torch.Tensor, time_len: int) -> torch.Tensor:
    """``(C, T, B)`` bool → ``(C, B)`` index of the first True frame, ``time_len`` if never."""
    first = moved.float().argmax(dim=1).to(torch.long)
    return torch.where(moved.any(dim=1), first, torch.full_like(first, time_len))


def _demo_first_move_frames(
    term,
    cached: dict[str, torch.Tensor],
    pos_eps: float,
    quat_eps: float,
    joint_eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """First demo frame at which each ``(trajectory, slot)`` leaves its frame-0 state.

    Returns ``(pose_first, joint_first)``, both ``(num_traj, buffer_len)`` on the pose
    cache's device.  A slot the demonstration never disturbs gets ``T`` (one past the
    last frame) and so is never in play.  Pose and joint movement are tracked
    separately because they dissociate: a cabinet's root pose never moves while its
    drawer joint does, and gating its position term on the joint would hand back
    exactly the free reward this exists to remove.

    Built once per pose cache and stashed on the command term.  Frames past a
    trajectory's own length are zero padding and therefore read as "moved"; harmless,
    because every lookup clamps to ``length - 1``, one frame below the earliest index
    padding can occupy.
    """
    cache_key = (getattr(term, "_traj_pose_buffer_cache_key", None), pos_eps, quat_eps, joint_eps)
    hit = getattr(term, "_demo_first_move_cache", None)
    if hit is not None and getattr(term, "_demo_first_move_cache_key", None) == cache_key:
        return hit

    pos = cached.get("pos")
    quat = cached.get("quat")
    joint = cached.get("joint")
    if pos is None or quat is None or pos.ndim != 4:
        return None
    num_traj, time_len, buffer_len = pos.shape[0], pos.shape[1], pos.shape[2]
    device = pos.device
    pose_first = torch.full((num_traj, buffer_len), time_len, dtype=torch.long, device=device)
    joint_first = torch.full((num_traj, buffer_len), time_len, dtype=torch.long, device=device)

    # Chunked over trajectories: the (num_traj, T, slots) intermediates run to ~26M
    # entries on the 40-task bank.  This is a once-per-run cost, not a per-step one.
    chunk = 256
    for start in range(0, num_traj, chunk):
        stop = min(start + chunk, num_traj)
        pos_c, quat_c = pos[start:stop], quat[start:stop]
        moved = torch.norm(pos_c - pos_c[:, :1], dim=-1) > pos_eps
        # |<q_t, q_0>| → 1 for equal orientations, and the abs makes it double-cover safe.
        dot = (quat_c * quat_c[:, :1]).sum(dim=-1).abs().clamp(max=1.0)
        moved = moved | ((1.0 - dot) > quat_eps)
        pose_first[start:stop] = _first_true_frame(moved, time_len)
        if joint is not None and joint.ndim == 4 and joint.shape[-1] > 0:
            joint_c = joint[start:stop]
            moved_joint = (joint_c - joint_c[:, :1]).abs().amax(dim=-1) > joint_eps
            joint_first[start:stop] = _first_true_frame(moved_joint, time_len)

    result = (pose_first, joint_first)
    term._demo_first_move_cache = result
    term._demo_first_move_cache_key = cache_key
    return result


def _demo_slot_in_play(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_eps: float,
    quat_eps: float,
    joint_eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-env ``(num_envs, buffer_len)`` masks: has the demo started moving this slot yet.

    Until the demonstration touches an object, that object sits where it was reset,
    so an un-gated tracking term would score a perfect, action-independent match for
    doing nothing. Gating restricts scoring to slots the demonstration has actually
    put in play.

    Reads the same trajectory pose cache and the same clamped demo frame that
    :class:`~.observations.ObjectTargetPoseDiff` diffs against, so the gate and the
    error always refer to one demo frame. Returns ``None`` (no gating) whenever the
    demo command stack is absent.
    """
    # Keyed on the eps triple as well as the step: the three object terms share one
    # params dict today, but a per-term override must not read another term's mask.
    step_key = (getattr(env.unwrapped, "_sim_step_counter", None), pos_eps, quat_eps, joint_eps)
    cache = getattr(env.unwrapped, "_libero_reward_in_play_cache", None)
    if cache is not None and step_key[0] is not None and cache[0] == step_key:
        return cache[1]

    if getattr(env, "command_manager", None) is None:
        return None
    try:
        term = resolve_command_term(env, command_name)
    except Exception:
        return None
    cached = getattr(term, "_traj_pose_buffer_cache", None)
    active_traj = getattr(term, "_active_traj", None)
    frame_counters = getattr(term, "_frame_counters", None)
    if not isinstance(cached, dict) or active_traj is None:
        return None
    lengths = cached.get("lengths")
    if lengths is None:
        return None
    firsts = _demo_first_move_frames(term, cached, pos_eps, quat_eps, joint_eps)
    if firsts is None:
        return None
    pose_first, joint_first = firsts

    cache_device = pose_first.device
    traj_idx = active_traj.to(device=cache_device, dtype=torch.long)
    valid = (traj_idx >= 0) & (traj_idx < pose_first.shape[0])
    traj_safe = torch.where(valid, traj_idx, torch.zeros_like(traj_idx))
    step_raw = (
        frame_counters.to(device=cache_device, dtype=torch.long)
        if frame_counters is not None
        else torch.zeros_like(traj_safe)
    )
    # ObjectTargetPoseDiff diffs against the frame one step ahead of the counter.
    next_step = torch.minimum(step_raw + 1, torch.clamp(lengths[traj_safe], min=1) - 1)
    pose_mask = (next_step[:, None] >= pose_first[traj_safe]) & valid[:, None]
    joint_mask = (next_step[:, None] >= joint_first[traj_safe]) & valid[:, None]
    result = (
        pose_mask.to(device=env.device, dtype=torch.float32),
        joint_mask.to(device=env.device, dtype=torch.float32),
    )
    if step_key[0] is not None:
        env.unwrapped._libero_reward_in_play_cache = (step_key, result)
    return result


def _shared_pose_diff(
    env: ManagerBasedRLEnv,
    command_name: str,
    include_objects: bool,
    include_targets: bool,
    use_base_frame: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """One :class:`ObjectTargetPoseDiff` compute per sim step, shared by all reward terms.

    Returns ``(diff (N, B, slot_dim), active_mask (N, B))`` or ``None``.
    """
    step_key = getattr(env.unwrapped, "_sim_step_counter", None)
    cache = getattr(env.unwrapped, "_libero_reward_pose_diff_cache", None)
    if cache is not None and step_key is not None and cache[0] == step_key:
        return cache[1]

    term = getattr(env.unwrapped, "_libero_reward_pose_diff_term", None)
    if term is None:
        term = ObjectTargetPoseDiff(ManagerTermBaseCfg(func=ObjectTargetPoseDiff, params={}), env.unwrapped)
        env.unwrapped._libero_reward_pose_diff_term = term
    diff = term(
        env,
        command_name=command_name,
        include_objects=include_objects,
        include_targets=include_targets,
        use_base_frame=use_base_frame,
        flatten=False,
    )
    if diff.ndim < 3:
        return None
    layout = term._ensure_layout(env, include_objects=include_objects, include_targets=include_targets)
    active_mask = layout_active_mask(layout, diff.shape[0], diff.shape[1], diff.device)
    result = (diff, active_mask)
    if step_key is not None:
        env.unwrapped._libero_reward_pose_diff_cache = (step_key, result)
    return result


def _pose_diff_mean_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    include_objects: bool,
    include_targets: bool,
    use_base_frame: bool,
    dim_slice: slice,
    slot_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean error over active slots for the given diff dims → ``(err, has_active)``.

    ``slot_gate`` further restricts which slots count (see :func:`_demo_slot_in_play`).
    An env with no slot left after gating reports ``has_active=0``, so the term pays
    zero rather than the perfect score an empty mean would otherwise produce.
    """
    out = _shared_pose_diff(env, command_name, include_objects, include_targets, use_base_frame)
    if out is None:
        zeros = torch.zeros(env.num_envs, device=env.device)
        return zeros, zeros
    diff, active = out
    err = torch.norm(diff[:, :, dim_slice], dim=-1)  # (N, B)
    weight = active.float()
    if slot_gate is not None:
        weight = weight * slot_gate[:, : weight.shape[1]]
    count = weight.sum(dim=1)
    mean_err = (err * weight).sum(dim=1) / count.clamp(min=1.0)
    return mean_err, (count > 0).float()


def _resolve_slot_gate(
    env: ManagerBasedRLEnv,
    command_name: str,
    gate_until_demo_moves: bool,
    move_pos_eps: float,
    move_quat_eps: float,
    move_joint_eps: float,
    which: str,
) -> torch.Tensor | None:
    """Pick the pose or joint in-play mask, or ``None`` when gating is off/unavailable."""
    if not gate_until_demo_moves:
        return None
    masks = _demo_slot_in_play(env, command_name, move_pos_eps, move_quat_eps, move_joint_eps)
    if masks is None:
        return None
    return masks[0] if which == "pose" else masks[1]


def object_position_command_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "source_action",
    include_objects: bool = True,
    include_targets: bool = True,
    use_base_frame: bool = True,
    update_interval: int = 1,  # noqa: ARG001 — kept for backward-compatible parameter parity (step cache built-in)
    gate_until_demo_moves: bool = False,
    move_pos_eps: float = 0.005,
    move_quat_eps: float = 1.0e-3,
    move_joint_eps: float = 0.01,
) -> torch.Tensor:
    """Exp kernel on the mean object POSITION error vs the demo per-step poses.

    With ``gate_until_demo_moves`` only objects the demonstration has already started
    moving are scored; see :func:`_demo_slot_in_play`.
    """
    gate = _resolve_slot_gate(
        env, command_name, gate_until_demo_moves, move_pos_eps, move_quat_eps, move_joint_eps, "pose"
    )
    err, has_active = _pose_diff_mean_error(
        env, command_name, include_objects, include_targets, use_base_frame, slice(0, 3), gate
    )
    return torch.exp(-err / std) * has_active


def object_orientation_command_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "source_action",
    include_objects: bool = True,
    include_targets: bool = True,
    use_base_frame: bool = True,
    update_interval: int = 1,  # noqa: ARG001
    gate_until_demo_moves: bool = False,
    move_pos_eps: float = 0.005,
    move_quat_eps: float = 1.0e-3,
    move_joint_eps: float = 0.01,
) -> torch.Tensor:
    """Exp kernel on the mean object ORIENTATION (euler) error vs the demo per-step poses."""
    gate = _resolve_slot_gate(
        env, command_name, gate_until_demo_moves, move_pos_eps, move_quat_eps, move_joint_eps, "pose"
    )
    err, has_active = _pose_diff_mean_error(
        env, command_name, include_objects, include_targets, use_base_frame, slice(3, 6), gate
    )
    return torch.exp(-err / std) * has_active


def articulation_joint_position_command_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "source_action",
    include_objects: bool = True,
    include_targets: bool = True,
    use_base_frame: bool = True,
    update_interval: int = 1,  # noqa: ARG001
    gate_until_demo_moves: bool = False,
    move_pos_eps: float = 0.005,
    move_quat_eps: float = 1.0e-3,
    move_joint_eps: float = 0.01,
) -> torch.Tensor:
    """Exp kernel on the mean articulation JOINT error vs the demo per-step joint states.

    Gates on demo *joint* movement rather than root-pose movement: a cabinet whose
    drawer the demo opens never changes root pose, and vice versa.
    """
    gate = _resolve_slot_gate(
        env, command_name, gate_until_demo_moves, move_pos_eps, move_quat_eps, move_joint_eps, "joint"
    )
    err, has_active = _pose_diff_mean_error(
        env, command_name, include_objects, include_targets, use_base_frame, slice(6, None), gate
    )
    return torch.exp(-err / std) * has_active
