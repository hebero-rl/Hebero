# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DGPO observation terms for the harvest + CloneCfg LIBERO training path.

Term widths and concatenation order match :mod:`~...dgpo_layout`.
Scene assets are shared prototypes; per-env gather uses the deterministic
``env i → task i % n_tasks`` map from :mod:`~.combined`.

Quaternion order for EE pose terms is controlled by ``env.cfg.policy_quat_order``
(default ``wxyz`` for DGPO); sim math stays XYZW via :mod:`~.quat`.

Status
------
* **Real:** ``last_action``, multi-hot (36), ``ObjectTargetPoseBuffer`` (26×9),
  EE/joint/gripper proprio.
* **Conditional (demos):** privileged EE/joint/gripper diffs, ``demo_phase``,
  ``ObjectTargetPoseDiff``, and demo ``initial_state`` scene reset when
  :class:`~.demos.commands.SourceLiberoCommand` is installed; else zeros /
  authored prototype reset with locked widths.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg

from ...utils.commands import resolve_command, resolve_command_term
from ..dgpo_layout import (
    LIBERO_ORDERED_OBJECT_NAMES,
    LIBERO_SHARED_SUBTASK_ORDER,
    LIBERO_SHARED_SUBTASK_TAGS,
    POSE_BUFFER_JOINT_PAD,
    POSE_BUFFER_SLOT_DIM,
    encode_assignment_multi_hot,
    harvest_task_to_assignment_key,
)
from .demos.commands import SourceLiberoCommand
from .quat import DEFAULT_POLICY_QUAT_ORDER, QuatOrder, pose7_to_policy

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..tasks.harvest.prototypes import TaskBinding


def _policy_quat_order(env: ManagerBasedRLEnv) -> QuatOrder:
    """Return the validated ``env.cfg.policy_quat_order`` for emitted obs (default WXYZ)."""
    order = getattr(env.unwrapped.cfg, "policy_quat_order", DEFAULT_POLICY_QUAT_ORDER)
    return order if order in ("wxyz", "xyzw") else DEFAULT_POLICY_QUAT_ORDER


def _to_torch(data) -> torch.Tensor:
    """Convert ProxyArray / Warp / Tensor scene data to a torch tensor."""
    if isinstance(data, torch.Tensor):
        return data
    torch_attr = getattr(data, "torch", None)
    if torch_attr is not None:
        return torch_attr if isinstance(torch_attr, torch.Tensor) else torch_attr()
    return wp.to_torch(data)


def _task_bindings(env: ManagerBasedRLEnv) -> list[TaskBinding]:
    """Return the harvested task bindings from ``env.cfg.dgpo_task_bindings`` (raise if unset)."""
    bindings = getattr(env.unwrapped.cfg, "dgpo_task_bindings", None)
    if bindings is None:
        raise RuntimeError("DGPO obs terms require env.cfg.dgpo_task_bindings (set by make_libero_dgpo_env_cfg).")
    return bindings


def _env_assignment_keys(env: ManagerBasedRLEnv, bindings: list[TaskBinding]) -> list[str]:
    """Return cached per-env ``suite::id`` assignment keys via the ``env i % n_tasks`` map."""
    cache = getattr(env.unwrapped, "_compat_env_assignment_keys", None)
    n_tasks = len(bindings)
    if cache is not None and len(cache) == env.num_envs:
        return cache
    keys = [
        harvest_task_to_assignment_key(bindings[i % n_tasks].name, bindings[i % n_tasks].suite)
        for i in range(env.num_envs)
    ]
    env.unwrapped._compat_env_assignment_keys = keys
    return keys


def layout_active_mask(
    layout: dict,
    num_envs: int,
    buffer_len: int,
    device,
) -> torch.Tensor:
    """``(num_envs, buffer_len)`` bool mask of which pose-buffer columns each env uses.

    Derived from the layout's ``fill_plan``, which is fixed for a given
    ``(include_objects, include_targets)`` layout, so the mask is built once and
    cached on the layout dict rather than rebuilt every step by each consumer.

    Returned by reference; callers only read it (they multiply a diff by it), so
    no copy is made. Mutating the result would corrupt every later step.
    """
    cached = layout.get("active_mask")
    if cached is not None and cached.shape == (num_envs, buffer_len) and cached.device == torch.device(device):
        return cached
    mask = torch.zeros((num_envs, buffer_len), device=device, dtype=torch.bool)
    for _proto, col, env_ids_col, _is_artic in layout["fill_plan"]:
        mask[env_ids_col, col] = True
    layout["active_mask"] = mask
    return mask


def task_assignment_multi_hot_encoding(
    env: ManagerBasedRLEnv,
    full_sequence: bool = True,  # noqa: ARG001 — kept for playground param parity
    overlap_tags: Mapping[str, Sequence[str]] | None = None,
    shared_subtask_order: Sequence[str] | None = None,
    fallback_to_assignment: bool = True,
) -> torch.Tensor:
    """36-d multi-hot with shared subtask tags (DGPO column order).

    Built once and cached on the env. The encoding is a pure function of the
    per-env task assignment, and that assignment is fixed for the run
    (``env i -> task i % n_tasks``), so every step after the first was
    recomputing a constant: ~9.5 ms per call at 1000 envs, because the encoder
    rebuilds its tag tables on entry and then writes one GPU scalar per
    (env, tag) pair.

    Returning the cached tensor rather than a copy is safe: ObservationManager
    clones a term's output before applying noise / clipping / scaling
    (``observation_manager.py`` ``compute_group``), so nothing mutates it here.

    The cache key includes the term's parameters, since they select the tag
    vocabulary and therefore the column order; a caller that registers this term
    twice with different tags still gets the right encoding for each.
    """
    cache_key = (
        env.num_envs,
        id(overlap_tags) if overlap_tags is not None else None,
        id(shared_subtask_order) if shared_subtask_order is not None else None,
        bool(fallback_to_assignment),
    )
    cached = getattr(env.unwrapped, "_libero_task_multi_hot_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    bindings = _task_bindings(env)
    assignments = _env_assignment_keys(env, bindings)
    encoded = encode_assignment_multi_hot(
        assignments,
        device=env.device,
        overlap_tags=overlap_tags or LIBERO_SHARED_SUBTASK_TAGS,
        shared_subtask_order=shared_subtask_order or LIBERO_SHARED_SUBTASK_ORDER,
        fallback_to_assignment=fallback_to_assignment,
    )
    env.unwrapped._libero_task_multi_hot_cache = (cache_key, encoded)
    return encoded


def _ee_frame_pose_xyzw_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("franka_ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("franka_robot"),
) -> torch.Tensor:
    """EE pose in robot base frame with sim quaternion order (XYZW) → 7."""
    ee_frame = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]
    ee_pos_w = _to_torch(ee_frame.data.target_pos_w)[:, 0, :]
    ee_quat_w = _to_torch(ee_frame.data.target_quat_w)[:, 0, :]
    root_pos_w = _to_torch(robot.data.root_pos_w)
    root_quat_w = _to_torch(robot.data.root_quat_w)
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    return torch.cat((ee_pos_b, ee_quat_b), dim=-1)


def ee_frame_pose_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("franka_ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("franka_robot"),
) -> torch.Tensor:
    """EE pose in robot base frame [m, quat] → 7.

    Sim / math_utils use XYZW. The quat slice is converted to
    ``env.cfg.policy_quat_order`` (default ``wxyz`` for DGPO) at this boundary.
    """
    return pose7_to_policy(
        _ee_frame_pose_xyzw_in_base_frame(env, ee_frame_cfg, robot_cfg),
        to_order=_policy_quat_order(env),
    )


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("franka_robot"),
) -> torch.Tensor:
    """Parallel-jaw finger positions → 2 (playground sign convention)."""
    robot = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints(["panda_finger.*"], preserve_order=True)
    if len(joint_ids) < 2:
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    joint_pos = _to_torch(robot.data.joint_pos)
    left = joint_pos[:, joint_ids[0]].unsqueeze(1)
    right = (-1.0 * joint_pos[:, joint_ids[1]]).unsqueeze(1)
    return torch.cat((left, right), dim=1)


def ee_frame_pose_diff_in_base_frame(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("franka_ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("franka_robot"),
) -> torch.Tensor:
    """EE pose tracking error vs command; zeros when demo command is absent → 7.

    Internal math uses XYZW (sim + converted demo ``ee_pose``). The returned
    quat slice follows ``env.cfg.policy_quat_order`` (default ``wxyz``).
    """
    if getattr(env, "command_manager", None) is None:
        return torch.zeros((env.num_envs, 7), device=env.device, dtype=torch.float32)
    try:
        command = resolve_command(env, command_name)
    except Exception:
        return torch.zeros((env.num_envs, 7), device=env.device, dtype=torch.float32)
    if command is None or command.shape[-1] < 7:
        return torch.zeros((env.num_envs, 7), device=env.device, dtype=torch.float32)

    # Keep XYZW end-to-end for math_utils; convert only the emitted critic obs.
    ee = _ee_frame_pose_xyzw_in_base_frame(env, ee_frame_cfg=ee_frame_cfg, robot_cfg=robot_cfg)
    robot = env.scene[robot_cfg.name]
    root_pos_w = _to_torch(robot.data.root_pos_w)
    root_quat_w = _to_torch(robot.data.root_quat_w)
    des_pos_b, des_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, command[:, :3], command[:, 3:7]
    )
    pos_diff = ee[:, :3] - des_pos_b
    quat_diff = math_utils.quat_mul(ee[:, 3:7], math_utils.quat_conjugate(des_quat_b))
    return pose7_to_policy(torch.cat((pos_diff, quat_diff), dim=-1), to_order=_policy_quat_order(env))


def joint_pos_diff(
    env: ManagerBasedRLEnv,
    command_name: str = "joint_state",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"]),
) -> torch.Tensor:
    """Arm joint tracking error vs command; zeros when demo command is absent → 7."""
    asset = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
    joint_pos = _to_torch(asset.data.joint_pos)[:, joint_ids]
    if getattr(env, "command_manager", None) is None:
        return torch.zeros_like(joint_pos)
    try:
        command = resolve_command(env, command_name)
    except Exception:
        return torch.zeros_like(joint_pos)
    if command is None:
        return torch.zeros_like(joint_pos)
    dim = min(joint_pos.shape[1], command.shape[1])
    return joint_pos[:, :dim] - command[:, :dim]


def joint_vel_diff(
    env: ManagerBasedRLEnv,
    command_name: str = "joint_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"]),
) -> torch.Tensor:
    """Arm joint velocity error vs command; zeros when demo command is absent → 7.

    The velocity counterpart of :func:`joint_pos_diff`, reading the demo's
    ``joint_velocity`` view (``obs/joint_velocities`` in the assembled HDF5).  The
    command bank raises on a missing key at load time rather than substituting
    zeros, so a live command here is real demo data -- which matters more for this
    view than for the others: an all-zero velocity reference would turn
    :func:`~.demo_rewards.joint_velocity_command_error_exp` into a reward for
    standing still, the exact opposite of what it exists to do.
    """
    asset = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
    joint_vel = _to_torch(asset.data.joint_vel)[:, joint_ids]
    if getattr(env, "command_manager", None) is None:
        return torch.zeros_like(joint_vel)
    try:
        command = resolve_command(env, command_name)
    except Exception:
        return torch.zeros_like(joint_vel)
    if command is None:
        return torch.zeros_like(joint_vel)
    dim = min(joint_vel.shape[1], command.shape[1])
    return joint_vel[:, :dim] - command[:, :dim]


def demo_phase(
    env: ManagerBasedRLEnv,
    command_name: str = "source_action",
    include_start: bool = False,
) -> torch.Tensor:
    """Normalized demonstration cursor in [0, 1] → 1, or 2 with ``include_start``.

    The DeepMimic phase variable: how far along the reference the episode is,
    NOT privileged demonstration content (the reference pose itself stays in the
    critic-only diff terms).  The critic needs it because the episode horizon is
    the demo length: without a time-to-go signal, states that differ only in how
    much horizon remains are indistinguishable, and a truncation-terminal value
    target is then unlearnable rather than merely biased.  See
    :mod:`~...envs.dgpo_env_cfg` on ``is_finite_horizon``.

    ``include_start`` appends the cursor the episode was reset to, which is
    non-zero under a reserved start step or an RFCL reverse curriculum and is
    known at reset time (so a policy consuming it stays deployable).

    Returns zeros of the locked width when the demo command is absent.
    """
    width = 2 if include_start else 1
    zeros = torch.zeros(env.num_envs, width, device=env.device)
    if getattr(env, "command_manager", None) is None:
        return zeros
    try:
        term = resolve_command_term(env, command_name)
    except Exception:
        return zeros
    getter = getattr(term, "get_per_env_progress", None)
    if getter is None:
        return zeros

    phase = getter().to(device=env.device, dtype=torch.float32).view(env.num_envs, 1)
    if not include_start:
        return phase
    start_getter = getattr(term, "get_per_env_start_progress", None)
    start = (
        start_getter().to(device=env.device, dtype=torch.float32).view(env.num_envs, 1)
        if start_getter is not None
        else torch.zeros_like(phase)
    )
    return torch.cat([phase, start], dim=1)


def _compose_object_pose(
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
    base_pos: torch.Tensor | None,
    base_quat: torch.Tensor | None,
    use_base_frame: bool,
) -> torch.Tensor:
    """Compose ``(pos(3) [m], euler_xyz(3) [rad])`` object pose rows, in base or world frame.

    Input quats are sim XYZW; with ``use_base_frame`` the world pose is re-expressed in
    the robot base frame before the euler conversion. Returns ``(N, 6)``.
    """
    if use_base_frame and base_pos is not None and base_quat is not None:
        pos_b, quat_b = math_utils.subtract_frame_transforms(base_pos, base_quat, pos_w, quat_w)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_b)
        return torch.cat((pos_b, torch.stack((roll, pitch, yaw), dim=-1)), dim=-1)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_w)
    return torch.cat((pos_w, torch.stack((roll, pitch, yaw), dim=-1)), dim=-1)


class ObjectTargetPoseBuffer(ManagerTermBase):
    """Canonical 26×9 object/target pose buffer over shared harvest prototypes.

    Column order is :data:`LIBERO_ORDERED_OBJECT_NAMES`. Per env, only
    interest+target slots from that env's task are filled (playground parity).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        """Read the robot scene name from term params and defer layout construction."""
        super().__init__(cfg, env)
        params = getattr(cfg, "params", {}) or {}
        self._robot_name = str(params.get("robot_name", "franka_robot"))
        self._layout = None

    def reset(self, env_ids: Sequence[int] | torch.Tensor | int | slice | None = None) -> None:
        """No per-episode state to reset; the buffer is recomputed from live poses each call."""
        return None

    def _ensure_layout(
        self,
        env: ManagerBasedRLEnv,
        *,
        include_objects: bool,
        include_targets: bool,
    ) -> dict:
        """Build (and cache) the fill plan mapping prototypes to canonical columns.

        Resolves, per task, the active canonical column indices (interest and/or target
        object names) and, per distinct prototype, the global env ids whose column it
        fills. Cached per ``(include_objects, include_targets)`` key.
        """
        cache_key = (include_objects, include_targets)
        if self._layout is not None and self._layout.get("key") == cache_key:
            return self._layout
        bindings = _task_bindings(env)
        name_to_index = {name: idx for idx, name in enumerate(LIBERO_ORDERED_OBJECT_NAMES)}
        # canonical → prototype scene name (first registration wins; shared by construction)
        canonical_to_proto: dict[str, str] = {}
        canonical_is_articulation: dict[str, bool] = {}
        for task in bindings:
            for binding in task.object_bindings:
                if binding.canonical_name not in canonical_to_proto:
                    canonical_to_proto[binding.canonical_name] = binding.proto_name
                    canonical_is_articulation[binding.canonical_name] = binding.is_articulation

        # Per-task active column indices (interest ± targets).
        task_active: list[tuple[int, ...]] = []
        for task in bindings:
            names: list[str] = []
            if include_objects:
                names.extend(task.interest_names)
            if include_targets:
                names.extend(task.target_names)
            # Fallback: if JSON lists empty, activate primary/target via bindings.
            if not names and task.primary_proto is not None:
                for binding in task.object_bindings:
                    if binding.proto_name in {task.primary_proto, task.target_proto}:
                        names.append(binding.canonical_name)
            idxs = tuple(name_to_index[n] for n in dict.fromkeys(names) if n in name_to_index)
            task_active.append(idxs)

        n_tasks = len(bindings)
        env_task = torch.arange(env.num_envs, device=env.device, dtype=torch.long) % n_tasks
        active_mask = torch.zeros((env.num_envs, len(LIBERO_ORDERED_OBJECT_NAMES)), device=env.device, dtype=torch.bool)
        for task_idx, idxs in enumerate(task_active):
            if not idxs:
                continue
            env_rows = (env_task == task_idx).nonzero(as_tuple=True)[0]
            if env_rows.numel() == 0:
                continue
            active_mask[env_rows[:, None], torch.tensor(idxs, device=env.device, dtype=torch.long)] = True

        fill_plan: list[tuple[str, int, torch.Tensor, bool]] = []
        for canonical, proto in canonical_to_proto.items():
            col = name_to_index.get(canonical)
            if col is None:
                continue
            env_ids = active_mask[:, col].nonzero(as_tuple=True)[0]
            if env_ids.numel() == 0:
                continue
            fill_plan.append((proto, col, env_ids, canonical_is_articulation.get(canonical, False)))

        self._layout = {
            "key": cache_key,
            "fill_plan": fill_plan,
            "buffer_len": len(LIBERO_ORDERED_OBJECT_NAMES),
            "joint_dim": POSE_BUFFER_JOINT_PAD,
        }
        return self._layout

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        include_objects: bool = True,
        include_targets: bool = True,
        use_base_frame: bool = True,
        focused_only: bool = False,  # noqa: ARG002 — full 26 columns for checkpoint parity
        flatten: bool = True,
    ) -> torch.Tensor:
        """Compute the canonical object/target pose buffer from live sim state.

        Returns ``(num_envs, 26*9)`` when ``flatten`` (else ``(num_envs, 26, 9)``); per
        env only its task's interest+target columns are filled, the rest stay zero.
        Slot layout is ``[pos(3) [m], euler_xyz(3) [rad], joint pad(3) [rad|m]]``, poses
        in the robot base frame when ``use_base_frame``.
        """
        layout = self._ensure_layout(env, include_objects=include_objects, include_targets=include_targets)

        buffer_len = layout["buffer_len"]
        joint_dim = layout["joint_dim"]
        pose_buffer = torch.zeros(
            (env.num_envs, buffer_len, POSE_BUFFER_SLOT_DIM), device=env.device, dtype=torch.float32
        )
        selector = env.scene.selector
        robot = env.scene[self._robot_name]
        robot_pos = _to_torch(robot.data.root_pos_w)
        robot_quat = _to_torch(robot.data.root_quat_w)

        # Resolve the plan's selector ids once per layout. filter_reset_ids() is a
        # pure function of the selector layout and the plan's env_ids, both fixed
        # after scene construction, so doing it per obs call re-derived the same
        # ids every step -- one selector lookup per prototype per step, and each
        # lookup runs searchsorted plus two boolean-mask gathers per group.
        # Cached on the layout dict so it is invalidated whenever the layout is.
        resolved = layout.get("resolved_fill_plan")
        if resolved is None:
            resolved = []
            for proto_name, col, env_ids, is_articulation in layout["fill_plan"]:
                if proto_name not in env.scene.keys():
                    continue
                global_ids, view_ids = selector.filter_reset_ids(proto_name, env_ids)
                if global_ids.numel() == 0:
                    continue
                resolved.append((proto_name, col, global_ids, view_ids, is_articulation))
            layout["resolved_fill_plan"] = resolved

        # Transform every prototype's rows in one call rather than per prototype.
        # The frame maths is the same for all of them, so the only thing the loop
        # was buying was a smaller batch: at 26 slots that was ~26
        # subtract_frame_transforms + euler conversions per step, each on a few
        # hundred rows. The scatter targets are constant, so the concatenated
        # (env, column) index pair is cached on the layout with the plan.
        scatter = layout.get("fill_scatter_index")
        if scatter is None:
            env_index = torch.cat([g for _, _, g, _, _ in resolved]) if resolved else None
            col_index = (
                torch.cat([
                    torch.full((g.numel(),), col, dtype=torch.long, device=env.device)
                    for _, col, g, _, _ in resolved
                ])
                if resolved
                else None
            )
            scatter = (env_index, col_index)
            layout["fill_scatter_index"] = scatter
        env_index, col_index = scatter

        if env_index is not None:
            pos_w = torch.cat([_to_torch(env.scene[p].data.root_pos_w)[v, :3] for p, _, _, v, _ in resolved])
            quat_w = torch.cat([_to_torch(env.scene[p].data.root_quat_w)[v] for p, _, _, v, _ in resolved])
            pose = _compose_object_pose(
                pos_w, quat_w, robot_pos[env_index], robot_quat[env_index], use_base_frame
            )
            pose_buffer[env_index, col_index, :6] = pose

        # Joint padding stays per asset: articulations differ in joint count, so
        # there is no single width to concatenate to, and only a few slots are
        # articulations at all.
        if joint_dim > 0:
            for proto_name, col, global_ids, view_ids, is_articulation in resolved:
                if not is_articulation:
                    continue
                joint_pos = _to_torch(env.scene[proto_name].data.joint_pos)[view_ids]
                count = min(joint_pos.shape[1], joint_dim)
                pose_buffer[global_ids, col, 6 : 6 + count] = joint_pos[:, :count]

        if flatten:
            return pose_buffer.view(env.num_envs, -1)
        return pose_buffer


class ObjectTargetPoseDiff(ObjectTargetPoseBuffer):
    """Privileged object pose diff vs demo next-state (234-d).

    When :class:`~.commands.SourceLiberoCommand` is present and
    ``cache_pose_buffer_states`` is enabled, fills targets from the demo
    ``initial_state`` trajectory bank (canonical object names). Otherwise
    returns zeros with the locked width.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        """Read the demo command name / update interval defaults from term params."""
        super().__init__(cfg, env)
        params = getattr(cfg, "params", {}) or {}
        self._default_command_name = str(params.get("command_name", "source_action"))
        self._default_update_interval = int(params.get("update_interval", 1))

    def reset(self, env_ids: Sequence[int] | torch.Tensor | int | slice | None = None) -> None:
        """Invalidate the env-level pose-diff cache in addition to the base reset."""
        super().reset(env_ids=env_ids)
        if getattr(self._env.unwrapped, "_object_pose_diff_cache", None) is not None:
            setattr(self._env.unwrapped, "_object_pose_diff_cache", None)

    @staticmethod
    def _canonical_object_name(scene_key: str) -> str:
        """Strip playground ``_group_N`` suffixes / ``object_`` prefix from a demo asset key."""
        base_name = scene_key.split("/", 1)[0]
        base_name = re.sub(r"_group_\d+$", "", base_name)
        if base_name.startswith("object_"):
            base_name = base_name[len("object_") :]
        return base_name

    @staticmethod
    def _infer_state_time_len(state, infer_time_fn, fallback: int = 1) -> int:
        """Infer a trajectory state's time length via the command's helper; ``fallback`` on failure."""
        if state is None or not callable(infer_time_fn):
            return fallback
        try:
            inferred = infer_time_fn(state)
            if isinstance(inferred, (int, float)):
                return max(int(inferred), fallback)
            item_fn = getattr(inferred, "item", None)
            if callable(item_fn):
                item_val = item_fn()
                if isinstance(item_val, (int, float)):
                    return max(int(item_val), fallback)
        except Exception:
            pass
        return fallback

    def _build_traj_pose_buffer_cache(
        self,
        *,
        env: ManagerBasedRLEnv,
        term: SourceLiberoCommand,
        buffer_len: int,
        joint_dim: int,
        name_to_index: dict[str, int],
    ) -> dict[str, torch.Tensor] | None:
        """Build demo object-pose cache for privileged diffs.

        ``initial_state`` in assembled LIBERO HDF5 stores the **full** scene
        trajectory (T≈70–500), not a single frame. A naive per-timestep Python
        slice over 2000 demos is O(sum T) and can dominate env init (~60 s).
        This path fills each asset's ``(T, 7)`` pose tensor in one shot.
        """
        traj_states = getattr(term, "_traj_initial_states", None)
        if not traj_states:
            return None

        cache_device = getattr(term.cfg, "pose_buffer_cache_device", None) or env.device
        infer_time_fn = getattr(term, "_infer_state_time_length", None)
        traj_bank_lengths = getattr(term, "_traj_lengths", None)

        num_traj = len(traj_states)
        traj_lengths = torch.zeros(num_traj, dtype=torch.long, device=cache_device)
        max_time = 0
        for traj_idx, state in enumerate(traj_states):
            time_len = self._infer_state_time_len(state, infer_time_fn, fallback=1)
            if traj_bank_lengths is not None and traj_idx < traj_bank_lengths.shape[0]:
                try:
                    traj_len = int(traj_bank_lengths[traj_idx].item())
                    if traj_len > 0:
                        time_len = min(time_len, traj_len)
                except Exception:
                    pass
            traj_lengths[traj_idx] = time_len
            max_time = max(max_time, time_len)

        if max_time <= 0 or buffer_len <= 0:
            return None

        pos_cache = torch.zeros((num_traj, max_time, buffer_len, 3), device=cache_device)
        quat_cache = torch.zeros((num_traj, max_time, buffer_len, 4), device=cache_device)
        joint_cache = (
            torch.zeros((num_traj, max_time, buffer_len, joint_dim), device=cache_device) if joint_dim > 0 else None
        )
        joint_mask_cache = (
            torch.zeros((num_traj, max_time, buffer_len, joint_dim), device=cache_device, dtype=torch.bool)
            if joint_dim > 0
            else None
        )

        for traj_idx, state in enumerate(traj_states):
            if state is None:
                continue
            time_len = int(traj_lengths[traj_idx].item())
            if time_len <= 0:
                continue
            state_dict = state if isinstance(state, dict) else {}
            for entity_type in ("rigid_object", "articulation"):
                entity_states = state_dict.get(entity_type, {})
                if not isinstance(entity_states, dict):
                    continue
                for asset_name, asset_state in entity_states.items():
                    if not isinstance(asset_state, dict):
                        continue
                    # Skip robot articulation — only objects / cabinets.
                    if entity_type == "articulation" and asset_name in {"robot", "franka_robot"}:
                        continue
                    root_pose = asset_state.get("root_pose")
                    if root_pose is None or not torch.is_tensor(root_pose):
                        continue
                    canonical_name = self._canonical_object_name(asset_name)
                    name_idx = name_to_index.get(canonical_name)
                    if name_idx is None:
                        continue
                    pose = root_pose
                    if pose.ndim == 1:
                        pose = pose.unsqueeze(0)
                    if pose.ndim != 2 or pose.shape[-1] < 7:
                        continue
                    t_use = min(time_len, int(pose.shape[0]))
                    if t_use <= 0:
                        continue
                    pose_dev = pose[:t_use].to(device=cache_device, dtype=pos_cache.dtype)
                    pos_cache[traj_idx, :t_use, name_idx] = pose_dev[:, :3]
                    quat_cache[traj_idx, :t_use, name_idx] = pose_dev[:, 3:7]
                    if joint_dim > 0 and entity_type == "articulation" and joint_cache is not None:
                        joint_pos = asset_state.get("joint_position")
                        if joint_pos is not None and torch.is_tensor(joint_pos):
                            joints = joint_pos.unsqueeze(0) if joint_pos.ndim == 1 else joint_pos
                            if joints.ndim != 2:
                                continue
                            t_j = min(t_use, int(joints.shape[0]))
                            count = min(int(joints.shape[1]), joint_dim)
                            joint_cache[traj_idx, :t_j, name_idx, :count] = joints[:t_j, :count].to(
                                device=cache_device, dtype=joint_cache.dtype
                            )
                            if joint_mask_cache is not None:
                                joint_mask_cache[traj_idx, :t_j, name_idx, :count] = True

        return {
            "pos": pos_cache,
            "quat": quat_cache,
            "joint": joint_cache,
            "joint_mask": joint_mask_cache,
            "lengths": traj_lengths,
        }

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "source_action",
        include_objects: bool = True,
        include_targets: bool = True,
        use_base_frame: bool = True,
        focused_only: bool = False,
        flatten: bool = True,
        update_interval: int = 1,
    ) -> torch.Tensor:
        """Compute the live pose buffer minus the demo's next-step pose targets.

        Targets come from the per-trajectory ``initial_state`` cache at each env's demo
        frame counter + 1; inactive columns are zeroed. Returns ``(num_envs, 26*9)``
        when ``flatten`` (else ``(num_envs, 26, 9)``); degrades to zeros when the demo
        command / trajectory cache is absent.
        """
        pose_buffer = super().__call__(
            env,
            include_objects=include_objects,
            include_targets=include_targets,
            use_base_frame=use_base_frame,
            focused_only=focused_only,
            flatten=False,
        )
        target_buffer = pose_buffer.new_zeros(pose_buffer.shape)
        zeros = target_buffer.view(env.num_envs, -1) if flatten else target_buffer

        if getattr(env, "command_manager", None) is None:
            return zeros
        try:
            term = resolve_command_term(env, command_name)
        except Exception:
            return zeros
        if not isinstance(term, SourceLiberoCommand):
            return zeros
        if not getattr(getattr(term, "cfg", None), "cache_pose_buffer_states", False):
            return zeros

        layout = self._ensure_layout(env, include_objects=include_objects, include_targets=include_targets)
        name_to_index = {name: idx for idx, name in enumerate(LIBERO_ORDERED_OBJECT_NAMES)}
        buffer_len = int(pose_buffer.shape[1])
        joint_dim = int(layout["joint_dim"])

        traj_cache_key = (layout.get("key"), buffer_len, joint_dim)
        cached = getattr(term, "_traj_pose_buffer_cache", None)
        cached_key = getattr(term, "_traj_pose_buffer_cache_key", None)
        if cached is None or cached_key != traj_cache_key:
            cached = self._build_traj_pose_buffer_cache(
                env=env,
                term=term,
                buffer_len=buffer_len,
                joint_dim=joint_dim,
                name_to_index=name_to_index,
            )
            term._traj_pose_buffer_cache = cached
            term._traj_pose_buffer_cache_key = traj_cache_key if cached is not None else None
        if cached is None:
            return zeros

        pos_cache = cached.get("pos")
        quat_cache = cached.get("quat")
        lengths = cached.get("lengths")
        active_traj = getattr(term, "_active_traj", None)
        frame_counters = getattr(term, "_frame_counters", None)
        if pos_cache is None or quat_cache is None or lengths is None or active_traj is None:
            return zeros

        # Active columns from layout fill plan (canonical indices that this scene uses).
        active_cols = sorted({col for _, col, _, _ in layout["fill_plan"]})
        if not active_cols:
            return zeros
        name_idx = torch.tensor(active_cols, device=env.device, dtype=torch.long)
        cache_device = pos_cache.device
        name_idx_cache = name_idx.to(device=cache_device)

        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        traj_idx = active_traj[env_ids]
        valid = (traj_idx >= 0) & (traj_idx < pos_cache.shape[0])
        if not torch.any(valid):
            return zeros
        env_valid = env_ids[valid]
        traj_valid = traj_idx[valid].to(device=cache_device)

        step_raw = (
            frame_counters[env_valid].to(device=cache_device)
            if frame_counters is not None
            else torch.zeros_like(traj_valid, device=cache_device)
        )
        # Match playground: look one demo step ahead of the (post-compute) frame counter.
        next_step = torch.minimum(step_raw + 1, torch.clamp(lengths[traj_valid], min=1) - 1)

        pos_local = pos_cache[traj_valid, next_step].index_select(1, name_idx_cache).to(device=env.device)
        quat_w = quat_cache[traj_valid, next_step].index_select(1, name_idx_cache).to(device=env.device)

        robot = env.scene[self._robot_name]
        robot_pos = _to_torch(robot.data.root_pos_w)[env_valid]
        robot_quat = _to_torch(robot.data.root_quat_w)[env_valid]
        origin = env.scene.env_origins[env_valid, 0:3]
        pos_w = pos_local + origin[:, None, :]
        n_valid, count = pos_w.shape[0], pos_w.shape[1]
        if use_base_frame:
            base_pos_rep = robot_pos.repeat_interleave(count, dim=0)
            base_quat_rep = robot_quat.repeat_interleave(count, dim=0)
            pose_flat = _compose_object_pose(
                pos_w.reshape(-1, 3), quat_w.reshape(-1, 4), base_pos_rep, base_quat_rep, True
            )
        else:
            pose_flat = _compose_object_pose(pos_w.reshape(-1, 3), quat_w.reshape(-1, 4), None, None, False)
        pose_vec = pose_flat.view(n_valid, count, -1)
        end_dim = min(pose_vec.shape[-1], target_buffer.shape[-1])
        target_buffer[env_valid[:, None], name_idx[None, :], :end_dim] = pose_vec[:, :, :end_dim]

        if joint_dim > 0 and cached.get("joint") is not None:
            joint_values = cached["joint"][traj_valid, next_step].index_select(1, name_idx_cache).to(device=env.device)
            target_buffer[env_valid[:, None], name_idx[None, :], 6 : 6 + joint_dim] = joint_values

        # Only diff columns that are active for each env (pose_buffer already zeros inactive).
        diff = pose_buffer - target_buffer
        # Zero out inactive columns so unused slots do not inject spurious critic signal.
        active_mask = layout_active_mask(layout, env.num_envs, buffer_len, env.device)
        diff = diff * active_mask.unsqueeze(-1).to(dtype=diff.dtype)

        # update_interval kept for playground API parity (caching optional later).
        _ = update_interval
        return diff.view(env.num_envs, -1) if flatten else diff


# Remaining polish items for the DGPO harvest training path (not blockers).
DGPO_OBS_NOTES: tuple[str, ...] = (
    "Out of env scope: DAPG/ABC importance weights (algorithm / runner IW from SourceLiberoCommand metrics)",
    "Optional: gravity-warmup / mid-demo grasp open-offset polish from playground RLfD events",
)
# Backward-compatible alias.
COMPAT_OBS_TODOS = DGPO_OBS_NOTES
