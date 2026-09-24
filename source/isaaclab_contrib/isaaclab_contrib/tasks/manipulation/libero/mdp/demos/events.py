# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo-conditioned reset events for the LIBERO harvest + DGPO training path.

On reset, objects / articulations / robot joints are restored from the demo
``initial_state`` of the trajectory reserved by
:class:`~.commands.SourceLiberoCommand` (same traj the privileged diffs track).

Asset names in the HDF5 bank are playground-canonical (``alphabet_soup_1``,
``robot``). Harvest scenes use shared prototype names (``akita_black_bowl``,
``franka_robot``). Mapping uses each env's :class:`TaskBinding` plus
``selector.filter_reset_ids`` for view-local writes.

Demo poses are loaded already translated by harvest ``workspace_shift``
(``ROBOT_BASE_KITCHEN - task_robot_base``) inside the command bank so they
match raised fixtures; this module only adds ``env_origins`` for world writes.

Quaternion order: assembled demos store ``root_pose`` as WXYZ. Legacy HDF5
loads convert them to XYZW via :class:`~isaaclab.utils.datasets.HDF5DatasetFileHandler`
before this module writes through ``write_root_pose_to_sim_index``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ....utils.commands import resolve_command_term

from .commands import SourceLiberoCommand, TrajectoryState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ...tasks.harvest.prototypes import TaskBinding


logger = logging.getLogger(__name__)

_ROBOT_DEMO_NAMES = frozenset({"robot", "franka_robot"})
_ROBOT_SCENE_NAME = "franka_robot"
# Env-local z below this after workspace shift usually means the demo pose was
# not aligned with harvest kitchen-base normalisation (objects sit on the floor).
_SUSPICIOUS_ENV_LOCAL_Z_M = 0.05


def canonical_demo_asset_name(asset_name: str) -> str:
    """Strip playground group suffixes / ``object_`` prefix from a demo asset key."""
    base_name = asset_name.split("/", 1)[0]
    base_name = re.sub(r"_group_\d+$", "", base_name)
    if base_name.startswith("object_"):
        base_name = base_name[len("object_") :]
    return base_name


def build_canonical_to_proto_map(binding: TaskBinding) -> dict[str, str]:
    """Map demo-canonical object names → harvest prototype scene names for one task."""
    out: dict[str, str] = {}
    for obj in binding.object_bindings:
        out[obj.canonical_name] = obj.proto_name
        # Also accept already-canonicalised / stripped keys.
        out[canonical_demo_asset_name(obj.canonical_name)] = obj.proto_name
    return out


def resolve_demo_asset_to_scene(
    asset_name: str,
    *,
    entity_type: str,
    canonical_to_proto: dict[str, str],
) -> str | None:
    """Resolve a demo ``initial_state`` asset key to a harvest scene entity name.

    Args:
        asset_name: Key under ``initial_state/{rigid_object|articulation}/``.
        entity_type: ``\"rigid_object\"`` or ``\"articulation\"``.
        canonical_to_proto: Per-task map from :func:`build_canonical_to_proto_map`.

    Returns:
        Scene entity name, or ``None`` if the asset is not present in this task.
    """
    if entity_type == "articulation" and asset_name in _ROBOT_DEMO_NAMES:
        return _ROBOT_SCENE_NAME
    canonical = canonical_demo_asset_name(asset_name)
    if asset_name in canonical_to_proto:
        return canonical_to_proto[asset_name]
    if canonical in canonical_to_proto:
        return canonical_to_proto[canonical]
    # Direct hit when demos already store harvest prototype names.
    if asset_name in canonical_to_proto.values():
        return asset_name
    return None


def _resolve_env_ids(env_ids_arg, total_envs: int, device: torch.device) -> torch.Tensor:
    """Normalise ``None`` / slice / tensor / iterable / int env ids to a long tensor."""
    if env_ids_arg is None:
        return torch.arange(total_envs, device=device, dtype=torch.long)
    if isinstance(env_ids_arg, slice):
        start, stop, step = env_ids_arg.indices(total_envs)
        return torch.arange(start, stop, step, device=device, dtype=torch.long)
    if torch.is_tensor(env_ids_arg):
        return env_ids_arg.to(device=device, dtype=torch.long)
    if hasattr(env_ids_arg, "__iter__"):
        return torch.tensor([int(v) for v in env_ids_arg], device=device, dtype=torch.long)
    return torch.tensor([int(env_ids_arg)], device=device, dtype=torch.long)


def _task_bindings(env: ManagerBasedEnv) -> list[TaskBinding]:
    """Return the harvested task bindings from ``env.cfg.dgpo_task_bindings`` (raise if unset)."""
    bindings = getattr(env.cfg, "dgpo_task_bindings", None)
    if bindings is None:
        raise RuntimeError("Demo reset requires env.cfg.dgpo_task_bindings (set by make_libero_dgpo_env_cfg).")
    return bindings


def _squeeze_pose(root_pose: torch.Tensor) -> torch.Tensor:
    """Flatten a demo ``root_pose`` (possibly ``(1, 7)`` time-sliced) to a 1-D pose vector."""
    pose = root_pose
    if pose.ndim >= 2 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.ndim > 1:
        pose = pose.reshape(-1)
    return pose


def _resolve_robot_arm_gripper_ids(env: ManagerBasedEnv, articulation) -> tuple[list[int], list[int]]:
    """Resolve Franka arm / gripper joint ids for demo joint writes.

    Cached on the env. The fallback path runs two regex ``find_joints`` scans, and
    this used to be called once per env per reset; the joint layout is a property
    of the articulation, so it is resolved once per run.
    """
    cached = getattr(env, "_libero_demo_robot_joint_ids", None)
    if cached is not None:
        return cached

    arm_joint_ids: list[int] = []
    gripper_joint_ids: list[int] = []

    action_manager = getattr(env, "action_manager", None)
    action_terms = getattr(action_manager, "_terms", None)
    if action_terms is not None:
        arm_action = action_terms.get("arm") or action_terms.get("arm_action")
        runtime_arm = getattr(arm_action, "_joint_ids", None) if arm_action is not None else None
        if runtime_arm is not None and not isinstance(runtime_arm, slice):
            arm_joint_ids = [int(j) for j in runtime_arm]
        gripper_action = action_terms.get("gripper") or action_terms.get("gripper_action")
        runtime_grip = getattr(gripper_action, "_joint_ids", None) if gripper_action is not None else None
        if runtime_grip is not None and not isinstance(runtime_grip, slice):
            gripper_joint_ids = [int(j) for j in runtime_grip]

    if not arm_joint_ids:
        found, _ = articulation.find_joints(["panda_joint.*"], preserve_order=True)
        arm_joint_ids = [int(j) for j in found]
    if not gripper_joint_ids:
        found, _ = articulation.find_joints(["panda_finger.*"], preserve_order=True)
        gripper_joint_ids = [int(j) for j in found]
    resolved = (arm_joint_ids, gripper_joint_ids)
    env._libero_demo_robot_joint_ids = resolved
    return resolved


def _stage_dense(
    env: ManagerBasedEnv,
    rows: list[tuple[int, torch.Tensor]],
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage ``(env_idx, vector)`` rows into a dense ``(num_envs, width)`` buffer.

    Returns ``(dense, candidate_env_ids)``. Dense staging matters for correctness:
    :meth:`filter_reset_ids` may return the matched env ids reordered or subset
    relative to the candidates handed to it, so the caller must be able to index
    values by *global env id* rather than by position in ``rows``.

    Rows are grouped by vector width so ragged inputs still cost one ``stack``
    per distinct width (in practice one) instead of one copy per row.
    """
    dense = torch.zeros(env.scene.num_envs, width, device=env.device)
    by_width: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for env_idx, vec in rows:
        by_width.setdefault(int(vec.shape[-1]), []).append((env_idx, vec))
    for vec_width, group in by_width.items():
        ids = torch.tensor([e for e, _ in group], dtype=torch.long, device=env.device)
        values = torch.stack([v for _, v in group]).to(device=env.device, dtype=dense.dtype)
        copy_dim = min(vec_width, width)
        dense[ids, :copy_dim] = values[:, :copy_dim]
    candidates = torch.tensor(sorted({e for e, _ in rows}), dtype=torch.long, device=env.device)
    return dense, candidates


def _flush_pose_writes(env: ManagerBasedEnv, pose_rows: dict[str, list[tuple[int, torch.Tensor]]]) -> None:
    """One ``filter_reset_ids`` + one root-pose/velocity write per scene asset.

    Demo poses are env-local (XYZW, harvest kitchen frame after the command-bank
    workspace shift); the env origin is added before the world write. The
    unshifted-z diagnostic is kept but evaluated once per asset over all its envs
    instead of once per env, so it costs one device sync rather than ``num_envs``.
    """
    if not pose_rows:
        return
    selector = env.scene.selector
    scene_names = env.scene.keys()
    for asset_name, rows in pose_rows.items():
        if not rows or asset_name not in scene_names:
            continue
        dense, candidates = _stage_dense(env, rows, 7)
        global_ids, view_ids = selector.filter_reset_ids(asset_name, candidates)
        if global_ids.numel() == 0:
            continue
        poses = dense[global_ids].clone()
        suspicious = poses[:, 2] < _SUSPICIOUS_ENV_LOCAL_Z_M
        if bool(torch.any(suspicious)):
            logger.warning(
                "Demo reset wrote %s for %d/%d env(s) with env-local z < %.2f m. "
                "Check workspace_shift alignment (objects may fall to the floor).",
                asset_name,
                int(suspicious.sum()),
                int(poses.shape[0]),
                _SUSPICIOUS_ENV_LOCAL_Z_M,
            )
        poses[:, :3] += env.scene.env_origins[global_ids]
        asset = env.scene[asset_name]
        asset.write_root_pose_to_sim_index(root_pose=poses, env_ids=view_ids)
        asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((view_ids.shape[0], 6), device=env.device), env_ids=view_ids
        )


def _flush_joint_writes(
    env: ManagerBasedEnv,
    joint_rows: dict[str, tuple[bool, list[tuple[int, torch.Tensor]]]],
    *,
    robot_joint_noise_std: float,
) -> None:
    """One ``filter_reset_ids`` + one joint write per articulation.

    Robot path writes arm + gripper joints (positions and targets), with optional
    Gaussian arm noise clamped to the soft joint limits. Non-robot articulations
    (cabinet / stove) pad or truncate the demo joints to the asset's joint count.

    Demo ``joint_velocity`` is deliberately not read: both the robot and the
    non-robot path reset velocities to zero, which is what the per-env version
    did too (it computed a ``jv`` it never used).
    """
    if not joint_rows:
        return
    selector = env.scene.selector
    scene_names = env.scene.keys()
    for asset_name, (is_robot, rows) in joint_rows.items():
        if not rows or asset_name not in scene_names:
            continue
        width = max(int(vec.shape[-1]) for _, vec in rows)
        dense, candidates = _stage_dense(env, rows, width)
        global_ids, view_ids = selector.filter_reset_ids(asset_name, candidates)
        if global_ids.numel() == 0:
            continue
        asset = env.scene[asset_name]
        jp = dense[global_ids]
        if is_robot:
            _write_robot_joints(env, asset, jp, view_ids, robot_joint_noise_std)
        else:
            _write_object_joints(asset, jp, view_ids)


def _write_robot_joints(env, asset, jp: torch.Tensor, view_ids: torch.Tensor, noise_std: float) -> None:
    """Write batched arm (+ gripper) joint state and targets for the demo robot."""
    arm_ids, grip_ids = _resolve_robot_arm_gripper_ids(env, asset)
    arm_dim = len(arm_ids)
    if arm_dim == 0 or jp.shape[-1] < arm_dim:
        return
    arm_pos = jp[:, :arm_dim].clone()
    arm_vel = torch.zeros_like(arm_pos)
    if noise_std > 0.0:
        arm_pos = arm_pos + torch.randn_like(arm_pos) * noise_std
        import warp as wp

        soft = asset.data.soft_joint_pos_limits
        soft_t = soft if isinstance(soft, torch.Tensor) else wp.to_torch(soft)
        lim = soft_t[view_ids][:, arm_ids]
        arm_pos = arm_pos.clamp_(lim[..., 0], lim[..., 1])
    asset.write_joint_position_to_sim_index(position=arm_pos, joint_ids=arm_ids, env_ids=view_ids)
    asset.write_joint_velocity_to_sim_index(velocity=arm_vel, joint_ids=arm_ids, env_ids=view_ids)
    asset.set_joint_position_target_index(target=arm_pos, joint_ids=arm_ids, env_ids=view_ids)
    asset.set_joint_velocity_target_index(target=arm_vel, joint_ids=arm_ids, env_ids=view_ids)
    if grip_ids and jp.shape[-1] >= arm_dim + len(grip_ids):
        grip_pos = jp[:, arm_dim : arm_dim + len(grip_ids)].clone().abs()
        grip_vel = torch.zeros_like(grip_pos)
        asset.write_joint_position_to_sim_index(position=grip_pos, joint_ids=grip_ids, env_ids=view_ids)
        asset.write_joint_velocity_to_sim_index(velocity=grip_vel, joint_ids=grip_ids, env_ids=view_ids)
        asset.set_joint_position_target_index(target=grip_pos, joint_ids=grip_ids, env_ids=view_ids)


def _write_object_joints(asset, jp: torch.Tensor, view_ids: torch.Tensor) -> None:
    """Write batched joint state for a non-robot articulation (cabinet / stove)."""
    import warp as wp

    default_jp = asset.data.default_joint_pos
    default_t = default_jp if isinstance(default_jp, torch.Tensor) else wp.to_torch(default_jp)
    n_asset = int(default_t.shape[-1])
    out_pos = default_t[view_ids].clone()
    count = min(int(jp.shape[-1]), n_asset)
    out_pos[:, :count] = jp[:, :count]
    zero_vel = torch.zeros_like(out_pos)
    asset.write_joint_position_to_sim_index(position=out_pos, env_ids=view_ids)
    asset.write_joint_velocity_to_sim_index(velocity=zero_vel, env_ids=view_ids)
    asset.set_joint_position_target_index(target=out_pos, env_ids=view_ids)


def _collect_demo_writes(
    requested: torch.Tensor,
    reservations: dict[int, TrajectoryState | None],
    bindings: list[TaskBinding],
    n_tasks: int,
    *,
    include_articulations: bool,
    include_rigid_objects: bool,
) -> tuple[dict[str, list[tuple[int, torch.Tensor]]], dict[str, tuple[bool, list[tuple[int, torch.Tensor]]]]]:
    """Bucket every env's demo ``initial_state`` by scene asset.

    Pure Python/tensor-view bookkeeping — deliberately issues no CUDA work, so the
    only device traffic in a demo reset is the batched writes that follow.

    Returns ``(pose_rows, joint_rows)`` mapping scene entity name to the
    ``(env_idx, vector)`` rows destined for it; ``joint_rows`` values also carry
    the ``is_robot`` flag, which is a property of the scene entity, not the env.
    """
    proto_maps: dict[int, dict[str, str]] = {}
    pose_rows: dict[str, list[tuple[int, torch.Tensor]]] = {}
    joint_rows: dict[str, tuple[bool, list[tuple[int, torch.Tensor]]]] = {}

    def add_pose(scene_name: str, env_idx: int, root_pose: torch.Tensor) -> None:
        pose = _squeeze_pose(root_pose)
        if pose.shape[-1] < 7:
            return
        pose_rows.setdefault(scene_name, []).append((env_idx, pose[:7]))

    def add_joints(scene_name: str, env_idx: int, joint_position: torch.Tensor, is_robot: bool) -> None:
        jp = joint_position
        if jp.ndim >= 2 and jp.shape[0] == 1:
            jp = jp[0]
        if jp.ndim > 1:
            jp = jp.reshape(-1)
        if jp.numel() == 0:
            return
        joint_rows.setdefault(scene_name, (is_robot, []))[1].append((env_idx, jp))

    for raw_env_idx in requested.tolist():
        env_idx = int(raw_env_idx)
        state = reservations.get(env_idx)
        if state is None:
            continue
        task_idx = env_idx % n_tasks
        canonical_to_proto = proto_maps.get(task_idx)
        if canonical_to_proto is None:
            canonical_to_proto = build_canonical_to_proto_map(bindings[task_idx])
            proto_maps[task_idx] = canonical_to_proto

        if include_articulations and "articulation" in state:
            for asset_name, asset_state in state["articulation"].items():
                if not isinstance(asset_state, dict):
                    continue
                scene_name = resolve_demo_asset_to_scene(
                    asset_name, entity_type="articulation", canonical_to_proto=canonical_to_proto
                )
                if scene_name is None:
                    continue
                is_robot = scene_name == _ROBOT_SCENE_NAME
                root_pose = asset_state.get("root_pose")
                # Keep robot base from scene default; only joints come from the demo.
                if not is_robot and root_pose is not None and torch.is_tensor(root_pose):
                    add_pose(scene_name, env_idx, root_pose)
                joint_position = asset_state.get("joint_position")
                if joint_position is not None and torch.is_tensor(joint_position):
                    add_joints(scene_name, env_idx, joint_position, is_robot)

        if include_rigid_objects and "rigid_object" in state:
            for asset_name, asset_state in state["rigid_object"].items():
                if not isinstance(asset_state, dict):
                    continue
                scene_name = resolve_demo_asset_to_scene(
                    asset_name, entity_type="rigid_object", canonical_to_proto=canonical_to_proto
                )
                if scene_name is None:
                    continue
                root_pose = asset_state.get("root_pose")
                if root_pose is not None and torch.is_tensor(root_pose):
                    add_pose(scene_name, env_idx, root_pose)

    return pose_rows, joint_rows


def reset_libero_scene_to_demo_initial_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | Sequence[int] | None,
    command_name: str = "source_action",
    include_articulations: bool = True,
    include_rigid_objects: bool = True,
    robot_joint_noise_std: float = 0.0,
) -> None:
    """Restore scene entities from the demo ``initial_state`` of the reserved traj.

    Must run in ``mode=\"reset\"`` **before** :meth:`CommandManager.reset` so that
    :meth:`~.commands.SourceLiberoCommand.reserve_trajectories_for_envs` can pin
    the same trajectory the privileged diffs will advance.
    """
    if not command_name:
        raise ValueError("Parameter 'command_name' must be provided.")
    if getattr(env, "command_manager", None) is None:
        return
    try:
        term = resolve_command_term(env, command_name)
    except Exception:
        import warnings

        warnings.warn(
            f"Demo-init reset skipped: command '{command_name}' not resolvable — episodes will NOT"
            " start from demo initial states (objects keep their current poses).",
            stacklevel=2,
        )
        return
    if not isinstance(term, SourceLiberoCommand):
        return

    requested = _resolve_env_ids(env_ids, env.scene.num_envs, env.device)
    if requested.numel() == 0:
        return

    bindings = _task_bindings(env)
    n_tasks = len(bindings)
    reservations = term.reserve_trajectories_for_envs(requested)

    # Bucket every env's demo state by scene asset first, then issue ONE write per
    # asset. The previous shape -- a filter + write per (env, asset) -- cost
    # O(num_envs * assets) tiny kernel launches plus a device sync per env, and
    # scaled linearly with num_envs while the sim itself scales sublinearly, so it
    # absorbed the throughput the heterogeneous cloner was meant to buy.
    pose_rows, joint_rows = _collect_demo_writes(
        requested,
        reservations,
        bindings,
        n_tasks,
        include_articulations=include_articulations,
        include_rigid_objects=include_rigid_objects,
    )
    _flush_pose_writes(env, pose_rows)
    _flush_joint_writes(env, joint_rows, robot_joint_noise_std=robot_joint_noise_std)
