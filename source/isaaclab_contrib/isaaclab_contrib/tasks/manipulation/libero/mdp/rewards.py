# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LIBERO-specific reward terms.

These complement the shared multitask MDP terms
(:mod:`...multitask.mdp`).  LIBERO tasks are largely relational ("put object A
onto/into object B"), so we add a group-aware object-to-object distance shaping
term in addition to the reach / lift terms borrowed from the multitask module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab_contrib.tasks.manipulation.multitask.mdp.utils import ScatterResult, scatterable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def object_reached_target_mask(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    xy_threshold: float,
    height_threshold: float,
    speed_threshold: float | None = 0.4,
) -> torch.Tensor:
    """Group-local boolean mask of the LIBERO geometric success proxy.

    A task counts as solved when the manipulated object sits within
    ``xy_threshold`` [m] of the target in the ground plane, their heights differ
    by less than ``height_threshold`` [m], and (optionally) the object is nearly
    at rest (linear speed below ``speed_threshold`` [m/s]).  This approximates
    LIBERO's ``on`` / ``in`` BDDL goal predicates from object root poses without
    a full predicate evaluator; ``open`` / ``turnon`` operation goals fall back
    to the same object-to-target proxy.

    Args:
        env: The environment instance.
        object_cfg: Scene entity for the manipulated object (group-scoped).
        target_object_cfg: Scene entity for the target object (group-scoped).
        xy_threshold: Planar success tolerance [m].
        height_threshold: Vertical success tolerance [m].
        speed_threshold: Maximum object speed [m/s] for success; ``None`` to skip
            the velocity check.

    Returns:
        A boolean tensor of shape ``(group_size,)`` over ``object_cfg.view_ids``.
    """
    obj = env.scene[object_cfg.name]
    target = env.scene[target_object_cfg.name]
    obj_pos = wp.to_torch(obj.data.root_pos_w)[object_cfg.view_ids, :3]
    target_pos = wp.to_torch(target.data.root_pos_w)[target_object_cfg.view_ids, :3]
    xy_dist = torch.linalg.norm(obj_pos[:, :2] - target_pos[:, :2], dim=-1)
    height_dist = torch.abs(obj_pos[:, 2] - target_pos[:, 2])
    reached = (xy_dist < xy_threshold) & (height_dist < height_threshold)
    if speed_threshold is not None:
        speed = torch.linalg.norm(wp.to_torch(obj.data.root_lin_vel_w)[object_cfg.view_ids, :3], dim=-1)
        reached = reached & (speed < speed_threshold)
    return reached


@scatterable(output_dim=0)
def object_reached_target_bonus(
    env: ManagerBasedRLEnv,
    *,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.1,
    height_threshold: float = 0.1,
    speed_threshold: float | None = 0.4,
) -> ScatterResult:
    """Sparse ``1.0``/``0.0`` success signal for the LIBERO goal predicate.

    Emits ``1.0`` for group rows where :func:`object_reached_target_mask` holds,
    ``0.0`` otherwise.  Multiply by a large weight (e.g. ``10.0``) for a sparse
    success reward.

    Args:
        env: The environment instance.
        object_cfg: Scene entity for the manipulated object (group-scoped).
        target_object_cfg: Scene entity for the target object (group-scoped).
        xy_threshold: Planar success tolerance [m].
        height_threshold: Vertical success tolerance [m].
        speed_threshold: Maximum object speed [m/s] for success; ``None`` to skip.

    Returns:
        ``(env_ids, reward)`` where ``reward`` has shape ``(group_size,)``.
    """
    reached = object_reached_target_mask(
        env, object_cfg, target_object_cfg, xy_threshold, height_threshold, speed_threshold
    )
    return object_cfg.env_ids, reached.float()


@scatterable
def object_to_object_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    *,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Distance shaping between a manipulated object and a target object.

    Rewards bringing ``object_cfg`` close to ``target_object_cfg`` with a
    ``1 - tanh(d / std)`` kernel, where ``d`` is the Euclidean distance [m]
    between the two object root positions.  This approximates LIBERO's common
    "place A on/in B" goals without parsing BDDL predicates.

    Args:
        env: The environment instance.
        std: Tanh kernel width [m].
        object_cfg: Scene entity for the manipulated object (group-scoped).
        target_object_cfg: Scene entity for the target object (group-scoped).

    Returns:
        ``(env_ids, reward)`` where ``reward`` has shape ``(group_size,)``.
    """
    obj_pos = wp.to_torch(env.scene[object_cfg.name].data.root_pos_w)[object_cfg.view_ids, :3]
    target_pos = wp.to_torch(env.scene[target_object_cfg.name].data.root_pos_w)[target_object_cfg.view_ids, :3]
    dist = torch.linalg.norm(obj_pos - target_pos, dim=-1)
    return object_cfg.env_ids, 1.0 - torch.tanh(dist / std)
