# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LIBERO-specific termination terms.

These complement the shared multitask terminations
(:mod:`...multitask.mdp.terminations`).  LIBERO episodes end early on task
success so that the sparse ``goal`` reward can fire once and the environment
resets; success is evaluated with the same geometric proxy used by the sparse
success reward (see :func:`..rewards.object_reached_target_mask`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab_contrib.tasks.manipulation.multitask.mdp.utils import ScatterResult, scatterable

from .rewards import object_reached_target_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


@scatterable(output_dim=0, dtype=torch.bool)
def object_reached_target(
    env: ManagerBasedRLEnv,
    *,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.1,
    height_threshold: float = 0.1,
    speed_threshold: float | None = 0.4,
) -> ScatterResult:
    """Terminate when the LIBERO goal predicate is satisfied.

    Wraps :func:`..rewards.object_reached_target_mask` as a group-aware
    termination so episodes end the moment the manipulated object reaches its
    target (placed ``on`` / ``in`` it) and comes to rest.

    Args:
        env: The environment instance.
        object_cfg: Scene entity for the manipulated object (group-scoped).
        target_object_cfg: Scene entity for the target object (group-scoped).
        xy_threshold: Planar success tolerance [m].
        height_threshold: Vertical success tolerance [m].
        speed_threshold: Maximum object speed [m/s] for success; ``None`` to skip.

    Returns:
        ``(env_ids, done)`` where ``done`` is a boolean tensor of shape
        ``(group_size,)``.
    """
    reached = object_reached_target_mask(
        env, object_cfg, target_object_cfg, xy_threshold, height_threshold, speed_threshold
    )
    return object_cfg.env_ids, reached
