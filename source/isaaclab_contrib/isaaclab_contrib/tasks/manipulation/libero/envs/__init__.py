# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DGPO OSC env cfg factories for LIBERO harvest training.

The suite runs on the stock :class:`~isaaclab.envs.ManagerBasedRLEnv`: the
multi-task pieces are ordinary manager terms (``task_success_rate`` curriculum
term, gripper-curriculum action term, semantic command views resolved by the
MDP helpers) — see :mod:`isaaclab_contrib.tasks.manipulation.utils.task_sr`.
"""

from .dgpo_env_cfg import (
    make_libero_dgpo_env_cfg,
    make_libero_dgpo_play_cfg,
)

__all__ = [
    "make_libero_dgpo_env_cfg",
    "make_libero_dgpo_play_cfg",
]
