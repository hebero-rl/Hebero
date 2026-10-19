# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task modules for the LIBERO multitask environment.

Scenes are built by the :mod:`.harvest` implementation — object-level prototype
sharing (cloner + selector): identical object models across tasks collapse to one
shared :class:`~isaaclab.assets.AssetView`, with per-env task-gather MDP.

Genuinely shared code (the base task cfg + asset builders, and the data-driven
suite loader) lives in :mod:`.common`.

The names below are re-exported at the package root so ``from ...tasks import
build_combined_tasks`` style imports keep working.
"""

from .common import LiberoManipulationTaskCfg, resolve_reward_mode
from .harvest import build_combined_tasks
