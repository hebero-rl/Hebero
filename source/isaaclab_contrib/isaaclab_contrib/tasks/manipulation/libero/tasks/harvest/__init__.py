# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Harvest implementation: object-level prototype sharing (cloner + selector).

Identical object models across tasks are de-duplicated into one shared
:class:`~isaaclab.assets.AssetView` (:mod:`.prototypes`), and each task
contributes only its shared prototype assets (:mod:`.combined`); the
task-dependent MDP is gathered per env by task id (see :mod:`...mdp.combined`).
"""

from .combined import LiberoPrototypeTaskCfg, build_combined_tasks, build_prototype_cfgs
from .prototypes import (
    LIBERO_SUITES,
    ObjectBinding,
    Prototype,
    TaskBinding,
    harvest_libero_prototypes,
)
