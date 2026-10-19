# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo command stack and demo-conditioned reset events.

The single demo command term registers under its cfg attribute name
(``libero_demo``) on the stock :class:`~isaaclab.managers.CommandManager`;
semantic views (``ee_pose`` / ``source_action`` / ...) are resolved through
:mod:`isaaclab_contrib.tasks.manipulation.utils.commands`.
"""

from .commands import (
    CompatCommandsCfg,
    CompatDemoTaskConfig,
    DgpoCommandsCfg,
    DgpoDemoTaskConfig,
    SourceLiberoCommand,
    SourceLiberoCommandCfg,
    make_compat_commands_cfg,
    make_dgpo_commands_cfg,
)
from .events import reset_libero_scene_to_demo_initial_state

__all__ = [
    "CompatCommandsCfg",
    "CompatDemoTaskConfig",
    "DgpoCommandsCfg",
    "DgpoDemoTaskConfig",
    "SourceLiberoCommand",
    "SourceLiberoCommandCfg",
    "make_compat_commands_cfg",
    "make_dgpo_commands_cfg",
    "reset_libero_scene_to_demo_initial_state",
]
