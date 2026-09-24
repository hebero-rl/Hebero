# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot module definitions for the LIBERO multitask environment."""

from .franka_osc import FRANKA_OSC, LiberoFrankaOscRobotCfg

__all__ = [
    "FRANKA_OSC",
    "LiberoFrankaOscRobotCfg",
]
