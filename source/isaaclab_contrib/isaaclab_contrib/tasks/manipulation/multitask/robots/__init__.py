# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot module definitions for multitask environments.

This is the vendored subset shipped with Hebero: only the abstract
:class:`RobotModuleCfg` base is retained, since LIBERO provides its own concrete
robot module (:mod:`...libero.robots.franka`). The upstream concrete robots
(Franka / OpenArm / UR10) are not part of this extract.
"""

from ._base import RobotModuleCfg
