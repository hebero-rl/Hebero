# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Externally contributed Isaac Lab task environments (Hebero extract).

Importing this package registers the contributed gym environments. ``train.py`` /
``play.py`` import ``isaaclab_contrib.tasks`` for exactly this side effect. This
standalone extract ships the LIBERO multi-task environments.
"""

# Importing each leaf config package runs its ``gym.register`` calls.
from .manipulation.libero.config import franka  # noqa: F401
