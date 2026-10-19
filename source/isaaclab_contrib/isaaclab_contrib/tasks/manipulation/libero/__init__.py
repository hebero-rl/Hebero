# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LIBERO manipulation suites as a trainable multi-task RL environment.

The LIBERO-Spatial and LIBERO-Goal suites share a single Franka Panda robot and
are composed into one :class:`~isaaclab.envs.ManagerBasedRLEnvCfg` via the
:class:`~...multitask.registry.MultiTaskRegistry`.
"""
