# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quaternion order helpers for LIBERO demo / sim / policy boundaries.

Isaac Lab 3.x (Warp / PhysX / AssetView) uses scalar-last **XYZW**. Assembled
DGPO demos and policies trained on them historically used scalar-first **WXYZ**
for HDF5 ``obs/ee_states`` and proprio EE pose terms.

Use :attr:`policy_quat_order` / :attr:`demo_quat_order` on the env (or command)
cfg rather than a parallel conversion stack. Sim-side math always stays XYZW;
convert only at load and observation emit boundaries.
"""

from __future__ import annotations

from typing import Literal

import torch

import isaaclab.utils.math as math_utils

QuatOrder = Literal["wxyz", "xyzw"]

# Isaac Lab sim / math_utils / AssetView convention (not configurable).
SIM_QUAT_ORDER: QuatOrder = "xyzw"
# Default for DGPO training / assembled demos / old checkpoints.
DEFAULT_POLICY_QUAT_ORDER: QuatOrder = "wxyz"
DEFAULT_DEMO_QUAT_ORDER: QuatOrder = "wxyz"


def convert_quat(quat: torch.Tensor, *, from_order: QuatOrder, to_order: QuatOrder) -> torch.Tensor:
    """Convert quaternion ``(..., 4)`` between WXYZ and XYZW."""
    if from_order == to_order:
        return quat
    return math_utils.convert_quat(quat, to=to_order)


def convert_pose7(
    pose: torch.Tensor,
    *,
    from_order: QuatOrder,
    to_order: QuatOrder,
) -> torch.Tensor:
    """Convert pose ``(..., 7)`` = pos(3) + quat(4) between quat orders."""
    if pose.shape[-1] < 7:
        raise ValueError(f"Expected pose[..., 7], got shape {tuple(pose.shape)}.")
    if from_order == to_order:
        return pose
    out = pose.clone()
    out[..., 3:7] = convert_quat(pose[..., 3:7], from_order=from_order, to_order=to_order)
    return out


def pose7_to_sim(pose: torch.Tensor, *, from_order: QuatOrder = DEFAULT_DEMO_QUAT_ORDER) -> torch.Tensor:
    """Convert a ``(..., 7)`` pos+quat pose from the demo order (default WXYZ) to sim XYZW."""
    return convert_pose7(pose, from_order=from_order, to_order=SIM_QUAT_ORDER)


def pose7_to_policy(pose: torch.Tensor, *, to_order: QuatOrder = DEFAULT_POLICY_QUAT_ORDER) -> torch.Tensor:
    """Sim XYZW pose → policy / critic pose with the configured quat order."""
    return convert_pose7(pose, from_order=SIM_QUAT_ORDER, to_order=to_order)
