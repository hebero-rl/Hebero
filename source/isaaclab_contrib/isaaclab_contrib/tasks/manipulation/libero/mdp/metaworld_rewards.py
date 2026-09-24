# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metaworld-style goal-predicate dense reward + articulation goal predicates.

Port of the playground ``goal_predicate_dense_reward`` (Hamacher-product
composition) onto the harvest/Cloner API. Each task's raw BDDL goal predicates
(:attr:`TaskBinding.goals`) drive the shaping directly — no demos needed:

* **Relationship goals** (``in``/``on``): ``reach`` x ``place`` composed via the
  Hamacher soft-AND, plus a ``lift`` bonus tied to the object's initial z
  (gated by reach so gravity can't credit it). Honors per-goal ``height_diff``
  / ``orientation`` offsets and sub-body targets (``flat_stove_1/burnerplate``).
* **Articulation operation goals** (``open``/``close``/``turnon``):
  ``approach`` (EE to handle/knob body) x joint ``progress`` toward the
  operation's joint range.

This module holds only the reward. The goal-operand geometry it shapes over,
and the binary success predicates, live in :mod:`.combined` -- they are shared
with the task-success proxy and are not specific to this reward.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from ..tasks.harvest.prototypes import TaskBinding  # noqa: TC001
from .combined import (
    _articulation_joint_pos,
    _goal_target_pos,
    _locate,
    _resolve_articulation_joint_spec,
    _resolve_entity_pos,
    _resolve_tasks,
    _runtime,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import _to_torch

__all__ = ["metaworld_dense_reward", "hamacher_product"]



def hamacher_product(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft-AND of two sub-rewards in [0, 1]: ``a*b / (a + b - a*b + eps)``."""
    return (a * b) / (a + b - a * b + eps)


def _find_handle_body_index(asset, goal: dict, joint_pattern: str) -> int | None:
    """Index of the most likely handle/knob/door body of an articulation.

    LIBERO articulations don't follow ``joint_name == body_name`` (wooden
    cabinet: joint ``middle_level`` but body ``wooden_cabinet_middle_handle``);
    search ``body_names`` with prioritised regexes: layer-specific handle →
    joint-specific handle → layer stem → joint stem → generic
    handle/knob/door/drawer. ``None`` lets the caller fall back to the root.
    """
    body_names = list(getattr(asset, "body_names", []) or [])
    if not body_names:
        return None
    layer = goal.get("layer") or ""
    joint_stem = joint_pattern.replace("_level", "")
    candidates: list[str] = []
    if layer:
        candidates.append(rf"(?i).*{re.escape(layer)}.*handle.*")
    candidates.append(rf"(?i).*{re.escape(joint_pattern)}.*handle.*")
    if joint_stem != joint_pattern:
        candidates.append(rf"(?i).*{re.escape(joint_stem)}.*handle.*")
    if layer:
        candidates.append(rf"(?i).*{re.escape(layer)}.*")
    candidates.append(rf"(?i).*{re.escape(joint_pattern)}.*")
    if joint_stem != joint_pattern:
        candidates.append(rf"(?i).*{re.escape(joint_stem)}.*")
    candidates.extend([r"(?i).*handle.*", r"(?i).*knob.*", r"(?i).*door.*", r"(?i).*drawer.*"])
    for pattern in candidates:
        regex = re.compile(pattern)
        for i, bn in enumerate(body_names):
            if regex.match(bn):
                return i
    return None



# ---------------------------------------------------------------------------
# Metaworld-style dense reward
# ---------------------------------------------------------------------------


class metaworld_dense_reward(ManagerTermBase):
    """Metaworld-style dense reward driven by each task's BDDL goal predicates.

    Relationship goals compose ``reach`` x ``place`` via the Hamacher product
    plus a reach-gated ``lift`` bonus (relative to the object's initial z,
    re-snapshotted on reset). Articulation operations compose ``approach``
    (EE to handle body) x joint ``progress``. Each env's total is the mean over
    its task's goals, scaled by ``scale`` — per-step range is roughly
    ``[0, scale * (1 + lift_weight)]``. Returns ``(num_envs,)``.

    Args (via ``RewTerm.params``):
        std_reach: tanh width [m] for the reach/approach distance.
        std_place: tanh width [m] for place / translational joint progress.
        std_joint: tanh width [rad] for rotational joint progress (used when
            the joint range exceeds 0.3 — microwave door, stove knob).
        scale: final multiplier (10.0 matches Metaworld's typical range).
        lift_weight: weight of the reach-gated lift bonus.
        lift_threshold: rise [m] above initial z that saturates the lift bonus.
        ee_frame_name: scene key of the EE FrameTransformer.
    """

    def __init__(self, cfg, env):
        """Allocate the per-(task, object) initial-z buffers and the reset-refresh mask."""
        super().__init__(cfg, env)
        # (task_idx, obj_name) -> (n_task_envs,) initial z; refreshed per env on reset.
        self._init_obj_z: dict[tuple[int, str], torch.Tensor] = {}
        self._needs_refresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | int | slice | None = None) -> None:
        """Mark the given envs (or all) so their initial-z snapshots refresh on the next call."""
        if env_ids is None:
            self._needs_refresh.fill_(True)
            return
        if hasattr(env_ids, "__len__") and len(env_ids) == 0:
            return
        self._needs_refresh[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std_reach: float = 0.3,
        std_place: float = 0.2,
        std_joint: float = 0.5,
        scale: float = 10.0,
        lift_weight: float = 0.5,
        lift_threshold: float = 0.05,
        ee_frame_name: str = "franka_ee_frame",
    ) -> torch.Tensor:
        """Shape every env's reward from its task's goal predicates (see class docstring)."""
        tasks = _resolve_tasks(env, None)
        rt = _runtime(env, tasks)
        rewards = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

        ee_frame = env.scene[ee_frame_name]
        ee_pos_w = _to_torch(ee_frame.data.target_pos_w)[:, 0, :3]

        for task_idx, binding in enumerate(tasks):
            goals = list(getattr(binding, "goals", ()))
            if not goals:
                continue
            env_ids = rt.env_ids_for(task_idx)
            if env_ids.numel() == 0:
                continue
            group_ee_pos = ee_pos_w[env_ids]
            needs_refresh = self._needs_refresh[env_ids]

            accum = torch.zeros(env_ids.shape[0], dtype=torch.float32, device=env.device)
            count = 0
            for goal in goals:
                shaped = self._shape_single_goal(
                    env, binding, task_idx, goal, env_ids, group_ee_pos, needs_refresh,
                    std_reach, std_place, std_joint, lift_weight, lift_threshold,
                )
                if shaped is None:
                    continue
                accum += shaped
                count += 1
            if count > 0:
                rewards[env_ids] = accum / count
            self._needs_refresh[env_ids] = False

        return rewards * scale

    def _shape_single_goal(
        self,
        env: ManagerBasedRLEnv,
        binding: TaskBinding,
        task_idx: int,
        goal: dict,
        env_ids: torch.Tensor,
        group_ee_pos: torch.Tensor,
        needs_refresh: torch.Tensor,
        std_reach: float,
        std_place: float,
        std_joint: float,
        lift_weight: float,
        lift_threshold: float,
    ) -> torch.Tensor | None:
        """Shape one goal predicate over one task's envs; ``None`` when unresolvable."""
        if "relationship" in goal:
            obj_name = goal["ref_obj"]
            obj_pos = _resolve_entity_pos(env, binding, obj_name, task_idx)
            # Same target point the success predicate uses -- link pose plus the
            # asset-local ``target_offset`` and the world-frame ``orientation``
            # offset.  Shaping the policy toward the bare link origin pulled it
            # 0.15 m off the burner plate and 0.15 m below the drawer floor.
            target_pos = _goal_target_pos(env, binding, goal, task_idx)
            if obj_pos is None or target_pos is None:
                return None

            pos_diff = obj_pos - target_pos
            height_diff = float(goal.get("height_diff", 0.0))
            xy_dist = torch.linalg.vector_norm(pos_diff[:, :2].contiguous(), dim=1)
            # signed, so a negative height_diff (object below the target point,
            # e.g. the plate pushed to the table in front of the stove top) shapes
            # toward the right side rather than its mirror image
            d_obj_target = xy_dist + torch.abs(pos_diff[:, 2] - height_diff)

            d_ee_obj = torch.linalg.vector_norm(group_ee_pos - obj_pos, dim=1)
            reach = 1.0 - torch.tanh(d_ee_obj / max(std_reach, 1e-6))
            place = 1.0 - torch.tanh(d_obj_target / max(std_place, 1e-6))

            # Snapshot / refresh the object's initial z for the lift bonus.
            current_z = obj_pos[:, 2]
            key = (task_idx, obj_name)
            buf = self._init_obj_z.get(key)
            if buf is None or buf.shape[0] != current_z.shape[0]:
                self._init_obj_z[key] = current_z.detach().clone()
            else:
                self._init_obj_z[key] = torch.where(needs_refresh, current_z.detach(), buf)
            lift = torch.clamp((current_z - self._init_obj_z[key]) / max(lift_threshold, 1e-6), 0.0, 1.0)
            # Gate lift by reach: only credit lift when the gripper is near the object.
            return hamacher_product(reach, place) + lift_weight * (lift * reach)

        if "operation" in goal:
            spec = _resolve_articulation_joint_spec(goal)
            if spec is None:
                return None
            low, high, joint_pattern = spec
            joint_pos = _articulation_joint_pos(env, binding, goal["target"], joint_pattern, task_idx)
            if joint_pos is None:
                return None

            op = _locate(env, binding, task_idx, goal["target"])
            if op.asset is None:
                return None
            handle_idx = _find_handle_body_index(op.asset, goal, joint_pattern)
            if handle_idx is not None:
                handle_pos = _to_torch(op.asset.data.body_pos_w)[op.view_ids, handle_idx, :3]
            else:
                handle_pos = _to_torch(op.asset.data.root_pos_w)[op.view_ids, :3]

            d_ee_handle = torch.linalg.vector_norm(group_ee_pos - handle_pos, dim=1)
            approach = 1.0 - torch.tanh(d_ee_handle / max(std_reach, 1e-6))

            target_center = 0.5 * (low + high)
            target_half = max(0.5 * (high - low), 1e-3)
            dist_to_range = torch.clamp(torch.abs(joint_pos - target_center) - target_half, min=0.0)
            # Rotational joints (microwave door, stove knob) have radian-scale
            # ranges (> 0.3); translational drawers are metre-scale (< 0.3).
            std_effective = std_joint if float(high - low) > 0.3 else std_place
            progress = 1.0 - torch.tanh(dist_to_range / max(std_effective, 1e-6))
            return hamacher_product(approach, progress)

        return None
