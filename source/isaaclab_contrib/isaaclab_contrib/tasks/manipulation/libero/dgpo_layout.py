# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locked observation / action dimension contract for DGPO+ABC checkpoints.

Observation and action contract:

* Agent ``obs_groups``: actor=[policy, proprio], critic=[policy, proprio, privileged_proprio]
  (rsl-rl >= 4.0 key is ``actor``; legacy agent.yaml used ``policy`` for the same set)
* Action: OSC arm (6) + binary gripper (1) = 7
* Suite order in the checkpoint: ``libero_10`` / ``libero_object`` / ``libero_spatial`` /
  ``libero_goal`` (40 tasks). Harvest uses :data:`DGPO_ABC_HARVEST_SUITES`
  (``libero_long`` ≡ ``libero_10``) so sequential ``env i → task i % n`` matches.
* Control rate: ``sim.dt=1/60``, ``decimation=3`` → 0.05 s (matches demo preprocessing).

Widths are actor 300 / critic 552.  ``privileged_proprio`` carries
``gripper_state_diff`` (2) and ``demo_phase`` (2) on top of the demo diffs; see
:class:`~.envs.dgpo_mdp_cfg.DgpoPrivilegedProprioCfg` for why the critic needs
both.

The 24-dim ``gripper_force`` contact-force history is **no longer an observation
in any group**, which moved actor 324 → 300 and critic 572 → 548 (before the two
terms above brought the critic to 552). All methods use the same observation
groups, and demonstrations do not contain the contact-force history needed to
match online observations.

The ``contact_gripper`` sensor has been removed from the scene along with it --
nothing read it, and PhysX was computing fingertip contacts every step for a
result that was discarded.  ``DGPO_ABC_LEGACY_*_OBS_DIM`` record the pre-fix
widths so the paper DGPO+ABC checkpoint can still be identified; checkpoints
trained at 324/572 will not load into the current contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Canonical object / target columns for ObjectTargetPoseBuffer (26 × 9 = 234)
# ---------------------------------------------------------------------------

LIBERO_ORDERED_OBJECT_NAMES: tuple[str, ...] = (
    "alphabet_soup_1",
    "tomato_sauce_1",
    "cream_cheese_1",
    "butter_1",
    "moka_pot_1",
    "akita_black_bowl_1",
    "white_yellow_mug_1",
    "porcelain_mug_1",
    "black_book_1",
    "chocolate_pudding_1",
    "moka_pot_2",
    "salad_dressing_1",
    "bbq_sauce_1",
    "ketchup_1",
    "milk_1",
    "orange_juice_1",
    "wine_bottle_1",
    "plate_1",
    "basket_1",
    "flat_stove_1",
    "white_cabinet_1",
    "plate_2",
    "desk_caddy_1",
    "microwave_1",
    "wooden_cabinet_1",
    "wine_rack_1",
)

# Shared subtask tags that compress 40 tasks → 36 multi-hot dims.
LIBERO_SHARED_SUBTASK_ORDER: tuple[str, ...] = (
    "alphabet_soup_to_basket",
    "tomato_sauce_to_basket",
    "cream_cheese_to_basket",
    "butter_to_basket",
    "turn_on_flat_stove",
    "moka_pot_on_stove",
)

LIBERO_SHARED_SUBTASK_TAGS: dict[str, tuple[str, ...]] = {
    "libero_10::0": ("alphabet_soup_to_basket", "tomato_sauce_to_basket"),
    "libero_10::1": ("cream_cheese_to_basket", "butter_to_basket"),
    "libero_10::7": ("alphabet_soup_to_basket", "cream_cheese_to_basket"),
    "libero_10::2": ("turn_on_flat_stove", "moka_pot_on_stove"),
    "libero_10::8": ("moka_pot_on_stove",),
    "libero_object::0": ("alphabet_soup_to_basket",),
    "libero_object::5": ("tomato_sauce_to_basket",),
    "libero_object::1": ("cream_cheese_to_basket",),
    "libero_object::6": ("butter_to_basket",),
    "libero_goal::7": ("turn_on_flat_stove",),
}

# Checkpoint task sequence (encoding space). ``libero_10`` ≡ harvest ``libero_long``.
DGPO_ABC_TASK_SEQUENCE: tuple[str, ...] = tuple(
    [f"libero_10::{i}" for i in range(10)]
    + [f"libero_object::{i}" for i in range(10)]
    + [f"libero_spatial::{i}" for i in range(10)]
    + [f"libero_goal::{i}" for i in range(10)]
)

# Harvest ``(suite_name, prefix)`` order matching DGPO+ABC ``env_task_assignments`` /
# ``full_task_sequence`` (libero_10 → object → spatial → goal).
DGPO_ABC_HARVEST_SUITES: tuple[tuple[str, str], ...] = (
    ("libero_long", "long"),
    ("libero_object", "object"),
    ("libero_spatial", "spatial"),
    ("libero_goal", "goal"),
)

# Locked control / episode knobs from DGPO+ABC ``env.yaml`` (sim.dt=1/60).
DGPO_ABC_DECIMATION: int = 3
DGPO_ABC_SIM_DT: float = 1.0 / 60.0
DGPO_ABC_EPISODE_LENGTH_S: float = 26.0
DGPO_ABC_CONTROL_DT: float = DGPO_ABC_SIM_DT * DGPO_ABC_DECIMATION  # 0.05 s

# OSC scales from playground OscPose multitask (pose_rel).
DGPO_ABC_OSC_POSITION_SCALE: float = 1.0 / 20.0
DGPO_ABC_OSC_ORIENTATION_SCALE: float = 1.0 / 2.0
DGPO_ABC_OSC_MOTION_STIFFNESS: tuple[float, ...] = (150.0, 150.0, 150.0, 150.0, 150.0, 150.0)
DGPO_ABC_OSC_NULLSPACE_STIFFNESS: float = 10.0
DGPO_ABC_GRIPPER_OPEN: float = 0.04
DGPO_ABC_GRIPPER_CLOSE: float = 0.0

# Harvest suite name / task-name prefix → DGPO+ABC assignment suite key.
HARVEST_SUITE_TO_ASSIGNMENT: dict[str, str] = {
    "libero_long": "libero_10",
    "libero_object": "libero_object",
    "libero_spatial": "libero_spatial",
    "libero_goal": "libero_goal",
}


def harvest_task_to_assignment_key(task_name: str, suite: str) -> str:
    """Map a harvest task (e.g. ``long_task2`` / suite ``libero_long``) → ``libero_10::2``."""
    suite_key = HARVEST_SUITE_TO_ASSIGNMENT.get(suite, suite)
    # Task names are ``{prefix}_task{id}`` (prefix from harvest suites).
    marker = "_task"
    if marker not in task_name:
        raise ValueError(f"Unexpected harvest task name '{task_name}' (expected '*_taskN').")
    task_id = int(task_name.rsplit(marker, 1)[-1])
    return f"{suite_key}::{task_id}"


POSE_BUFFER_JOINT_PAD: int = 3  # cabinet articulations pad to 3 joints
POSE_BUFFER_POSE_DIM: int = 6  # xyz + rpy
POSE_BUFFER_SLOT_DIM: int = POSE_BUFFER_POSE_DIM + POSE_BUFFER_JOINT_PAD  # 9
DGPO_ABC_ACTION_DIM: int = 7
DGPO_ABC_ACTOR_OBS_DIM: int = 300
DGPO_ABC_CRITIC_OBS_DIM: int = 552

# Pre-fix widths, kept so a checkpoint can be identified by shape; they are not
# supported contracts.
#   324 / 572: ``gripper_force`` (24) still sat in ``proprio``.
#   548:       ``privileged_proprio`` was missing ``gripper_state_diff`` (2) and
#              ``demo_phase`` (2); see :class:`~.envs.dgpo_mdp_cfg.DgpoPrivilegedProprioCfg`.
DGPO_ABC_LEGACY_ACTOR_OBS_DIM: int = 324
DGPO_ABC_LEGACY_CRITIC_OBS_DIM: int = 572
DGPO_ABC_LEGACY_CRITIC_OBS_DIM_NO_PHASE: int = 548


@dataclass(frozen=True)
class ObsTermSpec:
    """One observation term with a fixed feature width."""

    name: str
    dim: int
    notes: str = ""


@dataclass(frozen=True)
class ObsGroupSpec:
    """Concatenated observation group (``concatenate_terms=True``)."""

    name: str
    terms: tuple[ObsTermSpec, ...]

    @property
    def dim(self) -> int:
        """Return the concatenated feature width of all terms in this group."""
        return sum(term.dim for term in self.terms)


@dataclass(frozen=True)
class DimContract:
    """End-to-end actor / critic / action dimension contract."""

    name: str
    action_dim: int
    policy: ObsGroupSpec
    proprio: ObsGroupSpec
    privileged_proprio: ObsGroupSpec
    agent_policy_groups: tuple[str, ...]
    agent_critic_groups: tuple[str, ...]

    @property
    def actor_obs_dim(self) -> int:
        """Return the concatenated actor input width over ``agent_policy_groups``."""
        groups = {"policy": self.policy, "proprio": self.proprio, "privileged_proprio": self.privileged_proprio}
        return sum(groups[name].dim for name in self.agent_policy_groups)

    @property
    def critic_obs_dim(self) -> int:
        """Return the concatenated critic input width over ``agent_critic_groups``."""
        groups = {"policy": self.policy, "proprio": self.proprio, "privileged_proprio": self.privileged_proprio}
        return sum(groups[name].dim for name in self.agent_critic_groups)


def multi_hot_encoding_width(
    task_sequence: tuple[str, ...] = DGPO_ABC_TASK_SEQUENCE,
    overlap_tags: dict[str, tuple[str, ...]] = LIBERO_SHARED_SUBTASK_TAGS,
    shared_subtask_order: tuple[str, ...] = LIBERO_SHARED_SUBTASK_ORDER,
    fallback_to_assignment: bool = True,
) -> int:
    """Compute ``task_assignment_multi_hot_encoding`` width without a live env.

    Mirrors playground ``_build_overlap_tag_tensor`` column construction:
    shared tags (in declared order) first, then remaining fallback assignment
    keys in sorted order.
    """
    return len(build_overlap_tag_columns(task_sequence, overlap_tags, shared_subtask_order, fallback_to_assignment))


def build_overlap_tag_columns(
    task_sequence: tuple[str, ...] = DGPO_ABC_TASK_SEQUENCE,
    overlap_tags: dict[str, tuple[str, ...]] | Mapping[str, Sequence[str]] = LIBERO_SHARED_SUBTASK_TAGS,
    shared_subtask_order: tuple[str, ...] = LIBERO_SHARED_SUBTASK_ORDER,
    fallback_to_assignment: bool = True,
) -> tuple[str, ...]:
    """Return the ordered multi-hot column tags (width 36 for DGPO+ABC)."""
    all_tags: set[str] = set()
    for assignment in task_sequence:
        tags = tuple(overlap_tags.get(assignment, ()))
        if not tags and fallback_to_assignment:
            tags = (assignment,)
        all_tags.update(tags)
    ordered = [tag for tag in shared_subtask_order if tag in all_tags]
    for tag in sorted(all_tags):
        if tag not in ordered:
            ordered.append(tag)
    return tuple(ordered)


def encode_assignment_multi_hot(
    assignments: Sequence[str],
    *,
    device: str = "cpu",
    task_sequence: tuple[str, ...] = DGPO_ABC_TASK_SEQUENCE,
    overlap_tags: dict[str, tuple[str, ...]] | Mapping[str, Sequence[str]] = LIBERO_SHARED_SUBTASK_TAGS,
    shared_subtask_order: tuple[str, ...] = LIBERO_SHARED_SUBTASK_ORDER,
    fallback_to_assignment: bool = True,
):
    """Encode assignment keys to a multi-hot tensor ``(N, 36)`` without a live env.

    Returns a ``torch.Tensor`` when torch is importable; otherwise raises.
    """
    import torch

    ordered_tags = build_overlap_tag_columns(task_sequence, overlap_tags, shared_subtask_order, fallback_to_assignment)
    tag_map: dict[str, tuple[str, ...]] = {}
    for assignment in task_sequence:
        tags = tuple(overlap_tags.get(assignment, ()))
        if not tags and fallback_to_assignment:
            tags = (assignment,)
        tag_map[assignment] = tags
    tag_to_index = {tag: idx for idx, tag in enumerate(ordered_tags)}
    # Fill on the host, then transfer once. Writing out[row, idx] = 1.0 with the
    # tensor already on the device costs one kernel launch per set bit, i.e. one
    # per (env, tag) -- thousands of launches at 1000 envs to build a constant.
    out = torch.zeros((len(assignments), len(ordered_tags)), dtype=torch.float32)
    for row, assignment in enumerate(assignments):
        for tag in tag_map.get(assignment, ()):
            idx = tag_to_index.get(tag)
            if idx is not None:
                out[row, idx] = 1.0
    return out.to(device=device)


def dgpo_abc_contract() -> DimContract:
    """Return the locked DGPO+ABC dimension contract."""
    buffer_slots = len(LIBERO_ORDERED_OBJECT_NAMES)
    buffer_dim = buffer_slots * POSE_BUFFER_SLOT_DIM
    multi_hot_dim = multi_hot_encoding_width()

    policy = ObsGroupSpec(
        name="policy",
        terms=(
            ObsTermSpec("last_action", 7, "OSC arm (6) + gripper (1)"),
            ObsTermSpec("task_multi_hot", multi_hot_dim, "40 tasks compressed via shared subtask tags"),
            ObsTermSpec(
                "buffer_pose",
                buffer_dim,
                f"ObjectTargetPoseBuffer: {buffer_slots}×{POSE_BUFFER_SLOT_DIM}",
            ),
        ),
    )
    proprio = ObsGroupSpec(
        name="proprio",
        terms=(
            ObsTermSpec("eef_pose", 7, "xyz + quat_wxyz in base frame (DGPO+ABC checkpoint layout)"),
            ObsTermSpec("gripper_pos", 2),
            ObsTermSpec("joint_pos", 7, "arm joints only"),
            ObsTermSpec("joint_vel", 7),
            # gripper_force is deliberately absent from BOTH groups; see the
            # module docstring.  The sensor still exists, it just never reaches
            # a network.
        ),
    )
    privileged = ObsGroupSpec(
        name="privileged_proprio",
        terms=(
            ObsTermSpec("eef_pose_diff", 7, "xyz + quat_wxyz tracking error vs demo ee_pose"),
            ObsTermSpec("joint_pos_diff", 7, "arm joints only"),
            ObsTermSpec("object_target_pose_diff", buffer_dim, "ObjectTargetPoseDiff vs demo"),
            ObsTermSpec("gripper_state_diff", 2, "finger joints the gripper reward scores"),
            ObsTermSpec("demo_phase", 2, "normalized demo cursor + episode start cursor"),
        ),
    )
    return DimContract(
        name="DGPO+ABC",
        action_dim=DGPO_ABC_ACTION_DIM,
        policy=policy,
        proprio=proprio,
        privileged_proprio=privileged,
        agent_policy_groups=("policy", "proprio"),
        agent_critic_groups=("policy", "proprio", "privileged_proprio"),
    )


def verify_contract_arithmetic(contract: DimContract | None = None) -> DimContract:
    """Assert contract dims match the locked checkpoint targets.

    Raises:
        AssertionError: If any group or concatenated dim diverges.
    """
    contract = contract or dgpo_abc_contract()
    assert multi_hot_encoding_width() == 36, multi_hot_encoding_width()
    assert len(LIBERO_ORDERED_OBJECT_NAMES) == 26
    assert contract.policy.dim == 7 + 36 + 234, contract.policy.dim
    assert contract.proprio.dim == 7 + 2 + 7 + 7, contract.proprio.dim
    assert contract.privileged_proprio.dim == 7 + 7 + 234 + 2 + 2, contract.privileged_proprio.dim
    assert contract.actor_obs_dim == DGPO_ABC_ACTOR_OBS_DIM, contract.actor_obs_dim
    assert contract.critic_obs_dim == DGPO_ABC_CRITIC_OBS_DIM, contract.critic_obs_dim
    assert contract.action_dim == DGPO_ABC_ACTION_DIM
    return contract
