# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative MDP manager configs for the LIBERO DGPO env.

One ``@configclass`` per manager block — observations, each reward mode, and
terminations. All terms read the task bindings from
``env.cfg.dgpo_task_bindings`` at runtime, so no runtime data is threaded
through cfg params and the factory (:mod:`.dgpo_env_cfg`) only *selects* classes:

Observations (locked DGPO layout, actor=300 / critic=552):

* :class:`DgpoObservationsCfg` — ``policy`` (277) + ``proprio`` (23) +
  ``privileged_proprio`` (252, critic-only demo diffs + demo phase).

Rewards (``LIBERO_REWARD_MODE``, default ``world_state_tracking_reward``):

* :class:`LiberoWorldStateTrackingRewardsCfg` (default) — dense per-step tracking
  of the demonstration's world state (EE pose / gripper / object poses vs the demo
  command views) plus the sparse success term.
* :class:`LiberoGoalRewardsCfg` — sparse success bonus only (+ safety penalties);
  ``LIBERO_REWARD_MODE=goal``.
* :class:`LiberoMetaworldDenseRewardsCfg` — demo-free goal-predicate shaping
  (Hamacher reach x place / approach x joint progress) + success bonus;
  ``LIBERO_REWARD_MODE=metaworld_dense``.

Terminations:

* :class:`LiberoDgpoTerminationsCfg` — training: timeout + demo-command
  depletion + joint pos/vel early terminations (success does NOT terminate);
  eval (``LIBERO_EVALUATION=1``): timeout + success termination.
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as env_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ..mdp import demo_rewards
from ..mdp import observations as dgpo_mdp
from ..mdp.combined import (
    command_finished,
    joint_vel_exceeds_scaled_limit,
    libero_task_success,
    libero_task_success_reward,
)
from ..mdp.metaworld_rewards import metaworld_dense_reward
from ... import settings

# Geometric success proxy: object near rest (matches the harvest All env).
LIBERO_SUCCESS_SPEED_THRESHOLD = 0.05

# The demonstration-guided reward every method in the comparison runs on.
DEFAULT_LIBERO_REWARD_MODE = "world_state_tracking_reward"

_TRACK_STD = 0.1

_EE_POSE_PARAMS = {
    "std": _TRACK_STD,
    "frame_name": "franka_ee_frame",
    "command_name": "ee_pose",
    "target_index": 0,
}
_GRIPPER_PARAMS = {
    "std": _TRACK_STD,
    "asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_finger.*"]),
    "command_name": "gripper_state",
    "gate_on_gripper_curriculum": True,
}
_ARM_JOINT_PARAMS = {
    "std": _TRACK_STD,
    "asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"]),
    "command_name": "joint_state",
}
_ARM_JOINT_VEL_PARAMS = {
    "std": _TRACK_STD,
    "asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"]),
    "command_name": "joint_velocity",
}
_OBJECT_PARAMS = {
    "std": _TRACK_STD,
    "command_name": "source_action",
    "gate_until_demo_moves": True,
}


# ---------------------------------------------------------------------------
# Observations (locked DGPO layout)
# ---------------------------------------------------------------------------


@configclass
class DgpoPolicyCfg(ObsGroup):
    """Policy group: last_action + multi-hot + pose buffer → 277."""

    last_action = ObsTerm(func=env_mdp.last_action)
    task_multi_hot = ObsTerm(
        func=dgpo_mdp.task_assignment_multi_hot_encoding,
        params={
            "full_sequence": True,
            "overlap_tags": {
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
            },
            "shared_subtask_order": (
                "alphabet_soup_to_basket",
                "tomato_sauce_to_basket",
                "cream_cheese_to_basket",
                "butter_to_basket",
                "turn_on_flat_stove",
                "moka_pot_on_stove",
            ),
        },
    )
    buffer_pose = ObsTerm(
        func=dgpo_mdp.ObjectTargetPoseBuffer,
        params={
            "include_objects": True,
            "include_targets": True,
            "use_base_frame": True,
            "focused_only": False,
        },
    )

    def __post_init__(self):
        """Disable corruption and concatenate terms for the locked DGPO layout."""
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class DgpoProprioCfg(ObsGroup):
    """Proprio group → 23 (DGPO actor path).

    The 24-dim ``gripper_force`` contact-force history is deliberately absent from
    this group *and* from :class:`DgpoPrivilegedProprioCfg` -- demos contain no
    contact stream to match it against. See :mod:`~...dgpo_layout`.
    """

    eef_pose = ObsTerm(func=dgpo_mdp.ee_frame_pose_in_base_frame)
    gripper_pos = ObsTerm(func=dgpo_mdp.gripper_pos)
    joint_pos = ObsTerm(
        func=env_mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"])},
    )
    joint_vel = ObsTerm(
        func=env_mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"])},
    )

    def __post_init__(self):
        """Disable corruption and concatenate terms for the locked DGPO layout."""
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class DgpoPrivilegedProprioCfg(ObsGroup):
    """Privileged critic group without gripper force → 252.

    EE/joint/object diffs are real when :class:`~.commands.SourceLiberoCommand`
    is installed; otherwise zeros with locked widths.
    """

    eef_pose_diff = ObsTerm(
        func=dgpo_mdp.ee_frame_pose_diff_in_base_frame,
        params={"command_name": "ee_pose"},
    )
    joint_pos_diff = ObsTerm(
        func=dgpo_mdp.joint_pos_diff,
        params={
            "command_name": "joint_state",
            "asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"]),
        },
    )
    object_target_pose_diff = ObsTerm(
        func=dgpo_mdp.ObjectTargetPoseDiff,
        params={
            "command_name": "source_action",
            "include_objects": True,
            "include_targets": True,
            "use_base_frame": True,
            "focused_only": False,
        },
    )
    # The one reward input the critic was otherwise missing: `joint_pos_diff`
    # above is bound to panda_joint.*, so the finger joints that
    # `gripper_state_tracking` scores were unobservable to it.
    gripper_state_diff = ObsTerm(
        func=dgpo_mdp.joint_pos_diff,
        params={
            "command_name": "gripper_state",
            "asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_finger.*"]),
        },
    )
    # Time-to-go.  The episode horizon IS the demo length, so without this the
    # critic cannot tell two states apart by how much horizon is left, and the
    # finite-horizon value function it is asked to fit is not a function of the
    # observation.
    demo_phase = ObsTerm(
        func=dgpo_mdp.demo_phase,
        params={"command_name": "source_action", "include_start": True},
    )

    def __post_init__(self):
        """Disable corruption and concatenate terms for the locked DGPO layout."""
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class DgpoObservationsCfg:
    """Observation groups matching DGPO ``agent.yaml`` ``obs_groups``."""

    policy: DgpoPolicyCfg = DgpoPolicyCfg()
    proprio: DgpoProprioCfg = DgpoProprioCfg()
    privileged_proprio: DgpoPrivilegedProprioCfg = DgpoPrivilegedProprioCfg()



# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class LiberoBaseRewardsCfg:
    """Safety / stability penalties shared by every reward mode."""

    action_rate = RewTerm(func=env_mdp.action_rate_l2, weight=-0.0005)
    action_norm = RewTerm(func=env_mdp.action_l2, weight=-0.0005)
    joint_vel = RewTerm(
        func=env_mdp.joint_vel_l2,
        weight=-0.001,
        # arm joints only (synced to panda_joint.* to match the original reward term)
        params={"asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"])},
    )
    joint_pos_limits = RewTerm(
        func=env_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("franka_robot")},
    )
    joint_vel_limits = RewTerm(
        func=env_mdp.joint_vel_limits,
        weight=-0.5,
        params={"soft_ratio": 0.95, "asset_cfg": SceneEntityCfg("franka_robot")},
    )


@configclass
class LiberoGoalRewardsCfg(LiberoBaseRewardsCfg):
    """Sparse reward for baseline comparison: binary task success only."""

    task_success = RewTerm(
        func=libero_task_success_reward,
        weight=1.0,
        params={"speed_threshold": LIBERO_SUCCESS_SPEED_THRESHOLD},
    )


@configclass
class LiberoWorldStateTrackingRewardsCfg(LiberoBaseRewardsCfg):
    """Dense tracking of the demonstration's world state, plus the sparse success term.

    Per-step exponential-kernel rewards for matching the replayed demo's end-effector
    pose, arm joint position/velocity, gripper state and object/articulation poses.
    The demonstration acts as a per-step reference for the whole world state, not just
    the robot -- hence the name.  Needs assembled demos; the env factory falls back to
    the sparse ``goal`` reward with a warning when they are absent.
    """

    # per-step tracking of the demo command views
    end_effector_position_tracking = RewTerm(
        func=demo_rewards.frame_transformer_position_command_error_exp,
        weight=0.5,
        params=dict(_EE_POSE_PARAMS),
    )
    end_effector_orientation_tracking = RewTerm(
        func=demo_rewards.frame_transformer_orientation_command_error_exp,
        weight=1.0,
        params=dict(_EE_POSE_PARAMS),
    )
    gripper_state_tracking = RewTerm(
        func=demo_rewards.joint_position_command_error_exp,
        weight=0.4,
        params=dict(_GRIPPER_PARAMS),
    )
    joint_position_tracking = RewTerm(
        func=demo_rewards.joint_position_command_error_exp,
        weight=0.5,
        params=dict(_ARM_JOINT_PARAMS),
    )
    joint_velocity_tracking = RewTerm(
        func=demo_rewards.joint_velocity_command_error_exp,
        weight=0.1,
        params=dict(_ARM_JOINT_VEL_PARAMS),
    )
    objects_position_tracking = RewTerm(
        func=demo_rewards.object_position_command_error_exp,
        weight=0.1,
        params=dict(_OBJECT_PARAMS),
    )
    objects_orientation_tracking = RewTerm(
        func=demo_rewards.object_orientation_command_error_exp,
        weight=0.05,
        params=dict(_OBJECT_PARAMS),
    )
    articulation_joint_position_tracking = RewTerm(
        func=demo_rewards.articulation_joint_position_command_error_exp,
        weight=0.05,
        params=dict(_OBJECT_PARAMS),
    )

    # Sparse success signal.
    task_success = RewTerm(
        func=libero_task_success_reward,
        weight=1.0,
        params={"speed_threshold": LIBERO_SUCCESS_SPEED_THRESHOLD},
    )


@configclass
class LiberoMetaworldDenseRewardsCfg(LiberoBaseRewardsCfg):
    """Demo-free dense shaping from the BDDL goal predicates + sparse success bonus.

    Ported from ``MetaworldStyleDenseRewardCfg``: relationship goals compose
    ``reach`` x ``place`` (Hamacher product) with a reach-gated lift bonus;
    articulation goals (open/close/turn-on) compose ``approach`` x joint ``progress``.
    """

    task_success = RewTerm(
        func=libero_task_success_reward,
        weight=1.0,
        params={"speed_threshold": LIBERO_SUCCESS_SPEED_THRESHOLD},
    )
    goal_predicate_dense = RewTerm(
        func=metaworld_dense_reward,
        weight=1.0,
        params={
            "std_reach": 0.3,
            "std_place": 0.2,
            "std_joint": 0.5,
            "scale": 10.0,
            "lift_weight": 0.5,
            "lift_threshold": 0.05,
            "ee_frame_name": "franka_ee_frame",
        },
    )


LIBERO_REWARDS_CFGS: dict[str, type] = {
    "goal": LiberoGoalRewardsCfg,
    "world_state_tracking_reward": LiberoWorldStateTrackingRewardsCfg,
    "metaworld_dense": LiberoMetaworldDenseRewardsCfg,
}

LIBERO_REWARD_MODE_ALIASES: dict[str, str] = {"world_prediction": "world_state_tracking_reward"}


def resolve_reward_mode(mode: str) -> str:
    """Map a reward-mode string through the deprecated aliases and validate it."""
    mode = (mode or "").strip().lower() or DEFAULT_LIBERO_REWARD_MODE
    if mode in LIBERO_REWARD_MODE_ALIASES:
        import warnings

        new = LIBERO_REWARD_MODE_ALIASES[mode]
        warnings.warn(
            f"LIBERO_REWARD_MODE='{mode}' is the old name for '{new}'; use the new name.",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = new
    if mode not in LIBERO_REWARDS_CFGS:
        raise ValueError(f"LIBERO_REWARD_MODE must be one of {sorted(LIBERO_REWARDS_CFGS)}; got '{mode}'.")
    return mode


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


_EVAL_MODE = settings.truthy("LIBERO_EVALUATION")
_SUCCESS_TERMINATES = _EVAL_MODE or settings.truthy("LIBERO_SUCCESS_TERMINATION")


@configclass
class LiberoDgpoTerminationsCfg:
    """Termination scheme (success does NOT terminate in training).

    * Training: ``command_depleted`` truncation (demo command exhausted — the
      episode horizon) + joint pos/vel early terminations. Success keeps the
      episode running (the policy must hold the goal state); SR logging reads
      the mask stashed by :func:`libero_task_success` instead.
    * Eval (``LIBERO_EVALUATION=1``): ``command_depleted`` + success
      termination, so ``eval_libero.py`` counts one episode per success and
      failed episodes still end when the demo command is exhausted.

    The env factory swaps ``command_depleted`` for a plain fixed-length
    ``time_out`` when demos are not wired (no ``source_action`` progress).

    """

    command_depleted = DoneTerm(
        func=command_finished,
        params={"command_name": "source_action", "threshold": 1.0},
        time_out=True,
    )

    libero_success = (
        DoneTerm(
            func=libero_task_success,
            params={"speed_threshold": LIBERO_SUCCESS_SPEED_THRESHOLD},
        )
        if _SUCCESS_TERMINATES
        else None
    )

    joint_pos_out_of_limit = (
        DoneTerm(
            func=env_mdp.joint_pos_out_of_limit,
            params={"asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"])},
        )
        if not _EVAL_MODE
        else None
    )
    joint_vel_exceeds_limit = (
        DoneTerm(
            func=joint_vel_exceeds_scaled_limit,
            params={"scale": 1.5, "asset_cfg": SceneEntityCfg("franka_robot")},
        )
        if not _EVAL_MODE
        else None
    )
