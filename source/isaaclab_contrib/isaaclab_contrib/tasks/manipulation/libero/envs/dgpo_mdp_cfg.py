# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative MDP manager configs for the LIBERO DGPO env (playground-style).

One ``@configclass`` per manager block — observations, each reward mode, and
terminations. All terms read the task bindings from
``env.cfg.dgpo_task_bindings`` at runtime, so no runtime data is threaded
through cfg params and the factory (:mod:`.dgpo_env_cfg`) only *selects* classes:

Observations (locked DGPO layout, actor=300 / critic=552):

* :class:`DgpoObservationsCfg` — ``policy`` (277) + ``proprio`` (23) +
  ``privileged_proprio`` (252, critic-only demo diffs + demo phase).
  ``gripper_force`` is in neither group; see :class:`DgpoProprioCfg`.

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
# Fallback only -- every goal in config/ now carries its own `max_speed`.  0.4 m/s
# is a brisk carry rather than "at rest": it let a success be scored while the
# gripper was still ferrying the object, so a policy could hold it near the goal
# instead of placing it.  A settled object is well under 0.05.
LIBERO_SUCCESS_SPEED_THRESHOLD = 0.05

# The demonstration-guided reward every method in the comparison runs on.  `goal`
# (sparse) and `metaworld_dense` (demo-free) are different benchmarks, not
# different algorithms, so they are opt-in rather than the default.
DEFAULT_LIBERO_REWARD_MODE = "world_state_tracking_reward"

# Demo-tracking kernel bandwidth: exp(-||e|| / std), std = 0.1 for every term --
# metres AND radians alike, so the rotation kernel is 0.1 rad = 5.7 deg wide.
#
# Override individual widths with `env.rewards.<term>.params.std=` when needed.
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
    # Zeroes this term for envs whose gripper bit is currently driven by the
    # demonstration rather than the policy; see demo_rewards._gripper_reward_gate.
    # Applies to every method equally -- while the curriculum owns the bit the term
    # is action-independent for every method.
    "gate_on_gripper_curriculum": True,
}
_ARM_JOINT_PARAMS = {
    "std": _TRACK_STD,
    "asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"]),
    "command_name": "joint_state",
}
# Velocity error is in rad/s, not m or rad, so the shared _TRACK_STD is a coincidence
# of the playground values rather than the same quantity.  At the do-nothing fixed
# point this term reads exp(-||v_demo||/std), which at std=0.1 and teleop demo speeds
# is 1e-2..1e-4: a live gradient, but a small one.  Widen it here (0.3-0.5) if the
# freeze survives the term, at the cost of playground parity.
_ARM_JOINT_VEL_PARAMS = {
    "std": _TRACK_STD,
    "asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"]),
    "command_name": "joint_velocity",
}
_OBJECT_PARAMS = {
    "std": _TRACK_STD,
    "command_name": "source_action",
    # Score an object only once the demonstration has started moving it.  Ungated,
    # an object nobody has touched sits exactly where the demo has it and pays a
    # perfect score for standing still -- ~0.2/step across the three object terms,
    # collected by a policy that never acts.  See demo_rewards._demo_slot_in_play.
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
    this group *and* from :class:`DgpoPrivilegedProprioCfg`. All methods share
    these groups. Demonstrations contain no contact stream, so contact-force
    history cannot be matched between demonstration and online observations.
    See :mod:`~...dgpo_layout`.

    To reinstate it: re-add the ``contact_gripper`` ContactSensorCfg to
    :mod:`~...robots.franka_osc`, an observation term reading it, the term in *both*
    groups, and regenerate the demo datasets with real contact forces.
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
    # observation.  See `dgpo_env_cfg` on `is_finite_horizon` for why that
    # matters more here than the usual "nice to have".
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
    """Safety / stability penalties shared by every reward mode (playground values)."""

    action_rate = RewTerm(func=env_mdp.action_rate_l2, weight=-0.0005)
    action_norm = RewTerm(func=env_mdp.action_l2, weight=-0.0005)
    joint_vel = RewTerm(
        func=env_mdp.joint_vel_l2,
        weight=-0.001,
        # arm joints only (playground syncs this term to panda_joint.*)
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
        # weight 1.0, not 10.0: at 10 the sparse payout dominates every shaping term,
        # which makes reward hacking the cheapest way to earn return -- the predicate
        # for "A on B" is satisfied by shoving B under A just as well as by picking A
        # up, and shoving is far easier to find.  Kinematic receptacles close that
        # particular shortcut, but a payout an order of magnitude above the shaping
        # still biases search toward whatever satisfies the predicate rather than
        # toward the demonstrated motion.  Kept identical across all three reward
        # modes so the sparse term has the same magnitude for every method.
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

    Two properties of this set are load-bearing against the do-nothing fixed point,
    because zero action under ``pose_rel`` OSC *is* "hold still" and the action
    penalties' optimum sits exactly there:

    * ``joint_velocity_tracking`` is the only term whose error does not grow with the
      drift, so it still carries gradient once the pose kernels have saturated.
    * the object terms are gated on the demo having moved each object, so an untouched
      object no longer pays a perfect score for inaction.

    The gate does give up one thing: an object the demo never touches is no longer
    scored at all, so "don't knock the receptacle over" is left to the success
    predicate rather than being paid per step.
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
    # Joint-space tracking (playground TrackingRewardsCfg parity; both terms were
    # missing here).  The pose terms above all score *where* the arm is, and once it
    # has drifted more than a few std off the reference every one of them is
    # numerically flat -- leaving the action penalties, whose optimum is exactly zero
    # action, as the only remaining gradient.  That is the do-nothing fixed point.
    # `joint_position_tracking` densifies the same basin in joint space (the EE pose
    # is 6-D against 7 joints, so it constrains strictly less), and
    # `joint_velocity_tracking` is the one term whose error stays bounded by the
    # demonstration's own speed instead of growing with the drift, so it still points
    # somewhere useful from inside the plateau.
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

    # Sparse success signal.  This replaces the three success-gated *accumulated*
    # tracking payouts (end_effector_position_goal / _orientation_goal /
    # gripper_state_goal) this config used to carry.  Three reasons:
    #
    #  1. Those payouts were a delayed 0.2x duplicate of the per-step tracking
    #     terms -- the payout was sum_u kappa(e_u), and every kappa(e_u) had
    #     already been paid at step u by the per-step terms, just with weight 0.5
    #     instead of 0.1.  Same information, far worse credit assignment.
    #  2. Because the payout was a SUM, its size scaled with episode length, so the
    #     same success was worth 6.7x more from a full episode than from an RFCL
    #     Phase-1 episode injected at cursor 0.85 (0.390 vs 0.058 per term).  The
    #     reverse curriculum was penalised precisely when it needed the signal most.
    #  3. The running accumulator lived inside the reward term and appeared in no
    #     observation, so the payout was unpredictable from the observed state for
    #     both actor and critic.
    #
    # A plain success predicate has none of those properties, and it makes this
    # config structurally comparable to LiberoGoalRewardsCfg and
    # LiberoMetaworldDenseRewardsCfg, which both already use this exact term and
    # weight.  The aggregated term functions stay in ``demo_rewards`` for anyone
    # reinstating that reward shape.
    task_success = RewTerm(
        func=libero_task_success_reward,
        weight=1.0,
        params={"speed_threshold": LIBERO_SUCCESS_SPEED_THRESHOLD},
    )


@configclass
class LiberoMetaworldDenseRewardsCfg(LiberoBaseRewardsCfg):
    """Demo-free dense shaping from the BDDL goal predicates + sparse success bonus.

    Playground ``MetaworldStyleDenseRewardCfg`` port: relationship goals compose
    ``reach`` x ``place`` (Hamacher product) with a reach-gated lift bonus;
    articulation operation goals (open/close/turnon) compose ``approach`` x
    joint ``progress``. Params match the playground ``goal_predicate_dense`` term.
    """

    task_success = RewTerm(
        func=libero_task_success_reward,
        # weight 1.0, not 10.0: at 10 the sparse payout dominates every shaping term,
        # which makes reward hacking the cheapest way to earn return -- the predicate
        # for "A on B" is satisfied by shoving B under A just as well as by picking A
        # up, and shoving is far easier to find.  Kinematic receptacles close that
        # particular shortcut, but a payout an order of magnitude above the shaping
        # still biases search toward whatever satisfies the predicate rather than
        # toward the demonstrated motion.  Kept identical across all three reward
        # modes so the sparse term has the same magnitude for every method.
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

# ``world_state_tracking_reward`` was called ``world_prediction`` until 2026-08-11,
# and legacy launch configs still use the old name. Accepted with a warning rather than
# rejected, because a copy-pasted launch would otherwise fail after the env has
# already booted Isaac Sim.
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
# Success always ends an eval episode (that is what ``eval_libero.py`` counts).
# In training it is opt-in; see LiberoDgpoTerminationsCfg for what it changes.
_SUCCESS_TERMINATES = _EVAL_MODE or settings.truthy("LIBERO_SUCCESS_TERMINATION")


@configclass
class LiberoDgpoTerminationsCfg:
    """Playground termination scheme (success does NOT terminate in training).

    * Training: ``command_depleted`` truncation (demo command exhausted — the
      episode horizon) + joint pos/vel early terminations. Success keeps the
      episode running (the policy must hold the goal state); SR logging reads
      the mask stashed by :func:`libero_task_success` instead.
    * Eval (``LIBERO_EVALUATION=1``): ``command_depleted`` + success
      termination, so ``eval_libero.py`` counts one episode per success and
      failed episodes still end when the demo command is exhausted.

    The env factory swaps ``command_depleted`` for a plain fixed-length
    ``time_out`` when demos are not wired (no ``source_action`` progress).

    ``LIBERO_SUCCESS_TERMINATION=1`` moves the eval behaviour into training. That
    is not only a stopping rule: ``task_success`` pays 1.0 *per step* the predicate
    holds, so without termination its episode payout is ``(T - t*) * w * dt`` and
    every step saved is money -- it is the term that pays for finishing early.
    Terminating collapses the payout to a single step, which removes that bonus but
    inverts the pressure, because this term is a hard terminal (``time_out`` unset),
    so the value target at success is 0 rather than a bootstrap:

    * gained by succeeding at ``t*``: ``w * dt`` (0.05 at ``w = 1``)
    * given up: the remaining ``(T - t*)`` steps of tracking reward

    Measured on ``dgpo_all_beta05_fix_goal``, the tracking terms pay ~0.43/step
    against a one-shot 0.05 -- order 40:1 against ever tripping the predicate,
    whose best response is to hover just outside it until ``command_depleted``.
    Raise ``env.rewards.task_success.weight`` alongside the switch (break-even is
    ``w ~ (T - t*) * sum_k w_k * kernel_k``, order 40-100 at the current
    ``_TRACK_STD``), or widen ``_TRACK_STD`` so the tracking terms are worth
    following at all.
    """

    # the episode horizon: truncate (time-out flavored) when the demo command
    # is exhausted (train AND eval, playground parity)
    command_depleted = DoneTerm(
        func=command_finished,
        params={"command_name": "source_action", "threshold": 1.0},
        time_out=True,
    )

    # eval always; training under LIBERO_SUCCESS_TERMINATION=1
    libero_success = (
        DoneTerm(
            func=libero_task_success,
            params={"speed_threshold": LIBERO_SUCCESS_SPEED_THRESHOLD},
        )
        if _SUCCESS_TERMINATES
        else None
    )

    # training-only early terminations (playground values). Arm joints only:
    # the fingers' open position (0.04) sits exactly on the joint limit, so any
    # PD overshoot would otherwise kill every episode within a few steps.
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
