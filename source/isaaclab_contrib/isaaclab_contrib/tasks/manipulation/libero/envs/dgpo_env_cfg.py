# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a harvest+CloneCfg LIBERO env with DGPO observation / action layout.

This is the **first-class DGPO training path** on the harvest / cloner API
(not a temporary checkpoint-only bridge).

Goals
-----

1. Keep the efficient harvest scene (``CloneCfg`` / ``InclusionSet`` / shared
   prototypes).
2. Use OSC so ``action_dim == 7`` (pose_rel + binary gripper).
3. Install observation groups whose concatenated widths match actor=300 /
   critic=552 (DGPO layout in :mod:`~...dgpo_layout`).
4. When demos are available (``LIBERO_ASSEMBLED_DATASET_DIR``), install
   :class:`~...mdp.demos.commands.SourceLiberoCommand` so privileged
   EE/joint/object diffs are real, and reset objects/robot from demo
   ``initial_state``.

Quaternion order
----------------

* Sim math is always XYZW (Isaac Lab).
* ``policy_quat_order`` (default ``wxyz``) converts EE pose terms at the
  observation boundary — set to ``xyzw`` for a native Isaac Lab 3.x policy.
* ``demo_quat_order`` (default ``wxyz``) converts demo ``obs/ee_states`` on load.

Gym ids: ``Isaac-Hebero-All-State-v0`` (train) and
``Isaac-Hebero-All-State-Play-v0`` plus suite-subset ``*-Play-v0`` variants
(see :mod:`~...config.franka`).

Environment variables
---------------------

* ``LIBERO_ASSETS_DATA_DIR`` / ``LIBERO_CONFIG_DIR`` — USD + per-task JSON
  (required to build the harvest scene).
* ``LIBERO_ASSEMBLED_DATASET_DIR`` — HDF5 demo root
  (``{suite}_task{id}_*_demo.hdf5``). Defaults try ``datasets/demos`` (the
  Hugging Face pull) then the docker mount ``/data/libero/demos``.
* ``LIBERO_COMPAT_REQUIRE_DEMOS=1`` — fail hard when demos are missing.
* ``LIBERO_COMPAT_MAX_DEMOS_PER_TASK`` — optional positive int; load at most
  N demos per task (cuts HDF5 + pose-cache cost for play/debug).
* ``LIBERO_EVALUATION=1`` — disable random demo start-timestep sampling.
* ``GRIPPER_CURRICULUM`` (default ``1``) — SR-gated demo replay of the gripper:
  low-SR tasks follow the demo gripper command until their SR EMA crosses
  ``GRIPPER_CURRICULUM_SR_THRESHOLD`` (default ``0.2``). Training-only
  (requires demos; forced off under ``LIBERO_EVALUATION=1`` and in play cfgs).
  Set an expiry with ``env.actions.gripper.max_env_steps=<int>`` so a method whose
  SR never clears the threshold does not keep the demo gripper for the whole run.
* ``LIBERO_BOOTSTRAP_ON_TIMEOUT`` (default ``0``) — publish ``extras["time_outs"]``
  (``is_finite_horizon=False``) so a runner can bootstrap past horizon cuts.
  Leave disabled for the PPO configurations. See
  :data:`_BOOTSTRAP_ON_TIMEOUT` for the finite-horizon default.
* ``ROBOT_INIT_NOISE_STD`` — optional Gaussian noise on demo robot arm joints
  at reset (default ``0.0`` for eval; playground often uses ``0.05``).
* ``DGPO_ABC_CHECKPOINT`` — optional path for dim-contract checkpoint tests.

Train from scratch (harvest + OSC + DGPO obs)::

    export LIBERO_ASSETS_DATA_DIR=./datasets/USD
    export LIBERO_CONFIG_DIR=./config
    # optional demos for privileged critic / demo reset:
    export LIBERO_ASSEMBLED_DATASET_DIR=./datasets/demos
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Hebero-All-State-v0 --num_envs 2560 presets=physx

Eval smoke (once assets + demos are mounted)::

    export LIBERO_EVALUATION=1
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Hebero-All-State-Play-v0 --num_envs 40 \\
        --checkpoint /path/to/model.pt presets=physx --headless
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as env_mdp
from isaaclab.cloner import sequential
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.tasks.manipulation.multitask.config.demo.physics_cfg import MultitaskPhysicsCfg
from isaaclab_contrib.tasks.manipulation.utils.gripper_curriculum import CurriculumBinaryJointPositionActionCfg
from isaaclab_contrib.tasks.manipulation.multitask.registry import MultiTaskRegistry

from ...utils.task_sr import make_task_sr_curriculum_cfg
from ..assets import LIBERO_COMPLIANT_MATERIAL_CFG
from ..dgpo_layout import (
    DGPO_ABC_DECIMATION,
    DGPO_ABC_EPISODE_LENGTH_S,
    DGPO_ABC_HARVEST_SUITES,
    DGPO_ABC_SIM_DT,
    dgpo_abc_contract,
    harvest_task_to_assignment_key,
    verify_contract_arithmetic,
)
from ..mdp.combined import reset_libero_prototypes
from ..mdp.demos import events as dgpo_events
from ..mdp.demos.commands import DgpoDemoTaskConfig, build_env_task_assignments, make_dgpo_commands_cfg
from ..mdp.quat import DEFAULT_DEMO_QUAT_ORDER, DEFAULT_POLICY_QUAT_ORDER, QuatOrder
from ..robots.franka_osc import FRANKA_OSC
from ..tasks.harvest.combined import build_combined_tasks
from .dgpo_mdp_cfg import (
    DEFAULT_LIBERO_REWARD_MODE,
    LIBERO_REWARDS_CFGS,
    DgpoObservationsCfg,
    LiberoDgpoTerminationsCfg,
    resolve_reward_mode,
)
from ... import settings


# Default to a finite-horizon MDP (is_finite_horizon=True). IsaacLab publishes
# extras["time_outs"] only when is_finite_horizon=False; rsl-rl PPO then adds
# gamma * V(s_t) to the stored reward at timeouts. Keep that correction disabled
# for this benchmark. The privileged critic observes demo_phase, which exposes
# progress through the demonstration horizon.
_BOOTSTRAP_ON_TIMEOUT: bool = settings.truthy("LIBERO_BOOTSTRAP_ON_TIMEOUT", "0")


def _env_spacing_for(suites: tuple[tuple[str, str], ...]) -> float:
    """Return env spacing [m]: 3.0 when ``libero_long`` (larger scenes) is included, else 2.5."""
    return 3.0 if any(name == "libero_long" for name, _ in suites) else 2.5


def _install_single_franka_osc_actions(env_cfg) -> None:
    """Replace registry ScatteredActionTerm duplicates with one OSC + gripper.

    The factory registers FRANKA_OSC once per LIBERO task; the shared
    ``franka_robot`` then yields N identical OSC/gripper sub-terms. Install a
    single ``OperationalSpaceControllerAction`` + binary gripper instead.
    """
    specs = FRANKA_OSC.action_specs()
    # Preserve arm → gripper column order (6 + 1 = 7).
    env_cfg.actions.arm = specs["arm"][1]
    env_cfg.actions.gripper = specs["gripper"][1]


def _install_single_franka_reset_event(env_cfg) -> None:
    """Replace the per-task ``*_franka_reset_to_default`` copies with one term.

    The same duplication ``_install_single_franka_osc_actions`` fixes on the action
    side, on the reset side: the factory registers FRANKA_OSC once per LIBERO task, so
    :meth:`~...robots.franka_osc.LiberoFrankaOscRobotCfg.reset_events` contributes 40
    identical terms, all naming the one shared ``franka_robot``. They are not 40 slices
    of the work either -- ``mdp.reset_to_default`` reads only ``asset_cfg.name`` and
    ignores its ``selector``, so each term filtered and rewrote the default state of
    *every* resetting env. 40 copies of an idempotent write: one ``filter_reset_ids``
    and six device writes each, on every episode boundary.

    Collapsing them is behaviour-preserving because the write is idempotent (it comes
    from the fixed ``default_*`` buffers) and the surviving term keeps the position the
    copies held -- before ``libero_demo_reset``, which must still overwrite the joints
    with the demonstration's.
    """
    names = [n for n in vars(env_cfg.events) if n.endswith("_franka_reset_to_default")]
    if not names:
        return
    survivor = getattr(env_cfg.events, names[0])
    for name in names:
        setattr(env_cfg.events, name, None)
    for asset_cfg in survivor.params.get("asset_cfgs", ()):
        # Unscope it: the term now speaks for every env, not one clone group.
        asset_cfg.selector = None
    env_cfg.events.franka_reset_to_default = survivor


def make_libero_dgpo_env_cfg(
    suites: tuple[tuple[str, str], ...] = DGPO_ABC_HARVEST_SUITES,
    num_envs: int = 40,
    *,
    policy_quat_order: QuatOrder = DEFAULT_POLICY_QUAT_ORDER,
    demo_quat_order: QuatOrder = DEFAULT_DEMO_QUAT_ORDER,
) -> type:
    """Assemble harvest scene + OSC + DGPO observation groups (primary training path).

    Args:
        suites: ``(suite_name, prefix)`` pairs in task-index order.  Default is
            :data:`~...dgpo_layout.DGPO_ABC_HARVEST_SUITES` (libero_10 / object /
            spatial / goal) so sequential ``env i → task i % n`` matches the
            DGPO assignment order.
        num_envs: Number of parallel environments (must be >= number of tasks).
        policy_quat_order: Quat order emitted in EE pose / EE pose-diff obs
            (default ``wxyz`` for DGPO; use ``xyzw`` for native Isaac Lab 3.x).
        demo_quat_order: Quat order of demo ``obs/ee_states`` before conversion
            to sim XYZW (default ``wxyz``).

    Returns:
        A :class:`~isaaclab.envs.ManagerBasedRLEnvCfg` **subclass** with CloneCfg
        harvest layout, OSC actions (dim 7), and obs groups sized for
        actor=300 / critic=552.
    """
    if policy_quat_order not in ("wxyz", "xyzw"):
        raise ValueError(f"policy_quat_order must be 'wxyz' or 'xyzw', got {policy_quat_order!r}.")
    if demo_quat_order not in ("wxyz", "xyzw"):
        raise ValueError(f"demo_quat_order must be 'wxyz' or 'xyzw', got {demo_quat_order!r}.")

    verify_contract_arithmetic()
    contract = dgpo_abc_contract()

    task_cfgs, prototypes, bindings = build_combined_tasks(suites)
    if num_envs < len(task_cfgs):
        raise ValueError(f"num_envs ({num_envs}) must be >= number of tasks ({len(task_cfgs)}).")

    registry = MultiTaskRegistry()
    for task_cfg in task_cfgs:
        registry.register(FRANKA_OSC, task_cfg, group_name=task_cfg.name)

    base_cls = registry.build_env_cfg(
        num_envs=num_envs,
        env_spacing=_env_spacing_for(suites),
        replicate_physics=True,
        physics=MultitaskPhysicsCfg(),
        decimation=DGPO_ABC_DECIMATION,
        episode_length_s=DGPO_ABC_EPISODE_LENGTH_S,
        sim_dt=DGPO_ABC_SIM_DT,
    )
    parent_post_init = base_cls.__post_init__
    commands_cfg = make_dgpo_commands_cfg(bindings, num_envs, demo_quat_order=demo_quat_order)
    demos_wired = commands_cfg is not None and commands_cfg.libero_demo is not None

    # SR-gated gripper curriculum (playground ``GRIPPER_CURRICULUM``, default on).
    # Training-only: needs the demo command; forced off in eval mode.
    use_gripper_curriculum = (
        demos_wired
        and settings.truthy("GRIPPER_CURRICULUM", "1")
        and not settings.truthy("LIBERO_EVALUATION", "0")
    )
    gripper_curriculum_sr_threshold = settings.number("GRIPPER_CURRICULUM_SR_THRESHOLD", 0.2)
    # PhysX rigid contact spikes when the gripper closes; LIBERO's MuJoCo geoms are
    # compliant. Opt-in (``LIBERO_COMPLIANT_CONTACT=1``): it retunes contact for every
    # object, so a run either declares it or does not get it. ``truthy`` rather than
    # ``not_falsy`` so a malformed value stays off.
    use_compliant_contact = settings.truthy("LIBERO_COMPLIANT_CONTACT")

    meta = {
        "name": contract.name,
        "action_dim": contract.action_dim,
        "actor_obs_dim": contract.actor_obs_dim,
        "critic_obs_dim": contract.critic_obs_dim,
        "policy_dim": contract.policy.dim,
        "proprio_dim": contract.proprio.dim,
        "privileged_proprio_dim": contract.privileged_proprio.dim,
        "num_prototypes": len(prototypes),
        "num_tasks": len(task_cfgs),
        "policy_quat_order": policy_quat_order,
        "demo_quat_order": demo_quat_order,
        "mdp_status": {
            "multi_hot": "real",
            "buffer_pose": "real",
            "proprio": "real",
            "privileged_ee_joint_diff": "real_if_demos" if demos_wired else "zero_until_demos",
            "object_target_pose_diff": "real_if_demos" if demos_wired else "zero_until_demos",
            "demo_initial_state_reset": "real_if_demos" if demos_wired else "authored_prototypes",
            "demos_wired": demos_wired,
            "gripper_curriculum": use_gripper_curriculum,
        },
    }

    def _post_init(self):
        parent_post_init(self)
        self.is_finite_horizon = not _BOOTSTRAP_ON_TIMEOUT
        # Deterministic env→task map (same as harvest All env).
        self.scene.clone_cfg.clone_strategy = sequential
        # Match DGPO control rate (sim.dt * decimation = 0.05 s).
        self.decimation = DGPO_ABC_DECIMATION
        self.sim.dt = DGPO_ABC_SIM_DT
        self.sim.render_interval = DGPO_ABC_DECIMATION
        if use_compliant_contact:
            # Soft default contact, matching MuJoCo LIBERO (see the cfg's docstring).
            self.sim.physics_material = LIBERO_COMPLIANT_MATERIAL_CFG.copy()
        self.episode_length_s = DGPO_ABC_EPISODE_LENGTH_S
        # One OSC + gripper on the shared Franka (not N scattered copies), and one
        # robot reset event rather than one per task registration.
        _install_single_franka_osc_actions(self)
        _install_single_franka_reset_event(self)
        if use_gripper_curriculum:
            # Low-SR tasks replay the demo gripper (``source_action`` last column).
            # LIBERO assembled demos store the robosuite gripper sign (close >= 0),
            # opposite to BinaryJointPositionAction — invert (playground parity).
            plain_gripper = self.actions.gripper
            self.actions.gripper = CurriculumBinaryJointPositionActionCfg(
                asset_name=plain_gripper.asset_name,
                joint_names=plain_gripper.joint_names,
                open_command_expr=plain_gripper.open_command_expr,
                close_command_expr=plain_gripper.close_command_expr,
                source_command_name="source_action",
                command_index=-1,
                invert_source_command_sign=True,
                sr_threshold=gripper_curriculum_sr_threshold,
            )
        self.observations = DgpoObservationsCfg()
        if commands_cfg is not None:
            self.commands = commands_cfg
        # First-class quat-order flags (obs / demo boundaries).
        self.policy_quat_order = policy_quat_order
        self.demo_quat_order = demo_quat_order
        self.dgpo_layout = meta
        self.dgpo_task_bindings = bindings
        # Playground-compatible task-assignment view (``cfg.libero_config``) for
        # the RFCL curriculum and demo_focus helpers.
        if commands_cfg is not None and commands_cfg.libero_demo is not None:
            self.libero_config = commands_cfg.libero_demo.libero_config
        else:
            _assignments = build_env_task_assignments(bindings, num_envs)
            _uniq = tuple(dict.fromkeys(_assignments))
            self.libero_config = DgpoDemoTaskConfig(
                env_task_assignments=_assignments, task_sequence=_uniq, full_task_sequence=_uniq
            )
        # Short-lived aliases for older call sites.
        self.compat_dim_contract = meta
        self.compat_task_bindings = bindings

        # ---- terminations: declarative cfg (playground scheme: success terminates
        # in eval only; training uses timeout + command depletion + joint limits) ----
        # Harvest prototype tasks leave termination_terms empty; replace the
        # registry's bare time_out with the full declarative cfg.
        self.terminations = LiberoDgpoTerminationsCfg()
        # command_depleted reads the demo command's progress metric — without
        # demos (plain PPO fallback) swap it for a plain fixed-length timeout
        # so episodes still have a horizon.
        if not demos_wired:
            self.terminations.command_depleted = None
            self.terminations.time_out = DoneTerm(func=env_mdp.time_out, time_out=True)

        # ---- curriculum: per-task SR EMA tracking (native episode-end hook) ----
        # Owns the TaskSRTracker (env.task_sr_tracker): Metrics/task_sr/* logs
        # and DGPO algorithm extras come from the RSL-RL wrapper, the gripper
        # curriculum action reads its replay mask.
        self.curriculum = make_task_sr_curriculum_cfg(
            labels=[harvest_task_to_assignment_key(b.name, b.suite) for b in bindings],
            success_term_names=("libero_success",),
            teacher_command_name="source_action" if demos_wired else None,
        )

        # ---- rewards: pick one declarative playground-style mode cfg ----
        # (registry only installs ramped penalties -> no task reward at all)
        reward_mode = resolve_reward_mode(settings.text("LIBERO_REWARD_MODE", DEFAULT_LIBERO_REWARD_MODE))
        if reward_mode == "world_state_tracking_reward" and not demos_wired:
            import warnings

            warnings.warn(
                "LIBERO_REWARD_MODE=world_state_tracking_reward requires assembled demos "
                "(LIBERO_ASSEMBLED_DATASET_DIR); falling back to the sparse goal reward.",
                stacklevel=2,
            )
            reward_mode = "goal"
        self.rewards = LIBERO_REWARDS_CFGS[reward_mode]()
        meta["reward_mode"] = reward_mode

        # Reset path: demo initial_state when traj bank is present; else authored poses.
        if demos_wired:
            # Drop per-group object jitter so demo poses are not overwritten.
            for attr_name in list(vars(self.events).keys()):
                if attr_name.endswith("_reset_objects_default") or attr_name.endswith("_reset_primary_uniform"):
                    setattr(self.events, attr_name, None)
            noise = settings.number("ROBOT_INIT_NOISE_STD", 0.0)
            self.events.libero_demo_reset = EventTerm(
                func=dgpo_events.reset_libero_scene_to_demo_initial_state,
                mode="reset",
                params={
                    "command_name": "source_action",
                    "include_articulations": True,
                    "include_rigid_objects": True,
                    "robot_joint_noise_std": noise,
                },
            )
        else:
            self.events.libero_reset_objects = EventTerm(
                func=reset_libero_prototypes,
                mode="reset",
                params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.0)}},
            )

    return configclass(type("_LiberoDgpoOscEnvCfg", (base_cls,), {"__post_init__": _post_init}))


def make_libero_dgpo_play_cfg(
    suites: tuple[tuple[str, str], ...] = DGPO_ABC_HARVEST_SUITES,
    *,
    policy_quat_order: QuatOrder = DEFAULT_POLICY_QUAT_ORDER,
    demo_quat_order: QuatOrder = DEFAULT_DEMO_QUAT_ORDER,
) -> type:
    """Play/eval variant: one env per task, no observation corruption."""
    num_tasks = sum(10 for _ in suites)  # each suite contributes 10 tasks
    # Prefer exact count from harvest when configs are available.
    try:
        task_cfgs, _, _ = build_combined_tasks(suites)
        num_tasks = len(task_cfgs)
    except FileNotFoundError:
        pass
    base_cls = make_libero_dgpo_env_cfg(
        suites=suites,
        num_envs=num_tasks,
        policy_quat_order=policy_quat_order,
        demo_quat_order=demo_quat_order,
    )
    parent_post_init = base_cls.__post_init__

    def _post_init(self):
        parent_post_init(self)
        self.observations.policy.enable_corruption = False
        self.observations.proprio.enable_corruption = False
        self.observations.privileged_proprio.enable_corruption = False
        # Play always evaluates the policy's own gripper (no demo replay).
        gripper_cfg = self.actions.gripper
        if isinstance(gripper_cfg, CurriculumBinaryJointPositionActionCfg):
            self.actions.gripper = BinaryJointPositionActionCfg(
                asset_name=gripper_cfg.asset_name,
                joint_names=gripper_cfg.joint_names,
                open_command_expr=gripper_cfg.open_command_expr,
                close_command_expr=gripper_cfg.close_command_expr,
            )

    return configclass(type("_LiberoDgpoOscPlayEnvCfg", (base_cls,), {"__post_init__": _post_init}))
