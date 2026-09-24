# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING, field

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDgpoAlgorithmCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlTokenPoolModelCfg,
)
from isaaclab_rl.rsl_rl.rfcl_cfg import RslRlRFCLCfg
from ..... import settings

# Assembled demonstrations provide reset states for the PPO RFCL curriculum.
_ASSEMBLED_DEMO_DIR = os.getenv(
    "LIBERO_ASSEMBLED_DATASET_DIR",
    "datasets/demos",
)

# Shared asymmetric critic toggle (``LIBERO_PRIVILEGED_CRITIC``).
# Every method uses the same observation groups.
_CRITIC_OBS = ["policy", "proprio"]
if settings.not_falsy("LIBERO_PRIVILEGED_CRITIC"):
    _CRITIC_OBS = ["policy", "proprio", "privileged_proprio"]

# Per-task success-rate rebalancing in IW variants. The shared PPO base enables
# IW; plain PPO and ABC select uniform weighting through a launch-time override.
# DAPG overrides IW below, and its IW variant enables it at launch. These weights
# rebalance tasks by success rate, independently of PPO's policy-ratio clipping.
_IW_MIN, _IW_MAX = 0.5, 2.0

# Shared PPO defaults, including SR weighting and stability bounds.
# Expand this dictionary in each algorithm config: configclass subclasses replace
# nested fields wholesale rather than merging their individual settings.
_PPO_ALGO_BASE = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.15,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=2.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.005,
    max_grad_norm=1.0,
    # Per-task SR rebalancing: enabled in the base; variants may override it.
    use_importance_weights=True,
    iw_from_task_sr=True,
    iw_max=_IW_MAX,
    iw_min=_IW_MIN,
    # Bound the adaptive learning rate and policy standard deviation for stability.
    # Reference learners inherit these bounds unless explicitly overridden below.
    lr_max=3.0e-4,
    std_max=0.6,
    std_floor=0.05,
    log_per_task_metrics=True,
)

# The ABC terms that distinguish IW-ABC from IW-PPO. Kept as a dict, like _PPO_ALGO_BASE,
# so the state-based and vision arms of the method are one definition rather than two --
# a configclass turns a class attribute into a dataclass field, so a subclass cannot
# reach ``ParentCfg.algorithm`` to reuse it.
_DGPO_ALGO_EXTRA = dict(
    use_adaptive_bc=True,
    bc_loss_coef_min=0.1,
    bc_loss_coef_max=1.0,
    bc_sr_low_threshold=0.1,
    bc_sr_high_threshold=0.5,
    bc_gripper_action_dim=1,
    # LIBERO assembled demos store the robosuite gripper sign (close >= 0),
    # opposite to BinaryJointPositionAction; flip the BC target.
    bc_invert_gripper_target_sign=True,
)

# Likewise for MT-DAPG.  std_floor 0.1 rather than the shared 0.05: DAPG supervises sigma
# through the demo log-prob and will drive it to 0 as mu -> a_demo, so this arm needs a
# higher floor than the others.
_DAPG_ALGO_EXTRA = dict(
    use_dapg_demo_loss=True,
    dapg_obs_groups=["policy", "proprio"],
    # DAPG demonstration-loss decay: lambda_0 * lambda_1^k over policy iterations
    # (Rajeswaran et al., 2018).
    dapg_lambda0=0.1,
    dapg_lambda1=0.995,
    # DAPG disables SR-adaptive imitation and PPO task weighting by default.
    # Enable use_importance_weights at launch for IW-DAPG.
    # The per-task SR EMA remains available for logging.
    dapg_sr_adaptive_weight=False,
    use_importance_weights=False,
    # Not algorithm choices, data-format corrections: the demo action vector's
    # trailing dim is the gripper, and LIBERO's assembled demos store the robosuite
    # sign (close >= 0), opposite to BinaryJointPositionAction. Both arms need this
    # or the teacher action means the wrong thing.
    bc_gripper_action_dim=1,
    bc_invert_gripper_target_sign=True,
    # Numerical guard, not a mechanism: DAPG's log-likelihood reaches log_std through
    # the demo log-prob and drives it to 0 as mu -> a_demo. dapg_detach_std (on by
    # default) removes the gradient; this floor catches what is left.
    std_floor=0.1,
)


@configclass
class LiberoAllMtPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Shared PPO configuration for the DGPO training framework.

    The existing base enables IW and disables BC/DAPG. Plain PPO passes
    ``agent.algorithm.use_importance_weights=False``. Other on-policy configs
    inherit the network, PPO hyperparameters and observation groups.

    Actor and critic architecture:
    ``[512, 256, 128]`` MLPs with ``noise_std_type="log"``.
    """

    num_steps_per_env = 16
    max_iterations = 30000
    save_interval = 100
    experiment_name = "libero_all_mtppo"
    # rsl-rl >= 4.0 uses the "actor" key (not legacy "policy"). Missing "actor"
    # falls back to the env's single "policy" group (277) and breaks ckpt load.
    # Actor: policy(277)+proprio(23)=300; critic: +privileged_proprio(252)=552
    # (LIBERO_PRIVILEGED_CRITIC=0 → symmetric critic=300).  The 24-dim
    # gripper_force history is in neither group; see dgpo_layout.
    obs_groups = {
        "actor": ["policy", "proprio"],
        "critic": _CRITIC_OBS,
    }
    run_name = ""
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        noise_std_type="log",
    )
    # RslRlDgpoAlgorithmCfg with every loss flag off is stock PPO -- the custom
    # update path is bypassed -- but it is what carries the importance-weight
    # fields, which plain RslRlPpoAlgorithmCfg's algorithm class would ignore.
    algorithm = RslRlDgpoAlgorithmCfg(**_PPO_ALGO_BASE)


@configclass
class LiberoAllIwAbcRunnerCfg(LiberoAllMtPPORunnerCfg):
    """IW-ABC: adaptive BC toward the demo teacher + SR-derived importance weights.

    The full proposed loss.  The per-task BC weight lerps from ``bc_loss_coef_max``
    to ``bc_loss_coef_min`` as that task's SR EMA rises; the PPO losses are
    reweighted toward below-mean tasks.  Teacher actions come from the env's
    ``source_action`` demo command (wired into ``extras["teacher_actions"]`` by the
    RSL-RL wrapper via the task-SR tracker).

    Adds adaptive BC to :class:`LiberoAllMtPPORunnerCfg`; the network, PPO
    hyperparameters, importance weights, and stability bounds are inherited.
    """

    experiment_name = "libero_all_iw_abc"
    algorithm = RslRlDgpoAlgorithmCfg(**_PPO_ALGO_BASE, **_DGPO_ALGO_EXTRA)


@configclass
class LiberoAllPpoRfclRunnerCfg(LiberoAllMtPPORunnerCfg):
    """PPO with RFCL reverse/forward curriculum reset states and no BC loss.

    Inherits the shared PPO network, hyperparameters, and task-SR weighting.
    ``RFCLEnvWrapper`` replaces reset states with curriculum-selected demo states:
    Phase 1 moves a reverse cursor through each demo; Phase 2 resets at the
    start once a task's active cursors reach the beginning. The wrapper leaves
    PPO's rollout buffer and loss unchanged.

    ``train.py`` attaches the wrapper when ``rfcl_cfg`` is populated and logs
    curriculum progress through the on-policy runner's metrics hook."""

    experiment_name = "libero_all_ppo_rfcl"
    rfcl_cfg: RslRlRFCLCfg = field(
        default_factory=lambda: RslRlRFCLCfg(
            demo_hdf5_dir=_ASSEMBLED_DEMO_DIR,
            init_cursor_progress=0.85,
            reverse_step_size=8,
            sr_ema_alpha=0.05,
            phase1_demo_count=10,
        )
    )


@configclass
class LiberoAllMtDapgRunnerCfg(LiberoAllMtPPORunnerCfg):
    """MT-DAPG (Rajeswaran et al. 2018): PPO + demo log-likelihood, decayed by lambda(k).

    The demo term uses ``lambda_0 * lambda_1^k`` without SR-adaptive weighting.
    PPO task weighting is disabled by default; enable
    ``agent.algorithm.use_importance_weights=True`` for IW-DAPG.

    Offline demo pool from ``LIBERO_DAPG_DEMO_DIR`` (preprocessed
    ``obs/policy + obs/proprio + actions`` HDF5). When unset, DAPG falls back
    to the ONLINE rollout teacher actions (same states, no second forward); a
    directory that is set but unreadable raises instead of falling back.
    """

    experiment_name = "libero_all_mt_dapg"
    algorithm = RslRlDgpoAlgorithmCfg(
        **{**_PPO_ALGO_BASE, **_DAPG_ALGO_EXTRA},
        dapg_demo_hdf5_dir=settings.text("LIBERO_DAPG_DEMO_DIR") or None,
    )


# ---------------------------------------------------------------------------
# Vision (frozen Theia + attention pool) variants
# ---------------------------------------------------------------------------
# These pair with the ``*-Vision-v0`` gym ids.  The actor's ground-truth object-pose
# block (``buffer_pose``) is gone from the env; ``perception`` carries Theia patch
# tokens in its place, and :class:`~isaaclab_rl.rsl_rl.models.TokenPoolModel` pools them
# with a learned single-query attention head before the same [512, 256, 128] MLP.
#
# The critic still reads ``privileged_proprio`` when LIBERO_PRIVILEGED_CRITIC is on, so
# the asymmetry the state-based line already has is unchanged -- what moved is the
# ACTOR's access to object state, which is the point of the variant.
_VISION_ACTOR_OBS = ["policy", "proprio", "perception"]
_VISION_CRITIC_OBS = [*_CRITIC_OBS, "perception"]


def _vision_model_cfg(stochastic: bool) -> RslRlTokenPoolModelCfg:
    """Token-pool model cfg matching the state-based MLP shape.

    Same trunk as the state-based line ([512, 256, 128], elu, log-std 0.8) so the vision
    arm differs from it by its observations, not by its capacity.
    """
    distribution = None
    if stochastic:
        distribution = RslRlTokenPoolModelCfg.GaussianDistributionCfg(init_std=0.8, std_type="log")
    return RslRlTokenPoolModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=distribution,
    )


@configclass
class LiberoAllMtPPOVisionRunnerCfg(LiberoAllMtPPORunnerCfg):
    """MT-PPO on Theia tokens. Use with ``Isaac-Hebero-All-Vision-v0``."""

    experiment_name = "libero_all_mtppo_vision"
    obs_groups = {"actor": _VISION_ACTOR_OBS, "critic": _VISION_CRITIC_OBS}
    # The 3D ``perception`` group is what makes the legacy ActorCritic path unusable, so
    # these configs use the rsl-rl >= 4.0 actor/critic model keys rather than ``policy``.
    # Cleared to MISSING (not None): handle_deprecated_rsl_rl_cfg treats any non-MISSING
    # ``policy`` as a legacy config to migrate, and would warn about a None one.
    policy = MISSING
    actor: RslRlTokenPoolModelCfg = field(default_factory=lambda: _vision_model_cfg(stochastic=True))
    critic: RslRlTokenPoolModelCfg = field(default_factory=lambda: _vision_model_cfg(stochastic=False))


@configclass
class LiberoAllVisIwAbcRunnerCfg(LiberoAllMtPPOVisionRunnerCfg):
    """Vis IW-ABC (adaptive BC + SR importance weights) on Theia tokens.

    Differs from :class:`LiberoAllMtPPOVisionRunnerCfg` by the same loss terms that
    separate IW-ABC from MT-PPO in the state-based line, so the two comparisons stay
    parallel.
    """

    experiment_name = "libero_all_vis_iw_abc"
    algorithm = RslRlDgpoAlgorithmCfg(**_PPO_ALGO_BASE, **_DGPO_ALGO_EXTRA)


@configclass
class LiberoAllMtDapgVisionRunnerCfg(LiberoAllMtPPOVisionRunnerCfg):
    """MT-DAPG on Theia tokens.

    ``dapg_demo_hdf5_dir`` stays unset unless the preprocessed demos carry an
    ``obs/perception`` column -- the offline demo loss feeds recorded observations
    through the actor, and the low-dim HDF5s have no image features. Without it DAPG
    falls back to the online rollout teacher, which works unchanged here.
    """

    experiment_name = "libero_all_mt_dapg_vision"
    algorithm = RslRlDgpoAlgorithmCfg(
        **{**_PPO_ALGO_BASE, **_DAPG_ALGO_EXTRA},
        # Left unset: the offline demo loss feeds recorded observations through the actor
        # and the low-dim HDF5s have no ``obs/perception`` column.
        dapg_demo_hdf5_dir=None,
    )


# Import compatibility for saved configs using the previous method names.
LiberoAllDgpoRunnerCfg = LiberoAllIwAbcRunnerCfg
LiberoAllDgpoVisionRunnerCfg = LiberoAllVisIwAbcRunnerCfg
