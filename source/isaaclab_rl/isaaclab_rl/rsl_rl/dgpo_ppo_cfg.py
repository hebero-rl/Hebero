# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`~.dgpo_ppo.DGPO` (adaptive BC / DAPG / importance weights)."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .rl_cfg import RslRlPpoAlgorithmCfg


@configclass
class RslRlDgpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO + DGPO extensions (see :class:`~.dgpo_ppo.DGPO` for semantics).

    Every field maps 1:1 to a ``DGPO`` constructor kwarg. With all feature
    flags off this behaves exactly like stock PPO (the custom update path is
    bypassed), so it can also serve as the plain MT-PPO baseline config.
    """

    class_name: str = "isaaclab_rl.rsl_rl.dgpo_ppo:DGPO"

    # ---- importance-weighted PPO losses ----
    use_importance_weights: bool = False
    """Reweight surrogate/value/entropy per sample by ``extras["importance_weights"]``."""
    importance_weight_actor_only: bool = False
    """Reweight only the actor's terms, leaving the value loss at a plain mean.

    ``False`` weights the surrogate, value, and entropy terms. ``True`` leaves
    the critic regression unweighted while retaining task weights on the policy
    surrogate and entropy bonus.
    """
    importance_weight_key: str = "importance_weights"
    importance_weight_normalization: str = "mean"
    """``mean`` divides each mini-batch's weights by their mean (keeps loss scale)."""
    importance_weight_warmup_steps: int = 1000
    """Steps during which stored weights are forced to 1 (no reweighting)."""
    iw_from_task_sr: bool = False
    """Derive weights from the per-task SR EMA (below-mean tasks -> ``iw_max``)."""
    iw_max: float = 2.0
    iw_min: float = 0.5
    iw_dynamic_range: float | None = None
    """Peak-to-floor ratio overriding ``iw_min``/``iw_max``.

    Mean normalization removes absolute weight scale. Incompatible with
    ``iw_min_final``.
    """
    iw_min_final: float | None = None
    """Final IW floor after ``anneal_steps``; ``None`` keeps the floor constant."""
    iw_sr_balance_slope: float = 10.0
    """Sigmoid slope of the SR-relative-to-mean balancing (``iw_sr_shape='sigmoid'``)."""
    iw_sr_shape: str = "sigmoid"
    """SR-to-loss-weight mapping: ``sigmoid`` favors below-mean SR;
    ``beta`` peaks at ``iw_beta_target``. Neither changes rollout sampling.
    """
    iw_beta_target: float = -1.0
    """Beta-kernel target SR: negative follows the task mean; [0, 1] fixes the mode.

    A float sentinel avoids Hydra's NoneType restriction on numeric overrides.
    """
    iw_beta_target_final: float | None = None
    """Final beta target after ``anneal_steps``; ``None`` keeps the target fixed.

    Ignored when ``iw_beta_target`` is negative (follows the mean).
    """
    iw_beta_concentration: float = 1.0
    """Beta concentration ``kappa``; higher concentrates weight nearer the target."""
    iw_beta_eps: float = 1e-8
    """Guards the kernel at SR in {0, 1} so no task's weight reaches exactly the floor."""

    # ---- adaptive BC (online teacher from extras["teacher_actions"]) ----
    use_adaptive_bc: bool = False
    """Add a supervised MSE loss toward the demo teacher actions."""
    teacher_action_key: str = "teacher_actions"
    task_assignment_id_key: str = "task_assignment_ids"
    task_success_key: str = "task_success"
    bc_loss_coef_base: float = 1.0
    """Global multiplier on the BC loss."""
    bc_loss_coef_min: float = 0.0
    """BC weight floor; reached above ``bc_sr_high_threshold`` in ramp mode."""
    bc_loss_coef_max: float = 1.0
    """Per-task weight when the task's SR EMA <= ``bc_sr_low_threshold``."""
    bc_loss_coef_max_final: float | None = None
    """Final BC ceiling after ``anneal_steps``; ``None`` keeps the ceiling constant."""
    bc_sr_shape: str = "ramp"
    """BC schedule: ``ramp`` interpolates between the SR thresholds; ``beta``
    uses ``(1 - effective_sr)^kappa``, incorporating success and elapsed time.
    Leave ``bc_loss_coef_max_final`` unset with beta to avoid compounded annealing.
    """
    bc_beta_concentration: float = 2.0
    """``kappa`` for ``bc_sr_shape='beta'``; larger values yield faster decay.

    Applies to both the success-rate and time-annealing factors through
    ``effective_sr``. For example, ``kappa=2`` gives a time factor of 0.25
    halfway through annealing.
    """
    bc_sr_low_threshold: float = 0.1
    """Only used by ``bc_sr_shape='ramp'``."""
    bc_sr_high_threshold: float = 0.5
    """Only used by ``bc_sr_shape='ramp'``."""
    bc_sr_ema_alpha: float = 0.05
    """EMA smoothing for the per-task success rate driving the weight lerp."""
    bc_gripper_action_dim: int = 0
    """Trailing action dims treated as gripper (for optional sign inversion)."""
    bc_invert_gripper_target_sign: bool = False
    bc_pretrain_only: bool = False
    """Zero PPO surrogate/value/entropy: run pure BC (policy initialization)."""
    use_advantage_gate: bool = False
    """Gate per-sample bc_weight by 1[A(s, teacher) > A(s, action)] (IS estimate)."""

    # ---- time annealing (shared clock for iw_min_final / bc_loss_coef_max_final) ----
    anneal_steps: int = 0
    """Env steps the time axis spans; ``0`` disables it.

    Counted in rollout steps (one per ``process_env_step``) and carried through
    ``save``/``load``, so a resumed run continues the schedule rather than
    restarting it.
    """

    # ---- DAPG demo log-likelihood loss ----
    use_dapg_demo_loss: bool = False
    """Add -log pi(a_demo | s_demo), weighted by the per-task SR lerp."""
    dapg_loss_coef_base: float = 1.0
    dapg_demo_hdf5_paths: list[str] | None = None
    """Explicit demo file list (offline pool). Overrides ``dapg_demo_hdf5_dir``."""
    dapg_demo_hdf5_dir: str | None = None
    """Directory of preprocessed demos (``obs/{group}`` + ``actions``). When
    neither is set, DAPG falls back to the ONLINE rollout teacher actions."""
    dapg_task_ids: list[int] | None = None
    """Canonical task id per demo file (defaults to the file order)."""
    dapg_obs_groups: list[str] | None = None
    """Demo obs groups matching the actor's obs groups (default policy+proprio)."""
    dapg_max_demo_samples: int = 50000
    dapg_detach_std: bool = True
    """Detach sigma in the DAPG log-prob; demonstrations update the mean only."""
    dapg_sr_adaptive_weight: bool = True
    """Weight the DAPG demo term by the per-task SR lerp as well as by lambda(k).

    ``False`` uses only ``lambda_0 * lambda_1^k`` for the schedule. ``True``
    additionally interpolates between ``bc_loss_coef_max`` and
    ``bc_loss_coef_min`` according to each task's success-rate EMA.
    """
    dapg_lambda0: float = 0.1
    """Initial DAPG schedule scale (Rajeswaran et al., 2018), multiplying
    ``dapg_loss_coef_base``.
    """
    dapg_lambda1: float = 0.995
    """Per-iteration decay of the DAPG demo term (``lambda_1``); ``1.0`` disables it.

    ``k`` is incremented once per :meth:`update`, not per epoch or mini-batch, and
    is carried through ``save``/``load`` so a resumed run continues the decay
    instead of restarting it -- same treatment as ``anneal_steps``.

    When ``dapg_sr_adaptive_weight=True``, iteration decay multiplies the
    per-task SR weight. Set ``dapg_lambda1=1.0`` to disable iteration decay.
    """
    # ---- misc ----
    std_floor: float = 0.0
    """Clamp the Gaussian sigma from below after each optimizer step (0 = off)."""
    std_max: float = 0.0
    """Clamp the Gaussian sigma from above after each optimizer step (0 = off).

    Complements ``std_floor`` by limiting exploration variance. The entropy
    bonus encourages larger sigma, while mean-action BC does not directly
    constrain ``log_std``. Reference configurations set the bound explicitly.
    """
    lr_min: float = 1e-5
    """Lower bound of the adaptive-KL learning-rate schedule (``schedule='adaptive'``).

    Default matches the value hardcoded in upstream RSL-RL.
    """
    lr_max: float = 1e-2
    """Upper bound of the adaptive-KL learning-rate schedule (``schedule='adaptive'``).

    The default matches upstream RSL-RL. The schedule updates per mini-batch,
    so multiple adjustments can occur within one PPO iteration. Reference
    configurations may use a tighter bound relative to ``learning_rate``.
    """
    log_per_task_metrics: bool = False
    """Kept for playground parity; per-task SR curves come from the env-side
    ``Metrics/task_sr/*`` logging."""
