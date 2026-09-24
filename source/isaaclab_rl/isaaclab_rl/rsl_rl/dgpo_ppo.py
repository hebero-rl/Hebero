# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional extensions to RSL-RL >= 5 actor/critic-split PPO.

Select ``isaaclab_rl.rsl_rl.dgpo_ppo:DGPO`` through the algorithm's ``class_name``.

* ``use_importance_weights`` weights PPO surrogate, value, and entropy losses
  using supplied weights or task SR; ``importance_weight_actor_only`` excludes
  the value loss.
* ``use_adaptive_bc`` adds teacher-action MSE with a task-SR ramp or a
  success-and-time beta schedule, selected by ``bc_sr_shape``.
* ``use_dapg_demo_loss`` adds demonstration log-likelihood loss from offline
  observations/actions or online teacher actions. Its iteration decay is
  ``dapg_lambda0 * dapg_lambda1**k``; SR weighting is optional.
* ``bc_pretrain_only`` disables PPO surrogate, value, and entropy terms.

``TaskSRTracker`` computes the SR EMA; this class mirrors it via
``RslRlVecEnvWrapper`` and saves it under ``dgpo_state``. On resume, restore
that state into the newly created tracker with ``TaskSRTracker.restore``
(see ``train.py``). Older checkpoints without ``dgpo_state`` remain loadable.

Extras contract: ``teacher_actions`` (num_envs, A), ``task_assignment_ids`` and
``task_success`` (num_envs,), ``task_sr_ema`` and ``task_sr_warmed`` (n_tasks,),
and optional ``task_assignment_names`` (tuple). Parameter details are in
``RslRlDgpoAlgorithmCfg``.
"""

from __future__ import annotations

import glob
import math
import os
import warnings

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO

from .task_sr_weights import effective_sr, envelope_from_range, task_sr_weight, validate_iw_sr_shape


class DGPO(PPO):
    """PPO with importance-weighted losses, adaptive BC, and DAPG demo losses."""

    def __init__(
        self,
        *args,
        use_importance_weights: bool = False,
        importance_weight_actor_only: bool = False,
        importance_weight_key: str = "importance_weights",
        importance_weight_normalization: str = "mean",
        importance_weight_warmup_steps: int = 1000,
        iw_from_task_sr: bool = False,
        iw_max: float = 2.0,
        iw_min: float = 0.5,
        iw_sr_balance_slope: float = 10.0,
        iw_sr_shape: str = "sigmoid",
        iw_beta_target: float = -1.0,
        iw_beta_concentration: float = 1.0,
        iw_beta_eps: float = 1e-8,
        iw_dynamic_range: float | None = None,
        iw_min_final: float | None = None,
        iw_beta_target_final: float | None = None,
        bc_sr_shape: str = "ramp",
        bc_beta_concentration: float = 1.0,
        bc_loss_coef_max_final: float | None = None,
        anneal_steps: int = 0,
        use_adaptive_bc: bool = False,
        teacher_action_key: str = "teacher_actions",
        task_assignment_id_key: str = "task_assignment_ids",
        task_success_key: str = "task_success",
        task_sr_ema_key: str = "task_sr_ema",
        task_sr_warmed_key: str = "task_sr_warmed",
        bc_loss_coef_base: float = 1.0,
        bc_loss_coef_min: float = 0.05,
        bc_loss_coef_max: float = 1.0,
        bc_sr_low_threshold: float = 0.1,
        bc_sr_high_threshold: float = 0.5,
        bc_sr_ema_alpha: float = 0.05,
        bc_gripper_action_dim: int = 0,
        bc_invert_gripper_target_sign: bool = False,
        bc_pretrain_only: bool = False,
        use_dapg_demo_loss: bool = False,
        dapg_loss_coef_base: float = 1.0,
        dapg_demo_hdf5_paths: list[str] | None = None,
        dapg_demo_hdf5_dir: str | None = None,
        dapg_task_ids: list[int] | None = None,
        dapg_obs_groups: list[str] | None = None,
        dapg_max_demo_samples: int = 50000,
        dapg_detach_std: bool = True,
        dapg_lambda0: float = 0.1,
        dapg_lambda1: float = 0.995,
        dapg_sr_adaptive_weight: bool = True,
        std_floor: float = 0.0,
        std_max: float = 0.0,
        lr_min: float = 1e-5,
        lr_max: float = 1e-2,
        use_advantage_gate: bool = False,
        log_per_task_metrics: bool = False,
        **kwargs,
    ):
        """Store the DGPO knobs, then build stock PPO and the side buffers."""
        self.use_importance_weights = use_importance_weights
        self.importance_weight_actor_only = importance_weight_actor_only
        self.importance_weight_key = importance_weight_key
        self.importance_weight_normalization = importance_weight_normalization
        self.importance_weight_warmup_steps = importance_weight_warmup_steps
        self.iw_from_task_sr = iw_from_task_sr
        self.iw_max = iw_max
        self.iw_min = iw_min
        self.iw_sr_balance_slope = iw_sr_balance_slope
        self.iw_sr_shape = validate_iw_sr_shape(iw_sr_shape)
        self.iw_beta_target = iw_beta_target
        self.iw_beta_concentration = iw_beta_concentration
        self.iw_beta_eps = iw_beta_eps
        if iw_dynamic_range is not None:
            if iw_min_final is not None:
                raise ValueError(
                    "iw_dynamic_range supersedes the envelope, so iw_min_final has nothing to "
                    "anneal; use iw_beta_target_final to move the mode over time instead."
                )
            self.iw_min, self.iw_max = envelope_from_range(iw_dynamic_range)
        self.iw_dynamic_range = iw_dynamic_range
        self.iw_min_final = iw_min_final
        self.iw_beta_target_final = iw_beta_target_final
        self.bc_sr_shape = bc_sr_shape
        self.bc_beta_concentration = bc_beta_concentration
        self.bc_loss_coef_max_final = bc_loss_coef_max_final
        self.anneal_steps = int(anneal_steps)
        self.use_adaptive_bc = use_adaptive_bc
        self.teacher_action_key = teacher_action_key
        self.task_assignment_id_key = task_assignment_id_key
        self.task_success_key = task_success_key
        self.task_sr_ema_key = task_sr_ema_key
        self.task_sr_warmed_key = task_sr_warmed_key
        self.bc_loss_coef_base = bc_loss_coef_base
        self.bc_loss_coef_min = bc_loss_coef_min
        self.bc_loss_coef_max = bc_loss_coef_max
        self.bc_sr_low_threshold = bc_sr_low_threshold
        self.bc_sr_high_threshold = bc_sr_high_threshold
        # No longer used: the SR EMA is now mirrored from the env's tracker
        # (extras[task_sr_ema_key]), which owns its own alpha. Kept as an
        # accepted cfg field so existing agent configs don't fail to construct.
        self.bc_sr_ema_alpha = bc_sr_ema_alpha
        self.bc_gripper_action_dim = bc_gripper_action_dim
        self.bc_invert_gripper_target_sign = bc_invert_gripper_target_sign
        self.bc_pretrain_only = bc_pretrain_only
        self.use_dapg_demo_loss = use_dapg_demo_loss
        self.dapg_loss_coef_base = dapg_loss_coef_base
        self.dapg_demo_hdf5_paths = dapg_demo_hdf5_paths
        self.dapg_demo_hdf5_dir = dapg_demo_hdf5_dir
        self.dapg_task_ids = dapg_task_ids
        self.dapg_obs_groups = dapg_obs_groups if dapg_obs_groups is not None else ["policy", "proprio"]
        self.dapg_max_demo_samples = dapg_max_demo_samples
        self.dapg_detach_std = dapg_detach_std
        self.dapg_lambda0 = dapg_lambda0
        self.dapg_lambda1 = dapg_lambda1
        self.dapg_sr_adaptive_weight = dapg_sr_adaptive_weight
        self.std_floor = std_floor
        self.std_max = std_max
        if lr_min > lr_max:
            raise ValueError(f"lr_min ({lr_min}) must not exceed lr_max ({lr_max}).")
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.use_advantage_gate = use_advantage_gate
        self.log_per_task_metrics = log_per_task_metrics

        self._dapg_offline_loaded: bool = False
        # k in DAPG's lambda_0 * lambda_1^k. Counts custom update() calls (policy
        # iterations), not epochs or mini-batches, and is persisted by save/load.
        self._dapg_update_count: int = 0
        self._importance_weights: torch.Tensor | None = None
        self._importance_weight_step_counter: int = 0
        # Annealing clock.  Separate from the IW warmup counter above, which only
        # ticks while importance weights are on; the BC schedule needs one either way.
        self._anneal_step_counter: int = 0
        self._bc_teacher_actions: torch.Tensor | None = None
        self._adaptive_bc_weights: torch.Tensor | None = None
        self._task_assignment_ids: torch.Tensor | None = None
        self._task_assignment_names: tuple[str, ...] | None = None
        self._task_sr_ema: torch.Tensor | None = None
        self._task_sr_initialized: torch.Tensor | None = None
        self._bc_target_action_dim: int = 0
        self._demo_observations: TensorDict | None = None
        self._demo_teacher_actions: torch.Tensor | None = None
        self._demo_task_ids: torch.Tensor | None = None

        super().__init__(*args, **kwargs)
        # The adaptive-KL schedule only moves the LR *between* [lr_min, lr_max]; it
        # never re-clamps a starting LR that already sits outside them.  Do it once
        # here, after super() has created self.learning_rate and self.optimizer.
        clamped_lr = min(max(self.learning_rate, self.lr_min), self.lr_max)
        if clamped_lr != self.learning_rate:
            warnings.warn(
                f"learning_rate={self.learning_rate} is outside [lr_min={self.lr_min}, "
                f"lr_max={self.lr_max}]; clamping to {clamped_lr}.",
                stacklevel=2,
            )
            self.learning_rate = clamped_lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
        self._init_side_buffers()
        if use_dapg_demo_loss and (dapg_demo_hdf5_paths or dapg_demo_hdf5_dir):
            self._load_offline_demos()

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _init_side_buffers(self) -> None:
        """Allocate the per-transition side buffers aligned with the rollout storage."""
        num_steps = self.storage.num_transitions_per_env
        num_envs = self.storage.num_envs
        action_dim = int(self.storage.actions_shape[0])
        if self.use_importance_weights:
            self._importance_weights = torch.ones(num_steps, num_envs, 1, device=self.device)
        if self.use_adaptive_bc or self.use_dapg_demo_loss or self.bc_pretrain_only:
            self._bc_teacher_actions = torch.zeros(num_steps, num_envs, action_dim, device=self.device)
            self._adaptive_bc_weights = torch.full(
                (num_steps, num_envs, 1), float(self.bc_loss_coef_max), device=self.device
            )
        if self.use_adaptive_bc or self.use_dapg_demo_loss or self.use_importance_weights:
            self._task_assignment_ids = torch.zeros(num_steps, num_envs, 1, device=self.device, dtype=torch.long)

    @property
    def _uses_custom_update(self) -> bool:
        use_bc = (self.use_adaptive_bc or self.bc_pretrain_only) and self._bc_teacher_actions is not None
        return bool(
            (self.use_importance_weights and self._importance_weights is not None)
            or use_bc
            or self.use_dapg_demo_loss
        )

    # ------------------------------------------------------------------
    # Rollout hooks
    # ------------------------------------------------------------------

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        """Record the DGPO side data for this step, then run the stock bookkeeping."""
        step_idx = self.storage.step
        self._anneal_step_counter += 1
        if self.use_importance_weights and self._importance_weights is not None:
            weights = extras.get(self.importance_weight_key)
            if weights is None:
                weights = torch.ones_like(rewards)
            else:
                weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
            if self._importance_weight_step_counter < self.importance_weight_warmup_steps:
                weights = torch.ones_like(weights)
            self._importance_weights[step_idx].copy_(weights.view(-1, 1))
            self._importance_weight_step_counter += 1

        task_ids = extras.get(self.task_assignment_id_key)
        task_ids_tensor = (
            torch.as_tensor(task_ids, device=self.device, dtype=torch.long).view(-1) if task_ids is not None else None
        )

        if self._bc_teacher_actions is not None and self._adaptive_bc_weights is not None:
            teacher_targets = self._build_teacher_targets(extras.get(self.teacher_action_key))
            self._bc_teacher_actions[step_idx].copy_(teacher_targets)
            if task_ids_tensor is not None:
                bc_weights = self._compute_adaptive_bc_weights(task_ids_tensor)
                self._adaptive_bc_weights[step_idx].copy_(bc_weights.view(-1, 1))
            else:
                self._adaptive_bc_weights[step_idx].fill_(float(self.bc_loss_coef_max))

        if task_ids_tensor is not None:
            self._mirror_task_sr_ema(extras.get(self.task_sr_ema_key), extras.get(self.task_sr_warmed_key))
            if self._task_assignment_ids is not None:
                self._task_assignment_ids[step_idx].copy_(task_ids_tensor.view(-1, 1))

        # Latch human-readable per-tid names from the env (constant tuple).
        if self._task_assignment_names is None:
            names = extras.get("task_assignment_names")
            if isinstance(names, tuple):
                self._task_assignment_names = names

        # SR-derived importance weights: relative-to-mean balancing across tasks
        # (below-mean tasks get more on-policy gradient budget).
        if (
            self.use_importance_weights
            and self.iw_from_task_sr
            and self._importance_weights is not None
            and self._task_sr_ema is not None
            and self._task_sr_initialized is not None
            and task_ids_tensor is not None
            and self._importance_weight_step_counter >= self.importance_weight_warmup_steps
        ):
            self._ensure_task_sr_capacity(int(task_ids_tensor.max().item()) + 1)
            if bool(self._task_sr_initialized.any().item()):
                sr_mean = self._task_sr_ema[self._task_sr_initialized].mean()
            else:
                sr_mean = torch.zeros((), device=self.device)
            task_sr = self._task_sr_ema[task_ids_tensor]
            initialized = self._task_sr_initialized[task_ids_tensor]
            iw = task_sr_weight(
                task_sr,
                sr_mean,
                iw_min=self._annealed(self.iw_min, self.iw_min_final),
                iw_max=self.iw_max,
                shape=self.iw_sr_shape,
                slope=self.iw_sr_balance_slope,
                beta_target=self._current_iw_beta_target(),
                beta_concentration=self.iw_beta_concentration,
                beta_eps=self.iw_beta_eps,
            )
            neutral = 0.5 * (float(self.iw_max) + self._annealed(self.iw_min, self.iw_min_final))
            iw = torch.where(initialized, iw, torch.full_like(iw, neutral))
            self._importance_weights[step_idx].copy_(iw.view(-1, 1))

        super().process_env_step(obs, rewards, dones, extras)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:  # noqa: C901
        """PPO update with importance-weighted losses + BC / DAPG demo losses."""
        if not self._uses_custom_update:
            return super().update()
        if self.actor.is_recurrent or self.critic.is_recurrent:
            warnings.warn("DGPO falls back to stock PPO for recurrent policies.", stacklevel=2)
            return super().update()
        if self.rnd is not None or self.symmetry is not None:
            warnings.warn("DGPO does not combine with RND/symmetry; falling back to stock PPO.", stacklevel=2)
            return super().update()

        use_custom_bc = (
            (self.use_adaptive_bc or self.bc_pretrain_only)
            and self._bc_teacher_actions is not None
            and self._adaptive_bc_weights is not None
        )
        use_custom_dapg = self.use_dapg_demo_loss

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_bc_loss = 0.0 if use_custom_bc else None
        mean_bc_weight = 0.0 if use_custom_bc else None
        bc_weight_count = 0
        mean_advantage_gate = 0.0 if (use_custom_bc and self.use_advantage_gate) else None
        mean_dapg_loss = 0.0 if use_custom_dapg else None
        mean_dapg_weight = 0.0 if use_custom_dapg else None
        # Fixed for the whole update: k indexes policy iterations, not mini-batches.
        dapg_lambda = self._dapg_lambda() if use_custom_dapg else 1.0
        if use_custom_dapg:
            self._dapg_update_count += 1

        st = self.storage
        batch_size = st.num_envs * st.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        indices = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = st.observations.flatten(0, 1)
        actions = st.actions.flatten(0, 1)
        values = st.values.flatten(0, 1)
        returns = st.returns.flatten(0, 1)
        advantages = st.advantages.flatten(0, 1)
        old_actions_log_prob = st.actions_log_prob.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in st.distribution_params)
        flat_importance = (
            self._importance_weights.flatten(0, 1)
            if self.use_importance_weights and self._importance_weights is not None
            else torch.ones(batch_size, 1, device=self.device)
        )
        flat_teacher = self._bc_teacher_actions.flatten(0, 1) if self._bc_teacher_actions is not None else None
        flat_bc_weights = self._adaptive_bc_weights.flatten(0, 1) if self._adaptive_bc_weights is not None else None
        flat_task_ids = self._task_assignment_ids.flatten(0, 1) if self._task_assignment_ids is not None else None
        # Online DAPG reuses the PPO mini-batch (same states, same forward pass).
        online_dapg_ready = (
            use_custom_dapg and not self._dapg_offline_loaded and flat_teacher is not None and flat_task_ids is not None
        )

        for _ in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]

                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                old_params_batch = tuple(p[batch_idx] for p in old_distribution_params)
                weight_batch = flat_importance[batch_idx]

                if self.normalize_advantage_per_mini_batch:
                    with torch.no_grad():
                        advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                            advantages_batch.std() + 1e-8
                        )

                # Forward pass (updates the actor's distribution state).
                self.actor(obs_batch, stochastic_output=True)
                actions_log_prob_batch = self.actor.get_output_log_prob(actions_batch)
                value_batch = self.critic(obs_batch)
                params_batch = self.actor.output_distribution_params
                entropy_batch = self.actor.output_entropy
                action_mean_batch = self.actor.output_mean
                action_std_batch = self.actor.output_std

                # Adaptive learning rate via KL (disabled in pure-BC pretrain mode).
                if (not self.bc_pretrain_only) and self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl_mean = torch.mean(self.actor.get_kl_divergence(old_params_batch, params_batch))
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.gpu_world_size
                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(self.lr_min, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(self.lr_max, self.learning_rate * 1.5)
                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # PPO losses (importance-weighted per sample).
                if self.bc_pretrain_only:
                    weighted_surrogate = torch.tensor(0.0, device=self.device)
                    weighted_value_loss = torch.tensor(0.0, device=self.device)
                    entropy_term = torch.tensor(0.0, device=self.device)
                else:
                    ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                    surrogate = -torch.squeeze(advantages_batch) * ratio
                    surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    surrogate_term = torch.max(surrogate, surrogate_clipped).view(-1)

                    if self.use_clipped_value_loss:
                        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                            -self.clip_param, self.clip_param
                        )
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss_term = torch.max(value_losses, value_losses_clipped).view(-1)
                    else:
                        value_loss_term = (returns_batch - value_batch).pow(2).view(-1)

                    norm_weights = self._normalize_weights(weight_batch.view(-1))
                    # Actor terms (surrogate + entropy) always take the weights; the
                    # value loss opts out under importance_weight_actor_only, which
                    # separates "redistribute the policy-gradient budget" from
                    # "redistribute the critic's regression capacity".
                    weighted_surrogate = (norm_weights * surrogate_term).mean()
                    entropy_term = (norm_weights * entropy_batch.view(-1)).mean()
                    if self.importance_weight_actor_only:
                        weighted_value_loss = value_loss_term.mean()
                    else:
                        weighted_value_loss = (norm_weights * value_loss_term).mean()

                loss = weighted_surrogate + self.value_loss_coef * weighted_value_loss - self.entropy_coef * entropy_term

                # Adaptive BC loss toward the teacher actions.
                bc_loss_term = None
                if use_custom_bc and flat_teacher is not None and self._bc_target_action_dim > 0:
                    teacher_batch = flat_teacher[batch_idx][:, : self._bc_target_action_dim]
                    bc_weight_batch = flat_bc_weights[batch_idx].view(-1)
                    if self.use_advantage_gate:
                        with torch.no_grad():
                            teacher_log_prob = self.actor.get_output_log_prob(flat_teacher[batch_idx]).view(-1)
                            old_lp = old_actions_log_prob_batch.view(-1)
                            is_ratio = torch.exp((teacher_log_prob - old_lp).clamp(-5.0, 5.0))
                            adv = advantages_batch.view(-1)
                            advantage_gate = (adv * is_ratio - adv > 0).float()
                        bc_weight_batch = bc_weight_batch * advantage_gate
                        if mean_advantage_gate is not None:
                            mean_advantage_gate += advantage_gate.mean().item()
                    bc_per_sample = torch.mean(
                        torch.square(action_mean_batch[:, : self._bc_target_action_dim] - teacher_batch), dim=-1
                    )
                    bc_loss_term = (bc_weight_batch * bc_per_sample).mean()
                    loss = loss + self.bc_loss_coef_base * bc_loss_term
                    mean_bc_weight += bc_weight_batch.mean().item()
                    bc_weight_count += 1

                # DAPG demo log-likelihood loss.
                dapg_loss_term = None
                dapg_weights = None
                if use_custom_dapg and self._bc_target_action_dim > 0:
                    if self._dapg_offline_loaded and self._demo_teacher_actions is not None:
                        demo_indices = torch.randint(
                            0, self._demo_teacher_actions.shape[0], (obs_batch.batch_size[0],), device=self.device
                        )
                        demo_actions_batch = self._demo_teacher_actions[demo_indices, : self._bc_target_action_dim]
                        demo_task_ids_batch = self._demo_task_ids[demo_indices].view(-1)
                        dapg_weights = self._dapg_sample_weights(demo_task_ids_batch)
                        # Separate forward pass: demo obs come from a different distribution.
                        self.actor(self._demo_observations[demo_indices], stochastic_output=True)
                        demo_log_prob = self._dapg_log_prob(demo_actions_batch)
                        dapg_loss_term = -(dapg_weights * demo_log_prob.view(-1)).mean()
                    elif online_dapg_ready:
                        demo_actions_batch = flat_teacher[batch_idx][:, : self._bc_target_action_dim]
                        demo_task_ids_batch = flat_task_ids[batch_idx].view(-1)
                        dapg_weights = self._dapg_sample_weights(demo_task_ids_batch)
                        # Reuses the PPO forward pass on the same states (distribution
                        # still reflects obs_batch here — offline path above would
                        # have overwritten it, hence the ordering).
                        dapg_loss_term = -(
                            dapg_weights
                            * self._dapg_log_prob_from(action_mean_batch, action_std_batch, demo_actions_batch).view(-1)
                        ).mean()
                    if dapg_loss_term is not None:
                        loss = loss + self.dapg_loss_coef_base * dapg_lambda * dapg_loss_term
                        mean_dapg_loss += dapg_loss_term.item()
                        mean_dapg_weight += dapg_weights.mean().item()

                # Gradient step.
                self.optimizer.zero_grad()
                loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.std_floor > 0.0 or self.std_max > 0.0:
                    self._apply_std_bounds()

                mean_value_loss += float(weighted_value_loss.item())
                mean_surrogate_loss += float(weighted_surrogate.item())
                mean_entropy += float(entropy_term.item())
                if mean_bc_loss is not None and bc_loss_term is not None:
                    mean_bc_loss += bc_loss_term.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        self._refresh_demo_buffer_from_rollout()
        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if mean_bc_loss is not None:
            loss_dict["bc"] = mean_bc_loss / num_updates
        if mean_bc_weight is not None and bc_weight_count > 0 and self._adaptive_bc_weights is not None:
            loss_dict["metrics/bc_weight_mean"] = mean_bc_weight / bc_weight_count
            loss_dict["metrics/bc_weight_min"] = self._adaptive_bc_weights.min().item()
            loss_dict["metrics/bc_weight_max"] = self._adaptive_bc_weights.max().item()
        if mean_advantage_gate is not None:
            loss_dict["metrics/advantage_gate_mean"] = mean_advantage_gate / num_updates
        if mean_dapg_loss is not None:
            loss_dict["dapg"] = mean_dapg_loss / num_updates
            loss_dict["metrics/dapg_weight_mean"] = (mean_dapg_weight or 0.0) / num_updates
            loss_dict["metrics/dapg_lambda_now"] = dapg_lambda
        if self._task_sr_ema is not None and self._task_sr_initialized is not None:
            valid_sr = self._task_sr_ema[self._task_sr_initialized]
            if valid_sr.numel() > 0:
                loss_dict["metrics/task_sr_ema_mean"] = valid_sr.mean().item()
                loss_dict["metrics/task_sr_ema_min"] = valid_sr.min().item()
                loss_dict["metrics/task_sr_ema_max"] = valid_sr.max().item()
        if self.anneal_steps > 0:
            # what the schedule is actually applying right now, so a run can be read
            # back without recomputing the lerp from the step count
            loss_dict["metrics/anneal_fraction"] = self.anneal_fraction()
            if self.bc_loss_coef_max_final is not None:
                loss_dict["metrics/bc_coef_max_now"] = self._annealed(
                    self.bc_loss_coef_max, self.bc_loss_coef_max_final
                )
            if self.iw_min_final is not None:
                loss_dict["metrics/iw_min_now"] = self._annealed(self.iw_min, self.iw_min_final)
            if self.iw_beta_target_final is not None:
                loss_dict["metrics/iw_beta_target_now"] = self._current_iw_beta_target()
        return loss_dict

    # ------------------------------------------------------------------
    # Save / load (persist the adaptive-BC state)
    # ------------------------------------------------------------------

    def save(self) -> dict:
        """Return the stock save dict plus the DGPO algorithm state (SR EMAs)."""
        saved = super().save()
        state = {}
        if self._task_sr_ema is not None:
            state["_task_sr_ema"] = self._task_sr_ema.clone()
        if self._task_sr_initialized is not None:
            state["_task_sr_initialized"] = self._task_sr_initialized.clone()
        state["_anneal_step_counter"] = int(self._anneal_step_counter)
        state["_dapg_update_count"] = int(self._dapg_update_count)
        if state:
            saved["dgpo_state"] = state
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load stock models, then restore the DGPO state when present."""
        result = super().load(loaded_dict, load_cfg, strict)
        for field, val in loaded_dict.get("dgpo_state", {}).items():
            if field in ("_task_sr_ema", "_task_sr_initialized") and isinstance(val, torch.Tensor):
                setattr(self, field, val.to(self.device))
            elif field == "_anneal_step_counter":
                self._anneal_step_counter = int(val)
            elif field == "_dapg_update_count":
                self._dapg_update_count = int(val)
        return result

    # ------------------------------------------------------------------
    # Helpers (ported from the playground ImportanceAwarePPO)
    # ------------------------------------------------------------------

    def anneal_fraction(self) -> float:
        """Progress through the annealing window in ``[0, 1]``; ``0`` when disabled.

        Disabled has to read as "no time has elapsed", not "fully elapsed":
        :func:`~isaaclab_rl.rsl_rl.task_sr_weights.effective_sr` consumes this
        directly, and a fraction of 1 there collapses every success rate to 1 and
        zeroes the BC weight outright.  (:meth:`_annealed` short-circuits on
        ``anneal_steps`` before reaching this, so it is unaffected either way.)
        """
        if self.anneal_steps <= 0:
            return 0.0
        return min(self._anneal_step_counter / float(self.anneal_steps), 1.0)

    def _annealed(self, start: float, final: float | None) -> float:
        """``start`` lerped toward ``final`` by :meth:`anneal_fraction`."""
        if final is None or self.anneal_steps <= 0:
            return float(start)
        return float(start) + (float(final) - float(start)) * self.anneal_fraction()

    def _current_iw_beta_target(self) -> float:
        """``iw_beta_target`` after annealing; unchanged under the follow-the-mean sentinel."""
        if self.iw_beta_target is None or float(self.iw_beta_target) < 0.0:
            return self.iw_beta_target
        return self._annealed(self.iw_beta_target, self.iw_beta_target_final)

    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Normalize importance weights (``mean`` mode divides by the batch mean)."""
        if self.importance_weight_normalization == "mean":
            return weights / weights.mean().clamp_min(1e-6)
        return weights

    def _build_teacher_targets(self, teacher_actions) -> torch.Tensor:
        """Fit the extras teacher actions into the (num_envs, action_dim) buffer row."""
        target = torch.zeros(
            self._bc_teacher_actions.shape[1], self._bc_teacher_actions.shape[2], device=self.device
        )
        if teacher_actions is None:
            return target
        teacher = torch.as_tensor(teacher_actions, device=self.device, dtype=torch.float32)
        if teacher.ndim == 1:
            teacher = teacher.unsqueeze(-1)
        dim = min(target.shape[1], teacher.shape[1])
        if dim <= 0:
            return target
        target[:, :dim] = teacher[:, :dim]
        self._bc_target_action_dim = max(self._bc_target_action_dim, dim)
        gripper_dim = min(int(self.bc_gripper_action_dim), dim)
        if gripper_dim > 0 and self.bc_invert_gripper_target_sign:
            target[:, dim - gripper_dim : dim] *= -1.0
        return target

    def _ensure_task_sr_capacity(self, num_tasks: int) -> None:
        """Grow the per-task SR EMA buffers to hold at least ``num_tasks`` entries."""
        if self._task_sr_ema is None or self._task_sr_initialized is None:
            self._task_sr_ema = torch.zeros(num_tasks, device=self.device, dtype=torch.float32)
            self._task_sr_initialized = torch.zeros(num_tasks, device=self.device, dtype=torch.bool)
            return
        if self._task_sr_ema.numel() >= num_tasks:
            return
        old = self._task_sr_ema.numel()
        expanded_sr = torch.zeros(num_tasks, device=self.device, dtype=torch.float32)
        expanded_sr[:old] = self._task_sr_ema
        expanded_init = torch.zeros(num_tasks, device=self.device, dtype=torch.bool)
        expanded_init[:old] = self._task_sr_initialized
        self._task_sr_ema = expanded_sr
        self._task_sr_initialized = expanded_init

    def _compute_adaptive_bc_weights(self, task_ids: torch.Tensor) -> torch.Tensor:
        """Per-sample BC/DAPG weight: lerp max→min as the task's SR EMA rises."""
        if task_ids.numel() == 0:
            return torch.empty(0, device=self.device, dtype=torch.float32)
        self._ensure_task_sr_capacity(int(task_ids.max().item()) + 1)
        task_sr = self._task_sr_ema[task_ids]
        denom = max(float(self.bc_sr_high_threshold - self.bc_sr_low_threshold), 1.0e-6)
        progress = torch.clamp((task_sr - float(self.bc_sr_low_threshold)) / denom, 0.0, 1.0)
        coef_max = self._annealed(self.bc_loss_coef_max, self.bc_loss_coef_max_final)
        if self.bc_sr_shape == "beta":
            # The importance weights' kernel at target 0, fed the time-folded rate:
            # one expression for "the task learned" and "the run moved on".
            zero = torch.zeros((), device=task_sr.device, dtype=task_sr.dtype)
            return task_sr_weight(
                effective_sr(task_sr, self.anneal_fraction()),
                zero,
                iw_min=float(self.bc_loss_coef_min),
                iw_max=coef_max,
                shape="beta",
                beta_target=0.0,
                beta_concentration=float(self.bc_beta_concentration),
                beta_eps=self.iw_beta_eps,
            )
        return coef_max * (1.0 - progress) + float(self.bc_loss_coef_min) * progress

    def _mirror_task_sr_ema(self, task_sr_ema, task_sr_warmed) -> None:
        """Copy the env's ``TaskSRTracker`` EMA (the one true copy) into our buffers.

        This algorithm does not compute its own SR EMA -- see the module
        docstring. ``task_sr_ema`` / ``task_sr_warmed`` come from
        ``extras[task_sr_ema_key]`` / ``extras[task_sr_warmed_key]``
        (``TaskSRTracker.step_extras``); absent on envs that don't publish them
        (e.g. old checkpoints' envs, or non-LIBERO tasks), in which case this
        is a no-op and IW/BC weighting falls back to their neutral defaults.
        """
        if task_sr_ema is None:
            return
        ema = torch.as_tensor(task_sr_ema, device=self.device, dtype=torch.float32).view(-1)
        if ema.numel() == 0:
            return
        self._ensure_task_sr_capacity(ema.numel())
        warmed = (
            torch.as_tensor(task_sr_warmed, device=self.device, dtype=torch.bool).view(-1)
            if task_sr_warmed is not None
            else torch.ones_like(ema, dtype=torch.bool)
        )
        n = ema.numel()
        self._task_sr_ema[:n] = ema
        self._task_sr_initialized[:n] = warmed

    def _dapg_sample_weights(self, task_ids: torch.Tensor) -> torch.Tensor:
        """Per-sample weight on the DAPG demo term.

        ``dapg_sr_adaptive_weight=False`` returns ones, which is the original paper:
        there the demo term is scaled by ``lambda_0 * lambda_1^k`` and nothing else.
        The per-task SR lerp is DGPO's mechanism, so an MT-DAPG baseline that inherits
        it is not measuring DAPG.
        """
        if not self.dapg_sr_adaptive_weight:
            return torch.ones(task_ids.numel(), device=self.device, dtype=torch.float32)
        return self._compute_adaptive_bc_weights(task_ids)

    def _dapg_lambda(self) -> float:
        """DAPG's ``lambda_0 * lambda_1^k`` demo-term decay (Rajeswaran et al. 2018, eq. 4).

        ``k`` is the policy-iteration index, so this asks how far the RUN has got --
        orthogonal to the per-task SR lerp, which asks how far a given TASK has got.
        Returns ``dapg_lambda0`` unchanged when ``dapg_lambda1 == 1.0``.

        The paper additionally scales the demo term by ``max A`` over the on-policy
        batch, putting it in the units of the surrogate. That is not implemented here:
        the demo term is an SR-weighted log-likelihood on a fixed scale, and a
        batch-max advantage would make one task's demo weight depend on which other
        tasks happened to share the rollout -- the coupling per-task weighting exists
        to remove.
        """
        if self.dapg_lambda1 == 1.0:
            return float(self.dapg_lambda0)
        return float(self.dapg_lambda0) * float(self.dapg_lambda1) ** self._dapg_update_count

    def _dapg_log_prob(self, demo_actions: torch.Tensor) -> torch.Tensor:
        """log π(a_demo | s) from the actor's CURRENT distribution state (std optionally detached)."""
        return self._dapg_log_prob_from(self.actor.output_mean, self.actor.output_std, demo_actions)

    def _dapg_log_prob_from(
        self, mean: torch.Tensor, std: torch.Tensor, demo_actions: torch.Tensor
    ) -> torch.Tensor:
        """log-prob of demo actions under N(mean, std); detaching std keeps DAPG from collapsing σ."""
        if self.dapg_detach_std:
            std = std.detach()
        dist = torch.distributions.Normal(mean[:, : demo_actions.shape[-1]], std[:, : demo_actions.shape[-1]])
        return dist.log_prob(demo_actions).sum(dim=-1)

    def _apply_std_bounds(self) -> None:
        """Clamp the Gaussian σ into ``[std_floor, std_max]``; 0 disables either side.

        ``std_max`` is the counterpart to ``std_floor``.  The entropy bonus contributes
        a *constant* positive gradient to log_std (dH/dlog σ = 1 per dim), and Adam
        normalizes gradient magnitude, so a consistently-signed gradient moves the
        parameter by ~lr per step.  Whenever the surrogate term stops opposing it —
        e.g. adaptive BC takes over, and its loss is an MSE on the actor *mean* that
        carries no log_std gradient at all — σ grows without bound.  A ceiling turns
        that divergence into a bounded degradation.
        """
        lo = self.std_floor if self.std_floor > 0.0 else None
        hi = self.std_max if self.std_max > 0.0 else None
        dist = getattr(self.actor, "distribution", None)
        if dist is None:
            return
        if hasattr(dist, "log_std_param"):
            dist.log_std_param.data.clamp_(
                min=math.log(lo) if lo is not None else None,
                max=math.log(hi) if hi is not None else None,
            )
        elif hasattr(dist, "std_param"):
            dist.std_param.data.clamp_(min=lo, max=hi)

    # Backwards-compatible alias: older callers referenced _apply_std_floor.
    _apply_std_floor = _apply_std_bounds

    def _load_offline_demos(self) -> None:
        """Load DAPG demo transitions (``obs/{group}`` + ``actions``) from HDF5 once at init.

        Raises rather than degrades. Every failure here -- a path that is not a file,
        a file with no ``data`` group, a demo missing a configured obs group or its
        ``actions`` -- would otherwise leave ``_dapg_offline_loaded`` False, and the
        update silently falls back to the ONLINE rollout teacher. That fallback is a
        different algorithm running under the same experiment name, which is the worst
        way to lose a baseline: nothing crashes and the numbers look plausible.
        """
        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is required for offline DAPG demo loading.") from e

        hdf5_paths = self.dapg_demo_hdf5_paths
        if not hdf5_paths and self.dapg_demo_hdf5_dir:
            dir_path = os.path.abspath(self.dapg_demo_hdf5_dir)
            hdf5_paths = sorted(glob.glob(os.path.join(dir_path, "*.hdf5")))
            if not hdf5_paths:
                raise FileNotFoundError(f"[DAPG] dapg_demo_hdf5_dir={dir_path!r} contains no *.hdf5 files.")
            print(f"[DAPG] Auto-discovered {len(hdf5_paths)} HDF5 files from {dir_path}")
        if not hdf5_paths:
            raise ValueError(
                "[DAPG] use_dapg_demo_loss is on and offline demos were requested, but neither "
                "dapg_demo_hdf5_paths nor dapg_demo_hdf5_dir resolved to any file. Set "
                "LIBERO_DAPG_DEMO_DIR to the preprocessed demo directory, or leave both unset "
                "to opt into the online rollout teacher deliberately."
            )

        obs_groups = self.dapg_obs_groups
        task_ids_map = self.dapg_task_ids or list(range(len(hdf5_paths)))
        all_obs_by_group: dict[str, list[np.ndarray]] = {g: [] for g in obs_groups}
        all_acts: list[np.ndarray] = []
        all_tids: list[int] = []

        if len(task_ids_map) < len(hdf5_paths):
            raise ValueError(
                f"[DAPG] dapg_task_ids has {len(task_ids_map)} entries for {len(hdf5_paths)} HDF5 "
                "files. It must label every file, in the same sorted order."
            )

        for path_idx, hdf5_path in enumerate(hdf5_paths):
            task_id = task_ids_map[path_idx]
            if not hdf5_path or not os.path.isfile(hdf5_path):
                raise FileNotFoundError(
                    f"[DAPG] HDF5 path [{path_idx}] = {hdf5_path!r} is not a file. Each entry must "
                    "be a full path to an HDF5 file, not a directory."
                )
            with h5py.File(hdf5_path, "r") as f:
                if "data" not in f:
                    raise KeyError(
                        f"[DAPG] {hdf5_path!r} has no 'data' group (top-level keys: "
                        f"{sorted(f.keys())}). This loader expects the robomimic layout, "
                        "data/<demo>/obs/<group> + data/<demo>/actions."
                    )
                demo_keys = list(f["data"].keys())
                if not demo_keys:
                    raise ValueError(f"[DAPG] {hdf5_path!r} has an empty 'data' group.")
                for demo_key in demo_keys:
                    demo_grp = f[f"data/{demo_key}"]
                    group_chunks: dict[str, np.ndarray] = {}
                    for group_name in obs_groups:
                        key = f"obs/{group_name}"
                        if key not in demo_grp:
                            raise KeyError(
                                f"[DAPG] {hdf5_path!r} data/{demo_key} has no {key!r}. dapg_obs_groups="
                                f"{obs_groups} needs the PREPROCESSED dataset; the assembled demos do "
                                "not carry these obs keys."
                            )
                        group_chunks[group_name] = demo_grp[key][:]
                    if "actions" not in demo_grp:
                        raise KeyError(f"[DAPG] {hdf5_path!r} data/{demo_key} has no 'actions'.")
                    acts: np.ndarray = demo_grp["actions"][:]
                    n_steps = acts.shape[0]
                    for g in obs_groups:
                        if group_chunks[g].shape[0] != n_steps:
                            raise ValueError(
                                f"[DAPG] {hdf5_path!r} data/{demo_key}: obs/{g} has "
                                f"{group_chunks[g].shape[0]} rows but actions has {n_steps}. "
                                "Concatenating them would silently misalign state and action."
                            )
                        all_obs_by_group[g].append(group_chunks[g])
                    all_acts.append(acts)
                    all_tids.extend([task_id] * n_steps)

        if not all_acts:
            raise ValueError(
                f"[DAPG] {len(hdf5_paths)} HDF5 file(s) yielded no transitions; every 'data' group "
                "was empty."
            )

        limit = self.dapg_max_demo_samples
        acts_arr = np.concatenate(all_acts, axis=0)[:limit]
        tids_arr = np.array(all_tids[:limit], dtype=np.int64)
        obs_by_group = {
            g: torch.tensor(np.concatenate(all_obs_by_group[g], axis=0)[:limit], device=self.device, dtype=torch.float32)
            for g in obs_groups
        }
        n_trans = acts_arr.shape[0]
        self._demo_observations = TensorDict(obs_by_group, batch_size=[n_trans], device=self.device)
        self._demo_teacher_actions = torch.tensor(acts_arr, device=self.device, dtype=torch.float32)
        self._demo_task_ids = torch.tensor(tids_arr, device=self.device, dtype=torch.long).unsqueeze(-1)
        self._dapg_offline_loaded = True
        print(f"[DAPG] Loaded {n_trans} demo transitions from {len(hdf5_paths)} HDF5 files (groups={obs_groups}).")

    def _refresh_demo_buffer_from_rollout(self) -> None:
        """Snapshot the latest rollout as the online DAPG demo buffer (no offline demos only)."""
        if not self.use_dapg_demo_loss or self._dapg_offline_loaded:
            return
        if self._bc_teacher_actions is None or self._task_assignment_ids is None:
            return
        self._demo_observations = self.storage.observations.flatten(0, 1).clone()
        self._demo_teacher_actions = self._bc_teacher_actions.flatten(0, 1).clone()
        self._demo_task_ids = self._task_assignment_ids.flatten(0, 1).clone()
