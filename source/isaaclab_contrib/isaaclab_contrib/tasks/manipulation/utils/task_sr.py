# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-task success-rate tracking on native manager extension points.

Replaces the former ``PerTaskSREnvMixin`` / ``GripperCurriculumEnvMixin`` env
subclassing: the multi-task state lives in one :class:`TaskSRTracker` owned by
a standard :class:`~isaaclab.managers.CurriculumManager` term
(:class:`task_success_rate`), and the pieces that need it reach it through the
documented ``env.task_sr_tracker`` handle:

* the **curriculum term** updates the per-task SR EMA at episode end
  (``CurriculumManager.compute`` runs inside ``_reset_idx`` BEFORE the scene
  reset, so the pre-reset success signal is still readable);
* the **gripper curriculum action term** reads :meth:`TaskSRTracker.replay_mask`
  each step (see :mod:`.gripper_curriculum`);
* the **RSL-RL vec-env wrapper** merges :meth:`TaskSRTracker.step_log` into
  ``extras["log"]`` and injects :meth:`TaskSRTracker.step_extras` (the DGPO
  algorithm contract: ``task_assignment_ids`` / ``task_success`` /
  ``teacher_actions``) — duck-typed, so ``isaaclab_rl`` never imports this
  package.

Episode outcome semantics match the playground repo: an episode counts as
successful when the task's success signal holds at the FINAL pre-reset step.
MDP success predicates publish that signal via :func:`stash_task_success`
(evaluated by the reward manager every step in every reward mode, i.e. before
the reset runs).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase
from isaaclab.utils.configclass import configclass
from .. import settings

TASK_SR_EMA_ALPHA = settings.number("MTB_TASK_SR_EMA_ALPHA", 0.05)

#: Episodes a task must finish before its SR EMA is trusted by consumers.
#:
#: The EMA seeds on the first batch that finishes (``ema = batch mean``), which
#: on a full wave of a task's envs is already a reasonable estimate — but a
#: partial wave, or a run with few envs per task, can still seed it at 0.0 or
#: 1.0, and with alpha=0.05 it then takes ~14 waves to move even halfway back.
#: Gating the *consumers* on an episode count keeps the estimator honest without
#: distorting the EMA itself.
TASK_SR_MIN_EPISODES = settings.integer("MTB_TASK_SR_MIN_EPISODES", 20)

_STASH_ATTR = "_task_success_stash"


def stash_task_success(env, mask: torch.Tensor) -> None:
    """Publish this step's PRE-reset per-env success mask on the env.

    Called by ``libero_task_success`` during reward/termination evaluation. The stamp
    lets readers reject stale masks after mode changes.
    """
    setattr(env, _STASH_ATTR, (int(env.common_step_counter), mask))


def read_task_success(env, success_term_names: Sequence[str] = ()) -> torch.Tensor:
    """Return this step's PRE-reset per-env success mask (bool, ``(num_envs,)``).

    Prefers an active success *termination* term (eval schemes — its buffer is
    computed pre-reset), then the :func:`stash_task_success` mask, else zeros.
    """
    termination_manager = getattr(env, "termination_manager", None)
    if termination_manager is not None:
        for name in success_term_names:
            if name in termination_manager.active_terms:
                return termination_manager.get_term(name).to(dtype=torch.bool)
    stash = getattr(env, _STASH_ATTR, None)
    if stash is not None and env.common_step_counter - stash[0] <= 1:
        return stash[1].to(dtype=torch.bool)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


class TaskSRTracker:
    """Per-task SR EMA state for one multi-task env (env ``i`` runs task ``i % n``)."""

    def __init__(
        self,
        labels: Sequence[str],
        num_envs: int,
        device,
        ema_alpha: float = TASK_SR_EMA_ALPHA,
        min_episodes: int = TASK_SR_MIN_EPISODES,
    ) -> None:
        self.labels = tuple(labels)
        self.ema_alpha = float(ema_alpha)
        self.min_episodes = int(min_episodes)
        n_tasks = len(self.labels)
        self.env_task = torch.arange(num_envs, device=device, dtype=torch.long) % max(n_tasks, 1)
        self.ema = torch.zeros(n_tasks, device=device)
        self.seen = torch.zeros(n_tasks, dtype=torch.bool, device=device)
        # finished episodes per task; the warm-up gate for every consumer
        self.episodes = torch.zeros(n_tasks, dtype=torch.long, device=device)
        # per-suite grouping from ``suite::id`` labels (no groups otherwise)
        suites: dict[str, list[int]] = {}
        for idx, label in enumerate(self.labels):
            if "::" in label:
                suites.setdefault(label.split("::", 1)[0], []).append(idx)
        self.suites = {name: torch.tensor(ids, device=device) for name, ids in suites.items()}
        # refreshed by consumers for logging (gripper action term / wrapper)
        self.last_replay_mask: torch.Tensor | None = None
        # set by the owning curriculum term (suite-specific teacher command view)
        self.teacher_provider = None

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def restore(self, ema: torch.Tensor, initialized: torch.Tensor) -> None:
        """Seed the tracker from a checkpointed algorithm-side EMA (see ``train.py`` resume).

        ``env.task_sr_tracker`` is rebuilt from scratch by every ``gym.make()``,
        so without this call a resumed run's curriculum/logged SR would cold-start
        even though the actor/critic weights (and the algorithm's own EMA mirror,
        which is checkpointed) are fully warm. Tasks the checkpoint had already
        seeded are marked ``warmed`` immediately (``episodes = min_episodes``)
        rather than replayed from zero -- we don't know the true historical
        episode count, and trusting the restored estimate right away is the
        point of resuming.
        """
        n = min(ema.numel(), self.ema.numel())
        if n == 0:
            return
        init = initialized[:n].to(dtype=torch.bool, device=self.ema.device)
        self.ema[:n] = torch.where(init, ema[:n].to(dtype=self.ema.dtype, device=self.ema.device), self.ema[:n])
        self.seen[:n] |= init
        self.episodes[:n] = torch.where(
            init, torch.full_like(self.episodes[:n], self.min_episodes), self.episodes[:n]
        )

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_on_episode_end(self, done_ids: torch.Tensor, success_mask: torch.Tensor) -> None:
        """Fold finished episodes into their task EMAs (first batch seeds directly).

        One EMA step per task per call, on the MEAN outcome of that task's envs
        that finished together -- not one step per env.  The distinction is not
        cosmetic here: `command_depleted` truncates ~98% of episodes, and all of a
        task's envs are bound to demos of the same length, so a task's envs finish
        in one wave.  Folding them individually would apply ~envs-per-task EMA
        steps at once (25 envs at alpha=0.05 is an effective alpha of 0.72), which
        makes the estimate track the last wave rather than smooth over waves and
        leaves the gripper curriculum -- which thresholds this EMA every step --
        oscillating between replay and policy control.

        ``episodes`` still counts individual episodes: it gates ``warmed``, and
        what that gate wants to know is how much data the estimate rests on.
        """
        if done_ids.numel() == 0:
            return
        alpha = self.ema_alpha
        tasks = self.env_task[done_ids]
        outcomes = success_mask[done_ids].to(dtype=self.ema.dtype)
        for t in torch.unique(tasks).tolist():
            in_task = tasks == t
            batch_sr = outcomes[in_task].mean()
            if self.seen[t]:
                self.ema[t] = (1.0 - alpha) * self.ema[t] + alpha * batch_sr
            else:
                self.ema[t] = batch_sr
                self.seen[t] = True
            self.episodes[t] += int(in_task.sum().item())

    # ------------------------------------------------------------------
    # Consumers
    # ------------------------------------------------------------------

    @property
    def warmed(self) -> torch.Tensor:
        """Per-task bool: enough finished episodes for the EMA to mean anything."""
        return self.episodes >= self.min_episodes

    def replay_mask(self, sr_threshold: float | None) -> torch.Tensor:
        """Per-env bool mask: task not warmed up yet, or SR EMA below ``sr_threshold``.

        The warm-up term is what stops a single fluke episode from graduating a
        task out of the curriculum: seeding pins the EMA at 1.0 after one
        success, which clears any sane threshold immediately.
        """
        if sr_threshold is None:
            mask = torch.zeros_like(self.env_task, dtype=torch.bool)
        else:
            task_sr = self.ema[self.env_task]
            mask = (~self.warmed[self.env_task]) | (task_sr < float(sr_threshold))
        self.last_replay_mask = mask
        return mask

    def step_log(self) -> dict[str, torch.Tensor]:
        """``Metrics/task_sr/*`` entries (mean, per-suite means, one per task)."""
        log: dict[str, torch.Tensor] = {"Metrics/task_sr/mean": self.ema.mean()}
        for name, ids in self.suites.items():
            log[f"Metrics/task_sr/suite/{name}"] = self.ema[ids].mean()
        for idx, label in enumerate(self.labels):
            log[f"Metrics/task_sr/{label}"] = self.ema[idx]
        # How much of the mean above is warmed-up estimate vs. seeded noise.
        # Without this the early curve is unreadable: `mean` looks like a
        # success rate long before any task has the episodes to support one.
        warmed = self.warmed
        log["Metrics/task_sr/warmed_frac"] = warmed.float().mean()
        log["Metrics/task_sr/mean_warmed"] = self.ema[warmed].mean() if bool(warmed.any()) else self.ema.new_zeros(())
        if self.last_replay_mask is not None:
            log["Metrics/gripper_curriculum/replay_frac"] = self.last_replay_mask.float().mean()
        return log

    def step_extras(self, env) -> dict:
        """Per-step DGPO algorithm extras (task ids/names, success mask, teacher actions).

        Also publishes this tracker's own per-task EMA (``task_sr_ema`` /
        ``task_sr_warmed``) so the algorithm side can mirror it instead of
        keeping an independent copy — this tracker is the only place the SR
        EMA is actually computed; see :class:`~isaaclab_rl.rsl_rl.dgpo_ppo.DgpoPPO`.
        """
        extras = {
            "task_assignment_ids": self.env_task,
            "task_assignment_names": self.labels,
            "task_success": read_task_success(env, ("libero_success", "success")).float(),
            "task_sr_ema": self.ema,
            "task_sr_warmed": self.warmed,
        }
        if self.teacher_provider is not None:
            teacher = self.teacher_provider(env)
            if teacher is not None:
                extras["teacher_actions"] = teacher
        return extras


class task_success_rate(ManagerTermBase):
    """Curriculum term owning the :class:`TaskSRTracker` (native episode-end hook).

    ``CurriculumManager.compute(env_ids)`` runs at the top of ``_reset_idx`` —
    exactly once per finished episode, before the scene reset — so the term
    folds the pre-reset success signal into the per-task EMA there. Returns the
    mean SR (logged natively as ``Curriculum/<term_name>``).

    Params (via :class:`~isaaclab.managers.CurriculumTermCfg`):

    * ``labels`` — one label per task, env ``i`` runs task ``i % len(labels)``.
    * ``success_term_names`` — success *termination* terms to prefer (eval).
    * ``teacher_command_name`` — command view exported as ``teacher_actions``
      (``None`` disables; the DGPO/ABC online teacher).
    * ``ema_alpha`` — EMA smoothing factor.
    """

    def __init__(self, cfg: CurriculumTermCfg, env) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        labels = tuple(params.get("labels", ()))
        if not labels:
            raise ValueError("task_success_rate curriculum term requires non-empty 'labels'.")
        self._success_term_names = tuple(params.get("success_term_names", ("libero_success", "success")))
        self._tracker = TaskSRTracker(
            labels,
            env.num_envs,
            env.device,
            ema_alpha=float(params.get("ema_alpha", TASK_SR_EMA_ALPHA)),
            min_episodes=int(params.get("min_episodes", TASK_SR_MIN_EPISODES)),
        )
        teacher_command = params.get("teacher_command_name")
        if teacher_command:
            from .commands import resolve_command

            def _teacher(env, _name=teacher_command):
                try:
                    return resolve_command(env, _name)
                except KeyError:
                    return None

            self._tracker.teacher_provider = _teacher
        # the documented handle for the action term / RL wrapper
        env.task_sr_tracker = self._tracker

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        labels: Sequence[str] = (),
        success_term_names: Sequence[str] = (),
        teacher_command_name: str | None = None,
        ema_alpha: float = TASK_SR_EMA_ALPHA,
        min_episodes: int = TASK_SR_MIN_EPISODES,
    ) -> float:
        """Fold the finishing envs' pre-reset success into the EMA; return mean SR."""
        del labels, success_term_names, teacher_command_name, ema_alpha, min_episodes  # consumed by __init__
        if isinstance(env_ids, slice):
            done_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        else:
            done_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        # skip the startup reset (episode_length_buf is zeroed AFTER this hook,
        # so genuinely finished episodes still show their pre-reset length here)
        done_ids = done_ids[env.episode_length_buf[done_ids] > 0]
        if done_ids.numel() > 0:
            success = read_task_success(env, self._success_term_names)
            self._tracker.update_on_episode_end(done_ids, success)
        return float(self._tracker.ema.mean().item())


@configclass
class TaskSRCurriculumCfg:
    """Curriculum manager cfg with the single :class:`task_success_rate` term.

    Build via :func:`make_task_sr_curriculum_cfg` so ``labels`` comes from the
    Selector/registry task bindings.
    """

    task_sr: CurriculumTermCfg | None = None


def make_task_sr_curriculum_cfg(
    labels: Sequence[str],
    *,
    success_term_names: Sequence[str] = ("libero_success", "success"),
    teacher_command_name: str | None = "source_action",
    ema_alpha: float = TASK_SR_EMA_ALPHA,
    min_episodes: int = TASK_SR_MIN_EPISODES,
) -> TaskSRCurriculumCfg:
    """Curriculum cfg tracking per-task SR for the given task labels."""
    cfg = TaskSRCurriculumCfg()
    cfg.task_sr = CurriculumTermCfg(
        func=task_success_rate,
        params={
            "labels": tuple(labels),
            "success_term_names": tuple(success_term_names),
            "teacher_command_name": teacher_command_name,
            "ema_alpha": ema_alpha,
            "min_episodes": min_episodes,
        },
    )
    return cfg
