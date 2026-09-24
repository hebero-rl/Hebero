# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared mapping from a per-task success-rate EMA to a loss weight.

PPO importance weighting and adaptive BC use this common mapping to keep their
success-rate weighting conventions consistent.

Shapes
------
``sigmoid`` (default, and what every run so far used) is monotone in the success
rate, measured relative to the multi-task mean: tasks below the mean move toward
``iw_max``, tasks above it toward ``iw_min``.  Its failure mode is that it has no
floor by difficulty -- a task stuck near 0, whether unlearnable, mis-specified or
simply not yet reached, holds ``iw_max`` indefinitely and keeps taking gradient
budget from the tasks that can still move.

``beta`` replaces the sigmoid with a unimodal kernel peaking at a target success
rate, so the solved end and the hopeless end are both down-weighted:

    w_hat = (p+eps)^(k*t) (1-p+eps)^(k*(1-t)) / [ (t+eps)^(k*t) (1-t+eps)^(k*(1-t)) ]
    w     = iw_min + (iw_max - iw_min) * w_hat

This is the learnability shape of SFL's ``p(1-p)`` and of the Beta kernel in
Success Guided Sampling.  Both of those apply it to the *sampling* distribution
and then pass it through a softmax whose temperature tames the dynamic range.
Here it stays a loss weight, where there is no such softmax and the raw kernel
spans about 5000x at ``k=1`` (sqrt(eps) up to 0.5) -- enough to wreck the
surrogate.  Hence the affine map into the existing ``[iw_min, iw_max]``
envelope: the shape is kept, the safety range is the one the sigmoid path
already had.

``target = 0`` reduces the same kernel to ``(1 - p)^kappa``, monotonically
decreasing -- the shape an imitation weight wants.  So the demonstration weight
and the gradient-budget weight are one function read at two modes.

Normalisation is by the kernel's own mode rather than by the batch.  A
batch-relative scale would make one task's weight depend on which other tasks
happen to share the rollout, which is the coupling per-task weighting exists to
remove.
"""

from __future__ import annotations

import torch

VALID_IW_SR_SHAPES = ("sigmoid", "beta")


def effective_sr(task_sr: torch.Tensor, anneal_fraction: float) -> torch.Tensor:
    """Fold elapsed training into the success rate: ``1 - (1 - sr)(1 - rho)``.

    A schedule keyed on success alone never advances for a task whose success
    never arrives: the release condition is the success the weight is
    suppressing.  Nudging the rate toward 1 as training elapses adds the missing
    axis for free, because under the monotone kernel it factorises exactly:

        B(effective_sr; 0, k) = (1 - sr)^k (1 - rho)^k

    "decays as the task learns" and "decays as the run proceeds" in one
    expression, rather than a schedule wrapped around a schedule.
    """
    rho = min(max(float(anneal_fraction), 0.0), 1.0)
    return 1.0 - (1.0 - task_sr.clamp(0.0, 1.0)) * (1.0 - rho)


def envelope_from_range(dynamic_range: float) -> tuple[float, float]:
    """``(iw_min, iw_max)`` realising a peak-to-floor ratio of ``dynamic_range``.

    Both consumers divide the weights by their batch mean before use, so the
    envelope has one degree of freedom, not two:

        w = lo + (hi - lo) * w_hat = (hi - lo) * (w_hat + r),   r = lo / (hi - lo)

    the ``(hi - lo)`` factor cancels under ``w / mean(w)``, and the ratio the
    shape spans is ``(1 + r) / r``.  ``(0.5, 2)``, ``(1, 4)`` and ``(0.25, 1)``
    are the same setting written three ways; naming the ratio says which one.
    """
    d = float(dynamic_range)
    if d <= 1.0:
        raise ValueError(f"iw_dynamic_range must exceed 1 (got {d}); 1 means uniform weights")
    r = 1.0 / (d - 1.0)
    return r, 1.0 + r


def validate_iw_sr_shape(shape: str) -> str:
    """Raise early on a mistyped shape rather than silently falling back."""
    if shape not in VALID_IW_SR_SHAPES:
        raise ValueError(f"iw_sr_shape={shape!r} is not one of {VALID_IW_SR_SHAPES}")
    return shape


def task_sr_weight(
    task_sr: torch.Tensor,
    mean_sr: torch.Tensor,
    *,
    iw_min: float,
    iw_max: float,
    shape: str = "sigmoid",
    slope: float = 10.0,
    beta_target: float | None = -1.0,
    beta_concentration: float = 1.0,
    beta_eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample weight in ``[iw_min, iw_max]`` from a per-task SR EMA.

    Args:
        task_sr: SR EMA gathered per sample, any shape.
        mean_sr: scalar mean SR over the tasks actually observed.  Used as the
            reference point by ``sigmoid``, and as the target by ``beta`` when
            ``beta_target`` is negative or None.
        beta_target: mode of the beta kernel.  Negative (the -1.0 default) or
            ``None`` follows ``mean_sr``,
            which keeps the sigmoid path's relative-to-mean adaptivity -- the
            focus moves with the fleet as it improves.  A fixed value targets an
            absolute learnability frontier instead (0.5 is the Success Guided
            Sampling default).  SR is in [0, 1], so any negative value is
            unambiguously "follow the mean" -- the sentinel exists because a
            ``float | None`` field cannot be overridden from the Hydra command
            line (the resolved type pins to NoneType and a float override is
            rejected), which made this parameter unsweepable.
            The two disagree when the spread is wide and the
            mean is not at the frontier: with tasks at 0.7-0.99 the adaptive
            target focuses on ~0.85 and down-weights the hardest task at 0.7,
            which is usually not what is wanted.
    """
    iw_min = float(iw_min)
    iw_max = float(iw_max)

    if shape == "sigmoid":
        # Below-mean -> positive argument -> toward iw_max; above -> iw_min.
        frac = torch.sigmoid(float(slope) * (mean_sr - task_sr))
        return iw_min + (iw_max - iw_min) * frac

    validate_iw_sr_shape(shape)

    eps = float(beta_eps)
    kappa = float(beta_concentration)
    if beta_target is None or float(beta_target) < 0.0:
        target = mean_sr.clamp(eps, 1.0 - eps)
    else:
        target = torch.full_like(mean_sr, float(beta_target)).clamp(eps, 1.0 - eps)
    a = kappa * target
    b = kappa * (1.0 - target)
    sr = task_sr.clamp(0.0, 1.0)
    log_w = a * torch.log(sr + eps) + b * torch.log(1.0 - sr + eps)
    log_peak = a * torch.log(target + eps) + b * torch.log(1.0 - target + eps)
    w_hat = torch.exp(log_w - log_peak).clamp(0.0, 1.0)
    return iw_min + (iw_max - iw_min) * w_hat
