# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bound the disk a run spends on intermediate checkpoints.

Periodic saves exist for crash recovery, so they have to be frequent, but almost
none of them are ever read: what a run is later asked for is a milestone or its
final weights.  At the on-policy settings here -- ``save_interval=100`` over
30,000 iterations, ~9.5 MB per checkpoint -- keeping all of them costs ~2.9 GB
per run, which at five methods times three seeds is ~45 GB of files that exist
only because nothing deleted them.  Retaining every 2000 iterations instead
leaves 15 milestones plus the rolling point and the final weights: ~160 MB.

After each periodic save the pruner keeps:

* every checkpoint whose iteration is a multiple of ``history_step`` (milestones);
* the single most recent intermediate (the rolling resume point);
* ``model_final.pt``, which does not match the pattern and is never touched.

``history_step <= 0`` disables pruning. :class:`PrunedOnPolicyRunner` adds this
retention policy to the upstream on-policy runner.
"""

from __future__ import annotations

import os
import re

from rsl_rl.runners import OnPolicyRunner

CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")

# Fallback when the runner configuration omits the retention interval.
DEFAULT_HISTORY_CHECKPOINT_STEP: int = 2000


def prune_checkpoints(log_dir: str, history_step: int, *, source: str = "checkpoint_pruning") -> int:
    """Delete intermediate ``model_<iter>.pt`` files in ``log_dir``; return how many.

    Args:
        log_dir: Directory holding the run's checkpoints.
        history_step: Iteration granularity to retain; ``<= 0`` disables pruning.
        source: Label used in the warning emitted when a delete fails.

    Returns:
        Number of files removed.
    """
    if not history_step or history_step <= 0:
        return 0
    try:
        entries = os.listdir(log_dir)
    except OSError:
        return 0

    numbered = [(int(m.group(1)), name) for name in entries if (m := CHECKPOINT_RE.match(name))]
    if not numbered:
        return 0
    numbered.sort()
    latest_iter = numbered[-1][0]

    removed = 0
    for it, name in numbered:
        if it % history_step == 0 or it == latest_iter:
            continue
        try:
            os.remove(os.path.join(log_dir, name))
            removed += 1
        except OSError as exc:
            print(f"[{source}] WARNING: failed to prune checkpoint {name}: {exc}")
    return removed


class PrunedOnPolicyRunner(OnPolicyRunner):
    """:class:`~rsl_rl.runners.OnPolicyRunner` that prunes after every save.

    Upstream saves on a fixed interval and never removes anything.  Overriding
    :meth:`save` is enough because ``learn()`` reaches every periodic save through
    it, and the final save writes ``model_final.pt``, which the pattern excludes.

    ``history_checkpoint_step`` is read from the agent cfg dict, so it is settable
    from a runner cfg or a Hydra override.
    """

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        """Store the pruning granularity, then build the stock runner."""
        self.history_checkpoint_step: int = int(
            train_cfg.get("history_checkpoint_step", DEFAULT_HISTORY_CHECKPOINT_STEP)
        )
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save as usual, then drop the intermediates this run will never read."""
        super().save(path, infos)
        # Rank 0 owns the log dir under torch.distributed; the others share it and
        # would race on the same unlinks.
        if getattr(self, "gpu_global_rank", 0) != 0:
            return
        log_dir = getattr(getattr(self, "logger", None), "log_dir", None) or os.path.dirname(path)
        prune_checkpoints(log_dir, self.history_checkpoint_step, source="PrunedOnPolicyRunner")
