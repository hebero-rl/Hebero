"""RFCL curriculum scheduler and env wrapper for RSL-RL + Isaac Lab.

Usage in the training script (after creating the gym env, before wrapping with
RslRlVecEnvWrapper):

    from isaaclab_rl.rsl_rl import RFCLCurriculumWrapper, RslRlRFCLCfg

    rfcl_cfg = RslRlRFCLCfg(demo_hdf5_paths=[...], ...)
    env = RFCLCurriculumWrapper(env, rfcl_cfg, device=agent_cfg.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

The wrapper intercepts ``reset()`` and ``step()`` to:
  1. Provide RFCL-selected initial states to ``SourceLiberoCommand`` before each reset.
  2. Update curriculum progress counters after each episode.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .rfcl_cfg import RslRlRFCLCfg


# ---------------------------------------------------------------------------
# State storage helpers
# ---------------------------------------------------------------------------

def _load_demo_states(hdf5_path: str) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], np.ndarray]:
    """Load per-timestep physics states from a demo HDF5 file.

    Supports two file formats:

    1. ``states/`` group (replayed_demos_ format):
       Each demo has ``data/{demo}/states/{asset_type}/{asset_name}/{key}``
       with shape ``(T, D)``.

    2. ``initial_state/`` with T > 1 (assembled-demo format):
       Each demo has ``data/{demo}/initial_state/{asset_type}/{asset_name}/{key}``
       with shape ``(T, D)`` storing all trajectory timestep states.

    Returns ``(concatenated_states, lengths)`` where:
      - ``concatenated_states``: ``{asset_type -> {asset_name -> {key -> ndarray(total_T, D)}}}``
      - ``lengths``: per-demo trajectory lengths, shape ``(n_demos,)``
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py is required for RFCL demo loading. pip install h5py") from e

    def _read_state_group(grp) -> tuple[dict, int | None]:
        """Read one nested state group; return (nested_dict, T)."""
        out: dict = {}
        T = None
        for asset_type in grp.keys():
            out[asset_type] = {}
            for asset_name in grp[asset_type].keys():
                out[asset_type][asset_name] = {}
                for key, ds in grp[asset_type][asset_name].items():
                    arr = ds[:]
                    if T is None:
                        T = arr.shape[0]
                    out[asset_type][asset_name][key] = arr
        return out, T

    all_states: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {}
    lengths: list[int] = []

    with h5py.File(hdf5_path, "r") as f:
        for demo_key in sorted(f["data"].keys()):
            demo_grp = f[f"data/{demo_key}"]

            # Priority 1: explicit 'states/' group (replayed_demos_ format)
            if "states" in demo_grp:
                state_dict, T = _read_state_group(demo_grp["states"])
            # Priority 2: 'initial_state/' with T > 1 (assembled-demo format)
            elif "initial_state" in demo_grp:
                state_dict, T = _read_state_group(demo_grp["initial_state"])
                # Only use when multi-timestep (single initial state is not useful for curriculum)
                if T is None or T <= 1:
                    continue
            else:
                continue

            for asset_type, assets in state_dict.items():
                all_states.setdefault(asset_type, {})
                for asset_name, keys in assets.items():
                    all_states[asset_type].setdefault(asset_name, {})
                    for key, arr in keys.items():
                        all_states[asset_type][asset_name].setdefault(key, [])
                        all_states[asset_type][asset_name][key].append(arr)
            if T is not None:
                lengths.append(T)

    concat_states: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for asset_type, assets in all_states.items():
        concat_states[asset_type] = {}
        for asset_name, keys in assets.items():
            concat_states[asset_type][asset_name] = {k: np.concatenate(v, axis=0) for k, v in keys.items()}

    return concat_states, np.array(lengths, dtype=np.int64)


def _slice_state_at(states: dict, idx: int) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Slice per-timestep states at index ``idx`` → shape (1, D) per leaf."""
    out: dict = {}
    for asset_type, assets in states.items():
        out[asset_type] = {}
        for asset_name, keys in assets.items():
            out[asset_type][asset_name] = {k: v[idx : idx + 1] for k, v in keys.items()}
    return out


# ---------------------------------------------------------------------------
# Curriculum scheduler
# ---------------------------------------------------------------------------

class RFCLCurriculumScheduler:
    """Manages Phase 1 (reverse) and Phase 2 (forward) demo-state curricula.

    Attributes
    ----------
    phase : int
        Current curriculum phase (1 or 2).
    demo_ptrs : np.ndarray, shape (n_tasks, max_demos_per_task)
        Per-demo time-step cursors used in Phase 1.
    phase2_scores : np.ndarray, shape (total_states,)
        Prioritised sampling scores for Phase 2.
    """

    def __init__(
        self,
        cfg: RslRlRFCLCfg,
        device: str = "cpu",
        focus_set: frozenset[tuple[str, int]] | None = None,
        task_sequence: tuple[tuple[str, int], ...] | None = None,
    ):
        self.cfg = cfg
        self.device = device

        # Resolve HDF5 paths: explicit list takes priority; directory scan is the fallback.
        hdf5_paths = list(cfg.demo_hdf5_paths)
        if not hdf5_paths and cfg.demo_hdf5_dir:
            import glob as _glob
            import os as _os
            found = sorted(_glob.glob(_os.path.join(cfg.demo_hdf5_dir, "*.hdf5")))
            if not found:
                raise FileNotFoundError(
                    f"[RFCL] No *.hdf5 files found in demo_hdf5_dir='{cfg.demo_hdf5_dir}'"
                )
            hdf5_paths = found
            print(f"[RFCL] Auto-discovered {len(hdf5_paths)} HDF5 files from '{cfg.demo_hdf5_dir}'")

        # When LIBERO_FOCUS_TASKS is active, filter to demos matching the focused subset
        # so the curriculum only iterates over tasks the policy is actually training on.
        if focus_set is not None:
            from .demo_focus import filter_demo_paths, parse_demo_basename
            import os as _os
            before = len(hdf5_paths)
            hdf5_paths = filter_demo_paths(hdf5_paths, focus_set)
            if len(hdf5_paths) == 0:
                raise FileNotFoundError(
                    f"[RFCL] LIBERO_FOCUS_TASKS={sorted(focus_set)} matched 0 demo files "
                    f"(searched {before}).  Check filename convention '{{suite}}_task{{id}}_*.hdf5'."
                )
            if len(hdf5_paths) != len(focus_set):
                matched = {
                    parsed for p in hdf5_paths
                    if (parsed := parse_demo_basename(_os.path.basename(p))) is not None
                }
                missing = focus_set - matched
                raise FileNotFoundError(
                    f"[RFCL] LIBERO_FOCUS_TASKS has {len(focus_set)} tasks but only "
                    f"{len(hdf5_paths)} matching demo files found.  Missing: {sorted(missing)}"
                )
            print(
                f"[RFCL] LIBERO_FOCUS_TASKS active: filtered demos {before} → {len(hdf5_paths)} "
                f"(tasks: {sorted(focus_set)})"
            )

        # Reorder demos so demo index ``i`` corresponds to env task_id ``i``.
        # Without this, multi-task RFCL silently injects libero_goal states into
        # libero_object envs etc., because sorted(glob) is alphabetical
        # (libero_10 < libero_goal < libero_object < libero_spatial) while the
        # env's task_sequence is canonical
        # (libero_10, libero_object, libero_spatial, libero_goal).  See
        # ``sort_demo_paths_by_task_sequence`` for the full rationale.
        if task_sequence is not None:
            from .demo_focus import sort_demo_paths_by_task_sequence
            before_order = list(hdf5_paths)
            hdf5_paths = sort_demo_paths_by_task_sequence(hdf5_paths, task_sequence)
            if hdf5_paths != before_order:
                print(
                    f"[RFCL] Reordered demos to match env task_sequence "
                    f"(was alphabetical; {len(hdf5_paths)} files)."
                )

        # Load states per task
        self._task_states: list[dict] = []       # per-task concatenated state dict
        self._task_lengths: list[np.ndarray] = []  # lengths[i] = T for demo i in task
        self._task_offsets: list[np.ndarray] = []  # cumulative start index per demo
        # Parsed (suite, task_id) per loaded task — used for per-task / per-suite TB
        # metric breakdown.  None entry when basename does not match
        # ``{suite}_task{id}_*.hdf5`` (e.g. custom replayed-demo file names).
        self._task_labels: list[tuple[str, int] | None] = []

        from .demo_focus import parse_demo_basename
        import os as _os
        for path in hdf5_paths:
            states, lengths = _load_demo_states(path)
            self._task_states.append(states)
            self._task_lengths.append(lengths)
            offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
            self._task_offsets.append(offsets)
            self._task_labels.append(parse_demo_basename(_os.path.basename(path)))

        n_tasks = len(hdf5_paths)

        # Phase 1: per-task per-demo cursor
        # demo_ptrs[task_id][demo_idx] starts at init_cursor_progress * (T-1)
        # (default 1.0 → T-1, matches the paper).  Lower values skip the trivially-easy
        # near-goal prefix; useful when those samples inflate frontier SR.
        init_p = float(np.clip(getattr(cfg, "init_cursor_progress", 1.0), 0.0, 1.0))
        self._demo_ptrs: list[list[int]] = []
        for tid in range(n_tasks):
            self._demo_ptrs.append(
                [max(int(round(init_p * (int(t) - 1))), cfg.min_ptr) for t in self._task_lengths[tid]]
            )

        # Phase 1 per-demo success deque (m=per_demo_buffer_size most recent frontier episodes).
        # Pre-fill with zeros: a single success cannot push mean to 1.0 prematurely.
        # Used ONLY for the curriculum-advance decision (mean(buf) >= 1.0 → all m succeed).
        # Not a good monitoring signal — sits at 0 most of the time and spikes to 1.0
        # right before an advance.  For TensorBoard logging use ``_task_sr_ema`` below.
        m = int(self.cfg.per_demo_buffer_size)
        self._demo_succ_buf: list[list[deque]] = [
            [deque([0] * m, maxlen=m) for _ in range(len(self._task_lengths[tid]))]
            for tid in range(n_tasks)
        ]

        # Per-task smooth SR EMA over frontier-aligned episodes.  This is the
        # signal we log to TB for per-task / per-suite monitoring — same smoothing
        # constant as the global EMA but maintained independently per task.
        self._task_sr_ema: list[float] = [0.0] * n_tasks
        self._task_sr_init: list[bool] = [False] * n_tasks

        # Per-task phase (1=reverse, 2=forward).  A task enters phase 2 when its
        # *active* phase-1 demo subset (the first ``cfg.phase1_demo_count`` demos,
        # or all demos when that is None) has all cursors at ``cfg.min_ptr``.
        # In Phase 2 the full demo pool is unlocked and resets sample ``ptr=0``
        # from any demo — each demo's t=0 contributes a distinct natural initial
        # condition, broadening the init-state distribution the policy must solve.
        self._task_phase: list[int] = [1] * n_tasks

        # Per-task active phase-1 demo count (clamped to actual demo count per task
        # because tasks may have fewer demos than ``phase1_demo_count``).
        p1 = cfg.phase1_demo_count
        self._task_phase1_count: list[int] = []
        for tid in range(n_tasks):
            n_demos_tid = len(self._task_lengths[tid])
            self._task_phase1_count.append(min(int(p1), n_demos_tid) if p1 is not None else n_demos_tid)

        # Global SR tracking (monitoring only — no longer drives the phase switch).
        self._global_sr_ema: float = 0.0
        self._global_sr_init: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def phase(self) -> int:
        """Coarse global phase: 1 if any task is still in Phase 1, else 2.

        Phase is tracked **per-task** in ``self._task_phase``; this property exists
        only for backward-compat with callers and tests that read a single phase
        scalar.  Prefer ``self._task_phase[task_id]`` in new code.
        """
        return 1 if any(p == 1 for p in self._task_phase) else 2

    def validate_task_count(self, n_tasks: int) -> None:
        """Raise ValueError if n_tasks doesn't match the number of loaded HDF5 files."""
        n_loaded = len(self._task_states)
        if n_loaded > 0 and n_loaded != n_tasks:
            raise ValueError(
                f"[RFCL] HDF5 file count ({n_loaded}) != env task count ({n_tasks}). "
                f"Provide exactly one HDF5 file per task, either via demo_hdf5_paths or "
                f"demo_hdf5_dir (with {n_tasks} files inside)."
            )

    def get_initial_state(self, task_id: int, demo_idx: int | None = None) -> dict:
        """Return the initial physics state for one env at reset time.

        Parameters
        ----------
        task_id : int
            Integer task assignment from the env extras.
        demo_idx : int | None
            Which demo within the task.  When None a random demo is chosen.

        Returns
        -------
        State dict with the same nested structure as ``initial_state/`` in the HDF5:
        ``{asset_type -> {asset_name -> {key -> ndarray(1, D)}}}``.
        """
        state, _, _ = self.get_initial_state_and_ptr(task_id, demo_idx)
        return state

    def get_initial_state_and_ptr(
        self, task_id: int, demo_idx: int | None = None
    ) -> tuple[dict, int, bool]:
        """Return ``(state, ptr, at_frontier)``.

        - ``ptr`` is the trajectory timestep used to slice the state.  The command term
          must start replaying the demo trajectory from the same timestep so teacher
          actions stay coherent with the injected physical state.
        - ``at_frontier`` is True iff the sampled ``ptr`` equals the current Phase 1
          cursor ``t_i = self._demo_ptrs[task_id][demo_idx]``.  Only frontier samples
          should be fed into the per-demo success buffer (see ``update_success``).
          Always False in Phase 2 (forward curriculum has no "frontier" concept).
        """
        if task_id >= len(self._task_states):
            return {}, 0, False

        n_demos = len(self._task_lengths[task_id])
        if n_demos == 0:
            return {}, 0, False

        if demo_idx is None:
            demo_idx = int(np.random.randint(0, n_demos))
        else:
            demo_idx = int(demo_idx) % n_demos  # wrap to valid range

        # Phase 2: inject demo state at ptr=0 from the (already-uniformly-resampled)
        # demo idx — each demo's first timestep is a distinct natural initial
        # condition.  The full demo pool is unlocked here (sample_demo_idx already
        # samples from all demos, not just the phase-1 subset), so phase 2
        # effectively widens the init-state distribution beyond what the reverse
        # curriculum saw.  ``at_frontier=True`` keeps successes feeding SR EMAs.
        if self._task_phase[task_id] == 2:
            global_idx = int(self._task_offsets[task_id][demo_idx]) + 0
            return _slice_state_at(self._task_states[task_id], global_idx), 0, True

        T = int(self._task_lengths[task_id][demo_idx])
        base = int(self._demo_ptrs[task_id][demo_idx])
        ptr = self._sample_phase1_start_step(base, T)
        at_frontier = ptr == base

        global_idx = int(self._task_offsets[task_id][demo_idx]) + int(ptr)
        return _slice_state_at(self._task_states[task_id], global_idx), int(ptr), bool(at_frontier)

    def sample_demo_idx(self, task_id: int) -> int:
        """Sample a demo index within ``task_id`` using ``cfg.demo_density_mode``.

        Phase 1: only the first ``cfg.phase1_demo_count`` demos are eligible (or
        all demos when that is None).  ``"adaptive"`` density ∝ ``t_i / T_i``.
        Reverse-solved demos (``t_i <= min_ptr``) drop to ``1e-6``.

        Phase 2: the full demo pool is unlocked and sampled uniformly — each
        demo's ``ptr=0`` state is a distinct natural initial condition.
        """
        if task_id >= len(self._task_lengths):
            return 0
        n_demos = len(self._task_lengths[task_id])
        if n_demos == 0:
            return 0
        if self._task_phase[task_id] == 2:
            # Phase 2: all demos eligible; ptr=0 from each is a distinct init state.
            return int(np.random.randint(0, n_demos))
        # Phase 1: restrict to the active subset.
        n_active = self._task_phase1_count[task_id]
        if n_active <= 0:
            return 0
        if self.cfg.demo_density_mode == "uniform":
            return int(np.random.randint(0, n_active))
        weights = np.empty(n_active, dtype=np.float64)
        for d in range(n_active):
            t_i = int(self._demo_ptrs[task_id][d])
            T_i = max(int(self._task_lengths[task_id][d]), 1)
            if t_i <= 0:
                weights[d] = 1e-6
            else:
                weights[d] = float(t_i) / float(T_i)
        s = weights.sum()
        if s <= 0.0:
            return int(np.random.randint(0, n_active))
        weights /= s
        return int(np.random.choice(n_active, p=weights))

    def _sample_phase1_start_step(self, base: int, T: int) -> int:
        """Sample a Phase 1 start timestep using the distribution K from the paper.

        ``base`` is the current per-demo cursor ``t_i``; ``T`` is the demo length.
        Returns a ptr in ``[0, T - 1]``.
        """
        sampler = self.cfg.start_step_sampler
        T = max(int(T), 1)
        base = int(np.clip(base, 0, T - 1))
        if sampler == "fixed_point":
            return base
        if sampler == "geometric":
            cands = np.array([min(base + i, T - 1) for i in range(5)], dtype=np.int64)
            density = np.array([0.5, 0.25, 0.125, 0.0625, 0.0625], dtype=np.float64)
            return int(np.random.choice(cands, p=density))
        if sampler == "uniform":
            return int(np.random.randint(0, T))
        if sampler == "uniform_spike":
            # 50% at base, 50% uniform over [0, T).
            if np.random.rand() < 0.5:
                return base
            return int(np.random.randint(0, T))
        if sampler == "uniform_step":
            # 50% at base, 50% uniform over [0, base].
            if base == 0 or np.random.rand() < 0.5:
                return base
            return int(np.random.randint(0, base + 1))
        raise ValueError(
            f"[RFCL] Unknown start_step_sampler={sampler!r}; expected one of "
            f"fixed_point / geometric / uniform / uniform_spike / uniform_step"
        )

    def update_success(
        self,
        task_ids: torch.Tensor,
        demo_indices: torch.Tensor,
        successes: torch.Tensor,
        dones: torch.Tensor,
        at_frontier: torch.Tensor | None = None,
    ) -> None:
        """Update per-demo success buffers and advance curriculum pointers.

        Parameters
        ----------
        task_ids : (num_envs,) int tensor
        demo_indices : (num_envs,) int tensor  – which demo each env is on
        successes : (num_envs,) float tensor   – 0/1 outcome per env
        dones : (num_envs,) bool tensor
        at_frontier : (num_envs,) bool tensor | None
            Whether each env was sampled at the current Phase 1 cursor
            (``ptr == t_i``).  Only frontier-aligned episodes are pushed into
            the per-demo success buffer AND into the global SR EMA — non-frontier
            (rehearsal) samples are easier states near the goal and would
            inflate both signals.  Falls back to all-frontier when None for
            backward compat with callers that don't track this.
        """
        done_mask = dones.view(-1).bool().cpu().numpy()
        if not done_mask.any():
            return

        task_ids_np = task_ids.view(-1).cpu().numpy()
        demo_idxs_np = demo_indices.view(-1).cpu().numpy()
        succ_np = successes.view(-1).float().cpu().numpy()
        if at_frontier is None:
            frontier_np = np.ones_like(done_mask, dtype=bool)
        else:
            frontier_np = at_frontier.view(-1).bool().cpu().numpy()

        frontier_successes: list[int] = []
        per_task_frontier: dict[int, list[int]] = {}
        for env_idx in np.where(done_mask)[0]:
            if not frontier_np[env_idx]:
                continue
            tid = int(task_ids_np[env_idx])
            didx = int(demo_idxs_np[env_idx])
            if tid >= len(self._demo_succ_buf) or didx >= len(self._demo_succ_buf[tid]):
                continue
            s = int(round(float(succ_np[env_idx])))
            self._demo_succ_buf[tid][didx].append(s)
            frontier_successes.append(s)
            per_task_frontier.setdefault(tid, []).append(s)

        # Update per-task SR EMA over frontier successes (for TB logging only).
        alpha = float(self.cfg.sr_ema_alpha)
        for tid, vals in per_task_frontier.items():
            if tid >= len(self._task_sr_ema):
                continue
            batch_mean = float(np.mean(vals))
            if not self._task_sr_init[tid]:
                self._task_sr_ema[tid] = batch_mean
                self._task_sr_init[tid] = True
            else:
                self._task_sr_ema[tid] = (1.0 - alpha) * self._task_sr_ema[tid] + alpha * batch_mean

        self._maybe_advance_phase1()
        # Global SR EMA only sees frontier episodes — kept for monitoring only.
        # Phase 2 entry is per-task (driven by cursor reaching min_ptr), not by a
        # global SR threshold; the threshold-based switch was removed because it
        # caused critic explosion when slow tasks were dragged into Phase 2
        # prematurely by fast tasks raising the global SR.
        if frontier_successes:
            self._update_global_sr(np.asarray(frontier_successes, dtype=np.float32))

    def curriculum_info(self) -> dict:
        """Return diagnostic info for logging.

        Key metrics:
          rfcl/progress                   normalised mean cursor (ptr / (T-1)), 1.0 → 0.0
                                          Phase-2 tasks contribute 0.0.
          rfcl/init_state_sampled_timestep mean absolute ptr (timestep index into demo)
          rfcl/global_sr                  global per-episode success-rate EMA (frontier-only)
          rfcl/phase                      coarse global phase (1 if any task still in Phase 1)
          rfcl/phase_frac_phase2          fraction of tasks that have entered Phase 2
          rfcl/phase/task/{tag}           per-task phase (1 or 2)
        """
        info: dict[str, float] = {"rfcl/phase": float(self.phase)}
        task_abs: list[float] = []
        task_norm: list[float] = []
        task_sr: list[float] = []
        task_buf_mean: list[float] = []
        task_phase_vals: list[float] = []

        for tid in range(len(self._demo_ptrs)):
            ptrs = self._demo_ptrs[tid]
            lengths = self._task_lengths[tid]
            bufs = self._demo_succ_buf[tid]
            if len(ptrs) == 0:
                continue
            ph = int(self._task_phase[tid])
            if ph == 2:
                # Phase-2 task: every reset injects ptr=0, so progress/timestep are 0.
                t_abs = 0.0
                t_norm = 0.0
                t_buf = 0.0
            else:
                t_abs = float(np.mean([float(p) for p in ptrs]))
                t_norm = float(np.mean([float(p) / max(int(T) - 1, 1) for p, T in zip(ptrs, lengths)]))
                t_buf = float(np.mean([float(np.mean(b)) for b in bufs if len(b) > 0])) if bufs else 0.0
            t_sr = float(self._task_sr_ema[tid]) if self._task_sr_init[tid] else 0.0
            task_abs.append(t_abs)
            task_norm.append(t_norm)
            task_sr.append(t_sr)
            task_buf_mean.append(t_buf)
            task_phase_vals.append(float(ph))

            label = self._task_labels[tid] if tid < len(self._task_labels) else None
            tag = f"{label[0]}_task{label[1]}" if label is not None else f"idx{tid}"
            info[f"rfcl/task_sr/task/{tag}"] = t_sr
            info[f"rfcl/advance_ready/task/{tag}"] = t_buf
            info[f"rfcl/progress/task/{tag}"] = t_norm
            info[f"rfcl/init_state_sampled_timestep/task/{tag}"] = t_abs
            info[f"rfcl/phase/task/{tag}"] = float(ph)

        if task_abs:
            info["rfcl/init_state_sampled_timestep"] = float(np.mean(task_abs))
            info["rfcl/progress"] = float(np.mean(task_norm))
            info["rfcl/task_sr"] = float(np.mean(task_sr))
            info["rfcl/advance_ready"] = float(np.mean(task_buf_mean))
            info["rfcl/phase_frac_phase2"] = float(np.mean([1.0 if v >= 2.0 else 0.0 for v in task_phase_vals]))

        # Per-suite roll-up.  Indexing assumes every task contributed to task_*
        # lists in the loop above (i.e. all tasks have non-empty ptrs, which is
        # the only case where they're emitted).
        suite_sr: dict[str, list[float]] = {}
        suite_buf: dict[str, list[float]] = {}
        suite_norm: dict[str, list[float]] = {}
        suite_abs: dict[str, list[float]] = {}
        suite_phase: dict[str, list[float]] = {}
        for tid, label in enumerate(self._task_labels):
            if label is None or tid >= len(task_sr):
                continue
            suite = label[0]
            suite_sr.setdefault(suite, []).append(task_sr[tid])
            suite_buf.setdefault(suite, []).append(task_buf_mean[tid])
            suite_norm.setdefault(suite, []).append(task_norm[tid])
            suite_abs.setdefault(suite, []).append(task_abs[tid])
            suite_phase.setdefault(suite, []).append(task_phase_vals[tid])
        for suite, vals in suite_sr.items():
            info[f"rfcl/task_sr/suite/{suite}"] = float(np.mean(vals))
        for suite, vals in suite_buf.items():
            info[f"rfcl/advance_ready/suite/{suite}"] = float(np.mean(vals))
        for suite, vals in suite_norm.items():
            info[f"rfcl/progress/suite/{suite}"] = float(np.mean(vals))
        for suite, vals in suite_abs.items():
            info[f"rfcl/init_state_sampled_timestep/suite/{suite}"] = float(np.mean(vals))
        for suite, vals in suite_phase.items():
            info[f"rfcl/phase_frac_phase2/suite/{suite}"] = float(
                np.mean([1.0 if v >= 2.0 else 0.0 for v in vals])
            )

        info["rfcl/global_sr"] = self._global_sr_ema_value()
        return info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_advance_phase1(self) -> None:
        """Advance per-demo cursor when its frontier-success buffer is full of 1s.

        Matches the official RFCL criterion: ``mean(success_buffer) >= 1.0`` —
        the last ``per_demo_buffer_size`` frontier episodes must ALL succeed.
        After advancing, the buffer is reset to zeros so the new difficulty
        level has to re-prove itself.

        When a task's demos all reach ``cfg.min_ptr`` the task transitions to
        Phase 2 (forward training from every demo's ``ptr=0`` state).  Phase-2
        tasks are skipped here so their per-demo deques stop driving anything.
        """
        m = int(self.cfg.per_demo_buffer_size)
        min_ptr = int(self.cfg.min_ptr)
        for tid in range(len(self._demo_succ_buf)):
            if self._task_phase[tid] == 2:
                continue
            n_active = self._task_phase1_count[tid]
            advanced_any = False
            for didx in range(n_active):
                buf = self._demo_succ_buf[tid][didx]
                if float(np.mean(buf)) < 1.0:
                    continue
                current = self._demo_ptrs[tid][didx]
                new_ptr = max(min_ptr, current - self.cfg.reverse_step_size)
                if new_ptr < current:
                    self._demo_ptrs[tid][didx] = new_ptr
                    buf.clear()
                    for _ in range(m):
                        buf.append(0)
                    advanced_any = True

            # Task enters Phase 2 once every *active* demo in the phase-1 subset
            # has reverse-solved (cursor at min_ptr).  Held-out demos (idx >= n_active)
            # are ignored here; they get unlocked at Phase 2 entry.
            if advanced_any and all(
                int(self._demo_ptrs[tid][d]) <= min_ptr for d in range(n_active)
            ):
                self._task_phase[tid] = 2
                label = self._task_labels[tid] if tid < len(self._task_labels) else None
                tag = f"{label[0]}_task{label[1]}" if label is not None else f"idx{tid}"
                n_total = len(self._demo_ptrs[tid])
                print(
                    f"[RFCL] Task {tag} → Phase 2 (all {n_active} active demos reverse-solved). "
                    f"Phase 2 unlocks full demo pool: {n_total} demos, ptr=0 each."
                )

    def _update_global_sr(self, success_values: np.ndarray) -> None:
        if len(success_values) == 0:
            return
        batch_sr = float(success_values.mean())
        alpha = self.cfg.sr_ema_alpha
        if not self._global_sr_init:
            self._global_sr_ema = batch_sr
            self._global_sr_init = True
        else:
            self._global_sr_ema = (1.0 - alpha) * self._global_sr_ema + alpha * batch_sr

    def _global_sr_ema_value(self) -> float:
        return self._global_sr_ema if self._global_sr_init else 0.0

    # Phase 2 (forward) no longer uses prioritised timestep sampling.  A Phase-2 task
    # draws a demo uniformly from the full pool and always injects its ``ptr=0`` state
    # (see ``get_initial_state_and_ptr``), so no scheduler-side scoring state is needed.

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise mutable curriculum state for resume.

        Loaded HDF5 states/lengths/labels/offsets are immutable after init and
        therefore not persisted — they will be re-read from disk on resume.
        We snapshot only the runtime counters that the curriculum has built up
        (cursors, phase, deques, SR EMAs).
        """
        return {
            "demo_ptrs": [list(map(int, ptrs)) for ptrs in self._demo_ptrs],
            "demo_succ_buf": [
                [list(map(int, buf)) for buf in bufs]
                for bufs in self._demo_succ_buf
            ],
            "task_phase": list(map(int, self._task_phase)),
            "task_phase1_count": list(map(int, self._task_phase1_count)),
            "task_sr_ema": list(map(float, self._task_sr_ema)),
            "task_sr_init": list(map(bool, self._task_sr_init)),
            "global_sr_ema": float(self._global_sr_ema),
            "global_sr_init": bool(self._global_sr_init),
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore mutable curriculum state from a saved snapshot.

        Validates that the persisted task count matches the freshly loaded
        HDF5 task count — a mismatch usually means resume is being attempted
        with a different demo set, which would silently mis-index cursors.
        """
        n_tasks_now = len(self._demo_ptrs)
        n_tasks_sd = len(sd.get("demo_ptrs", []))
        if n_tasks_sd != n_tasks_now:
            raise ValueError(
                f"[RFCL] Checkpoint has {n_tasks_sd} tasks but current scheduler "
                f"loaded {n_tasks_now} tasks.  Demo set / focus filter likely "
                f"differs between save and resume."
            )

        m = int(self.cfg.per_demo_buffer_size)
        for tid in range(n_tasks_now):
            saved_ptrs = sd["demo_ptrs"][tid]
            if len(saved_ptrs) != len(self._demo_ptrs[tid]):
                raise ValueError(
                    f"[RFCL] Task {tid} demo count mismatch: ckpt has "
                    f"{len(saved_ptrs)} demos, scheduler has "
                    f"{len(self._demo_ptrs[tid])}."
                )
            self._demo_ptrs[tid] = [int(p) for p in saved_ptrs]

            saved_bufs = sd["demo_succ_buf"][tid]
            for didx, raw in enumerate(saved_bufs):
                self._demo_succ_buf[tid][didx] = deque(
                    [int(v) for v in raw][-m:], maxlen=m
                )
                # Pad short buffers (old m differs from current cfg) with zeros at the left.
                while len(self._demo_succ_buf[tid][didx]) < m:
                    self._demo_succ_buf[tid][didx].appendleft(0)

        self._task_phase = [int(p) for p in sd["task_phase"]]
        if "task_phase1_count" in sd:
            self._task_phase1_count = [int(c) for c in sd["task_phase1_count"]]
        self._task_sr_ema = [float(x) for x in sd["task_sr_ema"]]
        self._task_sr_init = [bool(x) for x in sd["task_sr_init"]]
        self._global_sr_ema = float(sd["global_sr_ema"])
        self._global_sr_init = bool(sd["global_sr_init"])
        n_phase2 = sum(1 for p in self._task_phase if p == 2)
        print(
            f"[RFCL] Curriculum state restored: {n_tasks_now} tasks, "
            f"{n_phase2} in Phase 2, global_sr_ema={self._global_sr_ema:.3f}"
        )


# ---------------------------------------------------------------------------
# Env wrapper
# ---------------------------------------------------------------------------

class RFCLEnvWrapper(gym.Wrapper):
    """Gymnasium wrapper that injects RFCL-selected initial states before each reset.

    Place this wrapper **before** ``RslRlVecEnvWrapper`` in the wrapper stack::

        env = gym.make(task_id, cfg=env_cfg)
        env = RFCLEnvWrapper(env, rfcl_cfg, device=device)
        env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

    The wrapper communicates with ``SourceLiberoCommand`` (the Isaac Lab command term
    that manages demo-trajectory playback) through its
    ``_rfcl_state_override`` attribute.  The attribute must be set on the command
    object before ``env.reset()`` is called; the command term reads it during its
    own reset to set the physics initial state.

    At each reset a new demo is sampled randomly for every resetting env.  The
    per-demo curriculum pointers (``_demo_ptrs[tid][didx]``) and success-rate EMAs
    are cached in the scheduler across episodes, so every demo advances at its own
    pace regardless of which envs happen to visit it.  This gives variance in the
    injected start state even when a demo's EMA is below the advance threshold.
    """

    def __init__(self, env: gym.Env, cfg: RslRlRFCLCfg, device: str = "cpu"):
        super().__init__(env)
        self.cfg = cfg
        self.device = device
        # Resolve focused-task filter from env (None when LIBERO_FOCUS_TASKS is unset
        # or covers all 40 canonical tasks).  Both demo loading and _get_task_ids
        # below pivot on this so internal task indexing stays consistent.
        from .demo_focus import resolve_focus_task_set, resolve_task_sequence
        self._focus_set = resolve_focus_task_set(env)
        # task_sequence (env-side ordering) — passed into the scheduler so demos
        # load in the same order the env emits task_ids.  Without this the
        # scheduler's alphabetical filesystem order misaligns with the env's
        # canonical [libero_10, libero_object, libero_spatial, libero_goal]
        # sequence and states get injected into the wrong suite's envs.
        self._task_sequence = resolve_task_sequence(env)
        self.scheduler = RFCLCurriculumScheduler(
            cfg,
            device=device,
            focus_set=self._focus_set,
            task_sequence=self._task_sequence,
        )

        # Validate that the number of loaded HDF5 files matches the env's task count.
        n_tasks = self._get_n_unique_tasks()
        if n_tasks is not None:
            self.scheduler.validate_task_count(n_tasks)

        n_loaded = len(self.scheduler._task_states)
        num_envs: int = getattr(env.unwrapped, "num_envs", 1)
        # Use max n_demos across all tasks so every demo (not just the first n_tasks) can be selected.
        # get_initial_state_and_ptr wraps didx via % n_demos so any value >= n_demos is safe.
        max_n_demos = max(
            (len(lengths) for lengths in self.scheduler._task_lengths if len(lengths) > 0),
            default=n_loaded,
        )
        self._env_demo_indices = np.random.randint(0, max(1, max_n_demos), size=num_envs)
        self._env_task_ids = np.zeros(num_envs, dtype=np.int64)
        self._env_phase1_ptrs = np.zeros(num_envs, dtype=np.int64)
        # Per-env "was this sample drawn at the demo's current Phase 1 cursor t_i?" flag.
        # Only frontier-aligned episodes feed the per-demo success buffer; non-frontier
        # samples (drawn closer to the goal by start_step_sampler) are rehearsal-only.
        self._env_phase1_at_frontier = np.zeros(num_envs, dtype=bool)

        # Initialize task IDs from static config (assignments don't change during training).
        self._get_task_ids()

        # Register the RFCL scheduler with SourceLiberoCommand so that every per-episode
        # reset during training uses curriculum states, not just the initial reset().
        command_term = self._get_source_libero_command()
        if command_term is not None:
            command_term._rfcl_scheduler = self.scheduler
            command_term._rfcl_env_task_ids = self._env_task_ids
            command_term._rfcl_env_demo_indices = self._env_demo_indices
            command_term._rfcl_env_phase1_ptrs = self._env_phase1_ptrs
            command_term._rfcl_env_at_frontier = self._env_phase1_at_frontier
            print("[RFCL] Registered RFCL scheduler with SourceLiberoCommand for per-episode curriculum injection.")

    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        # NOTE (Hebero): no explicit state injection here — the
        # scheduler registered on SourceLiberoCommand drives EVERY reservation
        # (initial reset AND per-episode resets) inside the command's
        # ``reserve_trajectories_for_envs``, keeping physics state, teacher
        # trajectory cursor, and curriculum pointer coherent.
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action: Any):
        # Snapshot the per-env curriculum bookkeeping BEFORE stepping.  The env resets
        # done envs *inside* ``step`` (``GroupedManagerBasedRLEnv._reset_idx``), and that
        # reset chain reaches ``SourceLiberoCommand.reserve_trajectories_for_envs``, which
        # rewrites these two arrays in place with the *next* episode's demo index and
        # frontier flag.  Reading them after the step would credit the finished episode's
        # outcome to the demo the env is about to run, so every demo's success buffer
        # would be filled from a common per-task pool and the cursors would advance in
        # lockstep at the task's average success rate instead of per-demo merit.
        # ``info["task_success"]`` itself is computed pre-reset and needs no snapshot.
        demo_idxs = torch.as_tensor(self._env_demo_indices.copy(), dtype=torch.long)
        at_frontier = torch.as_tensor(self._env_phase1_at_frontier.copy(), dtype=torch.bool)
        obs, rew, terminated, truncated, info = self.env.step(action)
        dones = torch.as_tensor(terminated | truncated, dtype=torch.bool)
        task_ids = self._get_task_ids()  # updates self._env_task_ids in-place
        if task_ids is not None and "task_success" in info:
            successes = torch.as_tensor(info["task_success"], dtype=torch.float32).view(-1)
            self.scheduler.update_success(task_ids, demo_idxs, successes, dones, at_frontier)
        return obs, rew, terminated, truncated, info

    # ------------------------------------------------------------------

    def _inject_rfcl_states(self) -> None:
        """(Hebero) Legacy no-op — the curriculum drives resets via the command.

        In the playground repo this method pushed per-env ``_rfcl_state_override``
        / ``_rfcl_start_steps_override`` before ``env.reset()``. Here the bench's
        ``SourceLiberoCommand.reserve_trajectories_for_envs`` queries the
        registered scheduler directly on EVERY reset (initial and per-episode),
        so no wrapper-side injection is needed — the command reserves the
        scheduler-chosen demo at the curriculum pointer, slices the demo state at
        that timestep for the reset event, and updates this wrapper's
        ``_env_demo_indices`` / ``_env_phase1_ptrs`` / ``_env_phase1_at_frontier``
        arrays in place.
        """

    def _get_n_unique_tasks(self) -> int | None:
        """Return the number of unique tasks in the env, or None if not discoverable.

        Prefers ``task_sequence`` (focused subset under LIBERO_FOCUS_TASKS) over
        ``full_task_sequence`` (canonical 40) so the loaded demos match the env.
        """
        env_cfg = getattr(self.env.unwrapped, "cfg", None)
        libero_cfg = getattr(env_cfg, "libero_config", None) if env_cfg is not None else None
        if libero_cfg is None:
            return None
        sequence = getattr(libero_cfg, "task_sequence", None) or getattr(libero_cfg, "full_task_sequence", None)
        if sequence:
            return len(sequence)
        assignments = getattr(libero_cfg, "env_task_assignments", None)
        if assignments:
            return len(set(assignments))
        return None

    def _get_source_libero_command(self):
        command_manager = getattr(self.env.unwrapped, "command_manager", None)
        if command_manager is None:
            return None
        try:
            return command_manager.get_term("source_action")
        except KeyError:
            pass
        # stock CommandManager registers the demo term under its cfg attribute
        # name; find the term exposing "source_action" as a semantic view
        for name in command_manager.active_terms:
            term = command_manager.get_term(name)
            keys = getattr(getattr(term, "cfg", None), "semantic_keys", None)
            if keys and "source_action" in keys:
                return term
        return None

    def _get_task_ids(self) -> torch.Tensor | None:
        """Return per-env task ids using ``task_sequence``-relative indexing.

        Uses the focused ``task_sequence`` (not ``full_task_sequence``) so the
        emitted ids index into ``scheduler._task_states`` (which was loaded in
        the same focused order via ``filter_demo_paths``).  These ids are
        internal to the RFCL curriculum and are NOT used by the PPO side
        (which receives canonical ids from ``vecenv_wrapper`` separately).
        """
        env_cfg = getattr(self.env.unwrapped, "cfg", None)
        libero_cfg = getattr(env_cfg, "libero_config", None) if env_cfg is not None else None
        assignments = getattr(libero_cfg, "env_task_assignments", None)
        if assignments is None:
            return None
        sequence = getattr(libero_cfg, "task_sequence", None) or getattr(libero_cfg, "full_task_sequence", None)
        if not sequence:
            sequence = sorted(set(assignments))
        num_envs = len(self._env_task_ids)
        if len(assignments) != num_envs:
            # Baked from the cfg's default num_envs before a CLI --num_envs override;
            # re-tile the deterministic env i -> task i % n cycle (Hebero).
            unique_cycle = list(dict.fromkeys(assignments))
            assignments = [unique_cycle[i % len(unique_cycle)] for i in range(num_envs)]
        assignment_to_id = {a: i for i, a in enumerate(sequence)}
        ids = torch.tensor([assignment_to_id[a] for a in assignments], dtype=torch.long)
        self._env_task_ids[:] = ids.numpy()
        return ids
