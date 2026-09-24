# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo-replay command stack for LIBERO DGPO training (SourceLiberoCommand port).

Loads assembled HDF5 demos and exposes semantic views ``ee_pose``,
``gripper_state``, ``joint_state``, ``joint_velocity``, ``source_action`` from
one shared trajectory bank.

Demo quaternion order is controlled by :attr:`SourceLiberoCommandCfg.demo_quat_order`
(default ``wxyz``). ``initial_state/*/root_pose`` is converted to sim XYZW by
the HDF5 loader when legacy; ``obs/ee_states`` is converted here via
:func:`~...mdp.quat.pose7_to_sim`.

Environment variables
---------------------

* ``LIBERO_ASSEMBLED_DATASET_DIR`` — directory of
  ``{suite}_task{id}_*_demo.hdf5`` files.
* ``LIBERO_COMPAT_REQUIRE_DEMOS`` — if ``1``/``true``, missing demos raise
  instead of falling back to zero privileged diffs.
* ``LIBERO_COMPAT_MAX_DEMOS_PER_TASK`` — optional positive int; load at most
  this many episodes per task HDF5 (play/debug speedup; default = all).
* ``LIBERO_RANDOM_INIT_STATE`` — if ``1``/``true``, each object's reset XY is
  redrawn uniformly from the box its own demos span, instead of replaying the
  reserved demo's placement (default: off). See
  :attr:`SourceLiberoCommandCfg.randomize_initial_state_xy`. Stays on under
  ``LIBERO_EVALUATION``.
* ``LIBERO_RANDOM_INIT_XY_SCALE`` — float, default ``1.0``; grows each of those
  boxes about its centre. ``1.0`` samples the demos' own region, which is *in*
  distribution; use ``>1`` for an actually out-of-distribution placement sweep.
* ``LIBERO_RANDOM_INIT_TIMESTEP`` — if ``1``/``true``, episodes reset to a start
  timestep sampled uniformly along the demo trajectory (default: off, always
  reset to the demo's t=0 initial state). Inflates the training-time
  ``Metrics/task_sr/*`` EMAs: episodes that start near the demo's end succeed
  almost for free. Before 2026-08, this was what ``LIBERO_RANDOM_INIT_STATE``
  selected.
* ``LIBERO_EVALUATION`` — if set, force-disable random start-timestep sampling.
"""

from __future__ import annotations

import glob
import logging
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

from ...dgpo_layout import harvest_task_to_assignment_key
from ..quat import DEFAULT_DEMO_QUAT_ORDER, QuatOrder, pose7_to_sim
from .... import settings

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ...tasks.harvest.prototypes import TaskBinding

logger = logging.getLogger(__name__)

TrajectoryState = dict[str, dict[str, dict[str, torch.Tensor]]]

UNIFIED_LIBERO_SEMANTIC_KEYS: tuple[str, ...] = (
    "ee_pose",
    "gripper_state",
    "joint_state",
    "joint_velocity",
    "source_action",
)
UNIFIED_LIBERO_COMMAND_KEYS: tuple[str, ...] = (
    "obs/ee_states",
    "obs/gripper_states",
    "obs/joint_states",
    "obs/joint_velocities",
    "actions",
)

_DEFAULT_DEMO_CANDIDATES: tuple[str, ...] = (
    # Hugging Face pull at <repo>/datasets (README §2.3).
    "datasets/demos",
)

# Axis spread below which a demo-derived reset range counts as "this entity does not
# move between demos" (fixed fixtures) -- 0.1 mm, well under LIBERO's ~5 cm regions.
_XY_RANGE_EPS: float = 1e-4


def _scale_range(
    lo_x: float, hi_x: float, lo_y: float, hi_y: float, scale: float
) -> tuple[float, float, float, float]:
    """Grow (or shrink) an XY box about its own centre by ``scale``."""
    if scale == 1.0:
        return (lo_x, hi_x, lo_y, hi_y)
    mid_x, half_x = (lo_x + hi_x) * 0.5, (hi_x - lo_x) * 0.5 * scale
    mid_y, half_y = (lo_y + hi_y) * 0.5, (hi_y - lo_y) * 0.5 * scale
    return (mid_x - half_x, mid_x + half_x, mid_y - half_y, mid_y + half_y)


def resolve_libero_demos_root(*, require: bool | None = None) -> str | None:
    """Resolve assembled demo directory from env / known defaults.

    Args:
        require: Override for ``LIBERO_COMPAT_REQUIRE_DEMOS``. When ``True`` and
            no directory is found, raises :class:`FileNotFoundError`.

    Returns:
        Absolute path to the demos root, or ``None`` if demos are optional and
        unavailable.
    """
    if require is None:
        require = settings.truthy("LIBERO_COMPAT_REQUIRE_DEMOS")
    env_root = settings.text("LIBERO_ASSEMBLED_DATASET_DIR")
    candidates = [env_root] if env_root else list(_DEFAULT_DEMO_CANDIDATES)
    for path in candidates:
        if path and os.path.isdir(path):
            return os.path.abspath(path)
    msg = (
        "LIBERO demos not found. Set LIBERO_ASSEMBLED_DATASET_DIR to a directory "
        "containing `{suite}_task{id}_*_demo.hdf5` (e.g. ./datasets/demos). "
        f"Tried: {candidates}"
    )
    if require:
        raise FileNotFoundError(msg)
    warnings.warn(msg + " Privileged diffs will stay zeros; actor dims unchanged.", stacklevel=2)
    return None


def build_env_task_assignments(bindings: Sequence[TaskBinding], num_envs: int) -> list[str]:
    """Map env index → assignment key with harvest ``env i → task i % n`` layout."""
    n_tasks = len(bindings)
    if n_tasks == 0:
        raise ValueError("dgpo_task_bindings is empty.")
    return [
        harvest_task_to_assignment_key(bindings[i % n_tasks].name, bindings[i % n_tasks].suite) for i in range(num_envs)
    ]


def build_assignment_workspace_shifts(
    bindings: Sequence[TaskBinding],
) -> dict[str, tuple[float, float, float]]:
    """Map assignment key (e.g. ``libero_object::0``) → harvest workspace shift [m]."""
    out: dict[str, tuple[float, float, float]] = {}
    for binding in bindings:
        key = harvest_task_to_assignment_key(binding.name, binding.suite)
        sx, sy, sz = binding.workspace_shift
        out[key] = (float(sx), float(sy), float(sz))
    return out


@dataclass
class DgpoDemoTaskConfig:
    """Minimal libero_config stand-in for :class:`SourceLiberoCommandCfg`.

    Also exposed as ``env_cfg.libero_config`` so playground-compatible consumers
    (RFCL curriculum and ``demo_focus`` helpers) can read
    the per-env task assignments and the task ordering without changes.
    """

    env_task_assignments: list[str]
    assignment_workspace_shifts: dict[str, tuple[float, float, float]] | None = None
    task_sequence: tuple[str, ...] = ()
    """Unique assignment keys in env order (``env i -> task_sequence[i % n]``)."""
    full_task_sequence: tuple[str, ...] = ()
    """Same as ``task_sequence`` (this repo has no focus-subset concept)."""


# Short-lived alias for older call sites / tests.
CompatDemoTaskConfig = DgpoDemoTaskConfig


def _clone_trajectory_state_dict(state: TrajectoryState | None) -> TrajectoryState | None:
    """Deep-copy a nested ``initial_state`` dict, cloning every tensor leaf."""
    if state is None:
        return None
    cloned: TrajectoryState = {}
    for entity_type, entity_data in state.items():
        entity_copy: dict[str, dict[str, torch.Tensor]] = {}
        for entity_name, states in entity_data.items():
            entity_copy[entity_name] = {
                key: (value.clone() if torch.is_tensor(value) else value) for key, value in states.items()
            }
        cloned[entity_type] = entity_copy
    return cloned


def apply_workspace_shift_to_trajectory_state(
    state: TrajectoryState,
    shift: tuple[float, float, float],
) -> TrajectoryState:
    """Add harvest workspace translation to every ``root_pose`` in a demo state.

    Assembled LIBERO demos store poses in the original per-task robot-base frame.
    Harvest scenes rigidly translate fixtures/objects by
    ``ROBOT_BASE_KITCHEN - task_robot_base`` so the shared Franka stays fixed.
    Demo ``initial_state`` / pose-buffer targets must receive the same shift,
    otherwise objects are written near the unshifted floor (z≈0) and appear to
    fall through the raised table/fixture.

    Args:
        state: Nested ``initial_state`` dict (mutated in place and returned).
        shift: Translation ``(dx, dy, dz)`` [m] from :attr:`TaskBinding.workspace_shift`.

    Returns:
        The same ``state`` after shifting all ``root_pose`` position components.
    """
    dx, dy, dz = (float(shift[0]), float(shift[1]), float(shift[2]))
    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        return state
    for entity_type in ("rigid_object", "articulation"):
        entities = state.get(entity_type)
        if not isinstance(entities, dict):
            continue
        for asset_state in entities.values():
            if not isinstance(asset_state, dict):
                continue
            root_pose = asset_state.get("root_pose")
            if root_pose is None or not torch.is_tensor(root_pose) or root_pose.shape[-1] < 3:
                continue
            pose = root_pose.clone()
            delta = torch.tensor([dx, dy, dz], device=pose.device, dtype=pose.dtype)
            pose[..., :3] = pose[..., :3] + delta
            asset_state["root_pose"] = pose
    return state


def _decode_assignment(assignment: Any) -> tuple[str, int]:
    """Decode a ``(suite, task_id)`` tuple or ``suite::id`` string into ``(suite, task_id)``."""
    if isinstance(assignment, tuple) and len(assignment) == 2:
        return str(assignment[0]), int(assignment[1])
    if isinstance(assignment, str) and "::" in assignment:
        suite_name, _, task_id_str = assignment.rpartition("::")
        return suite_name, int(task_id_str)
    raise ValueError(f"Unsupported Libero assignment format: {assignment!r}")


class _UnifiedLiberoCommandBackend:
    """One trajectory bank, five semantic command views."""

    def __init__(self, cfg: SourceLiberoCommandCfg, env: ManagerBasedEnv):
        """Allocate per-env command / cursor buffers, load all demo HDF5 banks, and
        resample + emit the first command row for every env."""
        self.cfg = cfg
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self._command_dim = cfg.command_dim

        self._traj_banks: dict[str, torch.Tensor] = {}
        self._traj_lengths: torch.Tensor | None = None
        self._assignment_ids: torch.Tensor | None = None
        self._assignment_lookup: dict[int, tuple[str, int]] = {}
        self._assignment_traj_indices: dict[int, torch.Tensor] = {}
        self._assignment_seq_cursor: torch.Tensor | None = None
        self._traj_initial_states: list[TrajectoryState | None] = []
        # assignment id -> {(entity_type, name): (x_lo, x_hi, y_lo, y_hi)}; see
        # _build_initial_state_xy_ranges. Empty unless randomize_initial_state_xy.
        self._assignment_xy_ranges: dict[int, dict[tuple[str, str], tuple[float, float, float, float]]] = {}
        # Filled by reserve_trajectories_for_envs (demo reset event) then consumed in _resample.
        self._reserved_traj_indices: dict[int, int] = {}
        self._reserved_start_steps: dict[int, int] = {}

        self._commands: dict[str, torch.Tensor] = {
            key: torch.zeros(self.num_envs, self._command_dim, device=self.device)
            for key in UNIFIED_LIBERO_SEMANTIC_KEYS
        }
        self._active_traj = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._traj_len_per_env = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._frame_counters = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_reset_start_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics: dict[str, torch.Tensor] = {}
        if getattr(cfg, "track_metrics", True):
            self.metrics["progress"] = torch.zeros(self.num_envs, device=self.device)

        self._load_trajectories()
        self._resample(torch.arange(self.num_envs, device=self.device, dtype=torch.long))
        self._update_command()

    def _env_ids_to_tensor(self, env_ids: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        """Normalise tensor / slice / sequence env ids to a long tensor on this device."""
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            start, stop, step = env_ids.indices(self.num_envs)
            if stop <= start:
                return torch.empty(0, dtype=torch.long, device=self.device)
            return torch.arange(start, stop, step, dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _extract_tensor(self, data: dict, keys: list[str]) -> torch.Tensor:
        """Walk a nested episode dict along ``keys`` and return the tensor leaf (raise if missing)."""
        current: Any = data
        for key in keys:
            if key not in current:
                raise KeyError(f"Key '{key}' not found in demo episode.")
            current = current[key]
        if not isinstance(current, torch.Tensor):
            raise TypeError(f"Dataset entry for {'/'.join(keys)!r} is not a torch.Tensor.")
        return current

    def _fit_command_dim(self, traj: torch.Tensor) -> torch.Tensor:
        """Pad or truncate a ``(T, D)`` trajectory's feature dim to ``command_dim`` (zero-padded)."""
        if traj.ndim != 2:
            raise ValueError(f"Expected 2D trajectory, got shape {tuple(traj.shape)}.")
        if traj.shape[1] == self._command_dim:
            return traj
        fitted = torch.zeros(traj.shape[0], self._command_dim, dtype=traj.dtype, device=traj.device)
        copy_dim = min(int(traj.shape[1]), self._command_dim)
        fitted[:, :copy_dim] = traj[:, :copy_dim]
        return fitted

    def _load_dataset_multi_key(
        self, dataset_path: str
    ) -> tuple[dict[str, list[torch.Tensor]], list[TrajectoryState | None]]:
        """Load all episodes of one task HDF5 into per-semantic-key trajectory lists + initial states.

        Legacy WXYZ ``obs/ee_states`` are converted to sim XYZW at load; honours
        ``max_demos_per_task``.
        """
        from isaaclab.utils.datasets import HDF5DatasetFileHandler

        handler = HDF5DatasetFileHandler()
        handler.open(dataset_path)
        # Legacy demos (format_version < 1) store quats as WXYZ. HDF5DatasetFileHandler
        # already rewrites initial_state root_pose → XYZW; obs/ee_states still needs
        # an explicit convert via demo_quat_order so privileged EE diffs / OSC stay XYZW.
        legacy_quat = handler.is_legacy_quaternion_format()
        demo_quat_order: QuatOrder = getattr(self.cfg, "demo_quat_order", DEFAULT_DEMO_QUAT_ORDER)
        result: dict[str, list[torch.Tensor]] = {k: [] for k in UNIFIED_LIBERO_SEMANTIC_KEYS}
        initial_states: list[TrajectoryState | None] = []
        max_demos = getattr(self.cfg, "max_demos_per_task", None)
        try:
            for episode_name in sorted(handler.get_episode_names()):
                if max_demos is not None and len(initial_states) >= int(max_demos):
                    break
                episode = handler.load_episode(episode_name, self.device)
                if episode is None:
                    continue
                for key, hdf5_key in zip(UNIFIED_LIBERO_SEMANTIC_KEYS, UNIFIED_LIBERO_COMMAND_KEYS):
                    traj = self._extract_tensor(episode.data, hdf5_key.split("/"))
                    traj = self._fit_command_dim(traj.to(self.device).float())
                    if key == "ee_pose" and legacy_quat and traj.shape[-1] >= 7:
                        traj = pose7_to_sim(traj, from_order=demo_quat_order)
                    result[key].append(traj)
                init_state = episode.get_initial_state() if "initial_state" in episode.data else None
                initial_states.append(_clone_trajectory_state_dict(init_state) if init_state is not None else None)
        finally:
            handler.close()
        return result, initial_states

    def _pack_multi_banks(
        self, per_key_lists: dict[str, list[torch.Tensor]]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Zero-pad and stack trajectories into ``(n_traj, max_T, 7)`` banks per semantic key.

        Returns ``(banks, lengths)`` where ``lengths`` is ``(n_traj,)`` per-trajectory
        time lengths.
        """
        keys = UNIFIED_LIBERO_SEMANTIC_KEYS
        n = len(per_key_lists[keys[0]])
        for key in keys:
            if len(per_key_lists[key]) != n:
                raise ValueError(f"Inconsistent trajectory counts for key '{key}'.")
        max_len = max((t.shape[0] for k in keys for t in per_key_lists[k]), default=0)
        if max_len <= 0:
            raise ValueError("All loaded trajectories are empty.")
        lengths = torch.zeros(n, dtype=torch.long, device=self.device)
        for idx in range(n):
            lengths[idx] = max(int(per_key_lists[k][idx].shape[0]) for k in keys)
        banks: dict[str, torch.Tensor] = {}
        for key in keys:
            bank = torch.zeros(n, max_len, self._command_dim, device=self.device)
            for idx, traj in enumerate(per_key_lists[key]):
                length = int(traj.shape[0])
                if length > 0:
                    bank[idx, :length] = traj
            banks[key] = bank
        return banks, lengths

    def _resolve_dataset_path(self, assignment: tuple[str, int]) -> str:
        """Locate the task's HDF5 under ``datasets_root`` (``{suite}_task{id}_*.hdf5``).

        Prefers ``*_demo.hdf5`` on multiple matches; raises when none exist.
        """
        suite_name, task_id = assignment
        pattern = os.path.join(self.cfg.datasets_root, f"{suite_name}_task{task_id}_*.hdf5")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"Could not find Libero demo for {suite_name} task {task_id} under '{self.cfg.datasets_root}' "
                f"(pattern: {pattern})."
            )
        if len(matches) == 1:
            return matches[0]
        # Prefer filenames ending in _demo.hdf5 when multiple matches exist.
        demo_matches = [m for m in matches if os.path.basename(m).endswith("_demo.hdf5")]
        if len(demo_matches) == 1:
            return demo_matches[0]
        if demo_matches:
            return demo_matches[0]
        raise FileNotFoundError(f"Multiple demos matched '{pattern}': {matches}")

    def _load_trajectories(self):
        """Load demos for every unique assignment into the shared trajectory banks.

        Resolves per-env assignments (re-tiling the ``env i -> task i % n`` cycle when
        ``--num_envs`` differs from the baked count), loads each unique task HDF5,
        applies harvest workspace shifts to the initial states, and packs the banks.
        """
        libero_cfg = getattr(self.cfg, "libero_config", None)
        if libero_cfg is None or not hasattr(libero_cfg, "env_task_assignments"):
            raise ValueError("libero_config with env_task_assignments is required.")
        raw_assignments = list(libero_cfg.env_task_assignments)
        if not raw_assignments:
            raise ValueError("libero_config.env_task_assignments is empty.")
        if len(raw_assignments) != self.num_envs:
            # env_task_assignments is baked from the config's default num_envs at
            # construction time, before the CLI `--num_envs` override is applied. Re-tile
            # the deterministic ``env i -> task (i % n_tasks)`` cycle to the actual env
            # count so any --num_envs works (mirrors build_env_task_assignments()).
            unique_cycle = list(dict.fromkeys(raw_assignments))
            raw_assignments = [unique_cycle[i % len(unique_cycle)] for i in range(self.num_envs)]
        assignments = [_decode_assignment(a) for a in raw_assignments]
        if not os.path.isdir(self.cfg.datasets_root):
            raise FileNotFoundError(f"Datasets root '{self.cfg.datasets_root}' does not exist.")

        assignment_to_id: dict[tuple[str, int], int] = {}
        unique_assignments: list[tuple[str, int]] = []
        assignment_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for env_idx, assignment in enumerate(assignments):
            if assignment not in assignment_to_id:
                assignment_to_id[assignment] = len(unique_assignments)
                unique_assignments.append(assignment)
            assignment_ids[env_idx] = assignment_to_id[assignment]

        self._assignment_ids = assignment_ids
        self._assignment_seq_cursor = torch.zeros(len(unique_assignments), dtype=torch.long, device=self.device)
        self._assignment_traj_indices = {}
        self._assignment_lookup = {}
        all_banks_per_key: dict[str, list[torch.Tensor]] = {k: [] for k in UNIFIED_LIBERO_SEMANTIC_KEYS}
        self._traj_initial_states = []
        cursor = 0

        for assignment in unique_assignments:
            dataset_path = self._resolve_dataset_path(assignment)
            per_key_lists, init_states = self._load_dataset_multi_key(dataset_path)
            if not per_key_lists[UNIFIED_LIBERO_SEMANTIC_KEYS[0]]:
                raise RuntimeError(f"No trajectories in '{dataset_path}'.")
            n_traj = len(per_key_lists[UNIFIED_LIBERO_SEMANTIC_KEYS[0]])
            indices = torch.arange(cursor, cursor + n_traj, dtype=torch.long, device=self.device)
            aid = assignment_to_id[assignment]
            self._assignment_traj_indices[aid] = indices
            self._assignment_lookup[aid] = assignment
            cursor += n_traj
            for key in UNIFIED_LIBERO_SEMANTIC_KEYS:
                all_banks_per_key[key].extend(per_key_lists[key])
            # Align demo poses with harvest kitchen-base workspace normalisation.
            assignment_key = f"{assignment[0]}::{assignment[1]}"
            shift_map = getattr(libero_cfg, "assignment_workspace_shifts", None) or {}
            shift = shift_map.get(assignment_key, (0.0, 0.0, 0.0))
            for state in init_states:
                if state is not None:
                    apply_workspace_shift_to_trajectory_state(state, shift)
            self._traj_initial_states.extend(init_states)

        self._traj_banks, self._traj_lengths = self._pack_multi_banks(all_banks_per_key)
        logger.info(
            "Loaded %d demo trajectories for %d unique assignments from %s",
            int(self._traj_lengths.shape[0]),
            len(unique_assignments),
            self.cfg.datasets_root,
        )
        self._build_initial_state_xy_ranges()

    def _build_initial_state_xy_ranges(self) -> None:
        """Per assignment, bound each entity's reset XY by its spread over the demos.

        Fills ``self._assignment_xy_ranges[aid][(entity_type, name)] = (x_lo, x_hi,
        y_lo, y_hi)``, read at reset by :meth:`_randomize_state_xy`. Entities whose
        spread is below ``_XY_RANGE_EPS`` on both axes are dropped, so fixed fixtures
        contribute nothing and the reset write is a no-op for them. The robot is
        excluded: its reset pose comes from the demo (plus ``ROBOT_INIT_NOISE_STD``).

        Ranges are read off ``_traj_initial_states`` *after* the workspace shift, so
        they are already in the harvest frame the reset event writes into.
        """
        self._assignment_xy_ranges = {}
        if not getattr(self.cfg, "randomize_initial_state_xy", False):
            return
        scale = float(getattr(self.cfg, "randomize_initial_state_xy_scale", 1.0))
        if scale <= 0.0:
            raise ValueError(f"randomize_initial_state_xy_scale must be > 0, got {scale}.")
        for aid, indices in self._assignment_traj_indices.items():
            # (entity_type, name) -> [min_x, max_x, min_y, max_y]
            bounds: dict[tuple[str, str], list[float]] = {}
            for ti in indices.tolist():
                state = self._traj_initial_states[int(ti)]
                if state is None:
                    continue
                for entity_type in ("rigid_object", "articulation"):
                    entities = state.get(entity_type)
                    if not isinstance(entities, dict):
                        continue
                    for name, asset_state in entities.items():
                        if entity_type == "articulation" and name == "robot":
                            continue
                        root_pose = asset_state.get("root_pose") if isinstance(asset_state, dict) else None
                        if root_pose is None or not torch.is_tensor(root_pose) or root_pose.shape[-1] < 2:
                            continue
                        # frame 0 == the demo's actual reset pose (later frames are the
                        # recorded trajectory, which reserve_* slices only for RFCL).
                        flat = root_pose.reshape(-1, root_pose.shape[-1])
                        x = float(flat[0, 0])
                        y = float(flat[0, 1])
                        cur = bounds.get((entity_type, name))
                        if cur is None:
                            bounds[(entity_type, name)] = [x, x, y, y]
                        else:
                            cur[0] = min(cur[0], x)
                            cur[1] = max(cur[1], x)
                            cur[2] = min(cur[2], y)
                            cur[3] = max(cur[3], y)
            kept = {
                key: _scale_range(lo_x, hi_x, lo_y, hi_y, scale)
                for key, (lo_x, hi_x, lo_y, hi_y) in bounds.items()
                if (hi_x - lo_x) > _XY_RANGE_EPS or (hi_y - lo_y) > _XY_RANGE_EPS
            }
            self._assignment_xy_ranges[aid] = kept
        movable = sum(len(v) for v in self._assignment_xy_ranges.values())
        logger.info(
            "Demo-derived reset XY ranges: %d movable entities over %d assignments (box scale %.2fx).",
            movable,
            len(self._assignment_xy_ranges),
            scale,
        )

    def _randomize_state_xy(self, state: TrajectoryState | None, aid: int) -> TrajectoryState | None:
        """Redraw each entity's XY uniformly inside its demo-derived range, in place."""
        if state is None:
            return None
        ranges = self._assignment_xy_ranges.get(int(aid))
        if not ranges:
            return state
        for (entity_type, name), (lo_x, hi_x, lo_y, hi_y) in ranges.items():
            asset_state = state.get(entity_type, {}).get(name)
            if not isinstance(asset_state, dict):
                continue
            root_pose = asset_state.get("root_pose")
            if root_pose is None or not torch.is_tensor(root_pose) or root_pose.shape[-1] < 2:
                continue
            draw = torch.rand(2, device=root_pose.device, dtype=root_pose.dtype)
            root_pose[..., 0] = lo_x + draw[0] * (hi_x - lo_x)
            root_pose[..., 1] = lo_y + draw[1] * (hi_y - lo_y)
        return state

    def _draw_sample_indices(self, env_ids_tensor: torch.Tensor) -> torch.Tensor:
        """Pick a demo per env from its assignment's pool: random or sequential (``sample_mode``)."""
        if self._assignment_ids is None or self._assignment_seq_cursor is None:
            raise RuntimeError("Assignment metadata not initialized.")
        assignment_ids = self._assignment_ids[env_ids_tensor]
        sampled = torch.zeros(env_ids_tensor.shape[0], dtype=torch.long, device=self.device)
        sequential = getattr(self.cfg, "sample_mode", "random") == "sequential"
        # One RNG draw / cursor update per unique assignment rather than per env: the
        # per-env loop issued O(num_envs) tiny CUDA ops plus a sync per env, which is
        # the dominant reset cost at high env counts.
        for aid_tensor in torch.unique(assignment_ids):
            aid = int(aid_tensor.item())
            candidates = self._assignment_traj_indices.get(aid)
            if candidates is None or candidates.numel() == 0:
                raise RuntimeError(f"No trajectories for assignment id {aid}.")
            positions = torch.nonzero(assignment_ids == aid_tensor, as_tuple=False).flatten()
            count = int(positions.numel())
            pool = int(candidates.numel())
            if sequential:
                # nonzero() yields ascending positions, i.e. the order the per-env loop
                # served them, so the k-th env of this group still takes cursor cur + k.
                cur = int(self._assignment_seq_cursor[aid].item())
                offsets = (torch.arange(count, device=self.device, dtype=torch.long) + cur) % pool
                sampled[positions] = candidates[offsets]
                self._assignment_seq_cursor[aid] = (cur + count) % pool
            else:
                sampled[positions] = candidates[torch.randint(0, pool, (count,), device=self.device)]
        return sampled

    def _apply_sampled_indices(self, env_ids_tensor: torch.Tensor, traj_indices: torch.Tensor):
        """Bind envs to the sampled trajectories: reset cursors to 0 and emit each demo's first row."""
        if env_ids_tensor.numel() == 0:
            return
        self._active_traj[env_ids_tensor] = traj_indices
        self._traj_len_per_env[env_ids_tensor] = self._traj_lengths[traj_indices]
        self._frame_counters[env_ids_tensor] = 0
        self._last_reset_start_steps[env_ids_tensor] = 0
        for key in UNIFIED_LIBERO_SEMANTIC_KEYS:
            self._commands[key][env_ids_tensor] = self._traj_banks[key][traj_indices, 0]

    def _apply_start_steps(self, env_ids_tensor: torch.Tensor, traj_indices: torch.Tensor, start_steps: torch.Tensor):
        """Move envs' frame cursors to the given start steps (clamped to demo length) and
        emit the corresponding command rows."""
        if env_ids_tensor.numel() == 0:
            return
        start_steps = start_steps.to(device=self.device, dtype=torch.long)
        traj_lengths = self._traj_lengths[traj_indices]
        max_valid = torch.clamp(traj_lengths - 1, min=0)
        clamped = torch.minimum(torch.clamp(start_steps, min=0), max_valid)
        self._frame_counters[env_ids_tensor] = clamped
        self._last_reset_start_steps[env_ids_tensor] = clamped
        for key in UNIFIED_LIBERO_SEMANTIC_KEYS:
            self._commands[key][env_ids_tensor] = self._traj_banks[key][traj_indices, clamped]

    def _update_command(self):
        """Write each env's current demo row from every semantic bank, then advance the
        frame cursor (clamped so the final row repeats past the demo end)."""
        traj_idx = self._active_traj
        traj_len = self._traj_len_per_env
        clamped_len = torch.clamp(traj_len, min=1)
        effective_idx = torch.minimum(self._frame_counters, clamped_len - 1)
        for key in UNIFIED_LIBERO_SEMANTIC_KEYS:
            self._commands[key][:] = self._traj_banks[key][traj_idx, effective_idx]
        self._frame_counters += 1
        self._frame_counters = torch.minimum(self._frame_counters, clamped_len)

    def get_per_env_progress(self) -> torch.Tensor:
        """Per-env demo cursor normalized to [0, 1] — ``frame / (T - 1)``, ``(num_envs,)``.

        The same quantity ``metrics["progress"]`` carries, exposed as a method so
        the ``demo_phase`` observation can read it without depending on metric
        tracking being enabled (``cfg.track_metrics``).
        """
        total = torch.clamp(self._traj_len_per_env - 1, min=1)
        effective = torch.minimum(self._frame_counters, total)
        return effective.float() / total.float()

    def get_per_env_start_progress(self) -> torch.Tensor:
        """Per-env normalized cursor the current episode was RESET to, ``(num_envs,)``.

        Zero for a plain reset; non-zero under a reserved start step or an RFCL
        reverse curriculum. Known at reset time, so a policy consuming it is still
        deployable.
        """
        total = torch.clamp(self._traj_len_per_env - 1, min=1)
        start = torch.minimum(self._last_reset_start_steps, total)
        return start.float() / total.float()

    def _update_metrics(self):
        """Update the ``progress`` metric = frame / (T - 1) per env, in [0, 1]."""
        if not getattr(self.cfg, "track_metrics", True):
            return
        self.metrics["progress"] = self.get_per_env_progress()

    def _resample(self, env_ids_tensor: torch.Tensor):
        """Assign a demo to each resetting env.

        Honors trajectory reservations made by the demo reset event so scene init and
        teacher replay use the SAME demo (including its reserved start step); envs
        without a reservation draw fresh via :meth:`_draw_sample_indices`.
        """
        if env_ids_tensor.numel() == 0:
            return
        # Prefer trajectories reserved by the demo-reset event (same traj as initial_state).
        forced_envs: list[int] = []
        forced_indices: list[int] = []
        remaining: list[int] = []
        for e in env_ids_tensor.tolist():
            ei = int(e)
            ridx = self._reserved_traj_indices.pop(ei, None)
            if ridx is not None:
                forced_envs.append(ei)
                forced_indices.append(int(ridx))
            else:
                remaining.append(ei)
        if forced_envs:
            fe = torch.tensor(forced_envs, device=self.device, dtype=torch.long)
            fi = torch.tensor(forced_indices, device=self.device, dtype=torch.long)
            self._apply_sampled_indices(fe, fi)
            steps = torch.tensor(
                [self._reserved_start_steps.pop(e, 0) for e in forced_envs],
                device=self.device,
                dtype=torch.long,
            )
            self._apply_start_steps(fe, fi, steps)
        if remaining:
            rem = torch.tensor(remaining, device=self.device, dtype=torch.long)
            sampled = self._draw_sample_indices(rem)
            self._apply_sampled_indices(rem, sampled)

    def reserve_trajectories_for_envs_rfcl(
        self, env_ids: Sequence[int] | torch.Tensor | slice, term
    ) -> dict[int, TrajectoryState | None]:
        """RFCL-curriculum reservation: demo index + start step come from the scheduler.

        Called instead of :meth:`reserve_trajectories_for_envs` when an
        ``RFCLEnvWrapper`` has registered its scheduler on the command term.
        Every per-episode reset queries the curriculum: the scheduler picks a
        demo (adaptive density) and a start pointer (reverse/forward cursor);
        the backend maps that to its OWN trajectory bank (same file/episode
        ordering: both load ``{suite}_task{id}_*.hdf5`` sorted, episodes sorted)
        and reserves it with the matching start step, returning the demo state
        sliced at the pointer so scene physics, teacher replay, and curriculum
        cursor stay coherent. The wrapper's per-env bookkeeping arrays
        (``_rfcl_env_demo_indices`` / ``_rfcl_env_phase1_ptrs`` /
        ``_rfcl_env_at_frontier``) are updated in place.
        """
        env_ids_tensor = self._env_ids_to_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return {}
        scheduler = term._rfcl_scheduler
        reservations: dict[int, TrajectoryState | None] = {}
        for e in env_ids_tensor.tolist():
            ei = int(e)
            tid = int(term._rfcl_env_task_ids[ei])
            didx = int(scheduler.sample_demo_idx(tid))
            _state, ptr, at_frontier = scheduler.get_initial_state_and_ptr(tid, didx)
            aid = int(self._assignment_ids[ei].item())
            candidates = self._assignment_traj_indices.get(aid)
            if candidates is None or candidates.numel() == 0:
                reservations[ei] = None
                continue
            ti = int(candidates[didx % candidates.numel()].item())
            traj_len = int(self._traj_lengths[ti].item())
            ptr = max(0, min(int(ptr), traj_len - 1))
            self._reserved_traj_indices[ei] = ti
            self._reserved_start_steps[ei] = ptr
            term._rfcl_env_demo_indices[ei] = didx
            term._rfcl_env_phase1_ptrs[ei] = ptr
            term._rfcl_env_at_frontier[ei] = bool(at_frontier)
            state: TrajectoryState | None = None
            if 0 <= ti < len(self._traj_initial_states) and self._traj_initial_states[ti] is not None:
                state = _clone_trajectory_state_dict(self._traj_initial_states[ti])
                state = self._slice_state_at_timestep(state, ptr)
                if ptr == 0:
                    state = self._randomize_state_xy(state, aid)
            reservations[ei] = state
        return reservations

    def reserve_trajectories_for_envs(
        self, env_ids: Sequence[int] | torch.Tensor | slice
    ) -> dict[int, TrajectoryState | None]:
        """Sample traj indices and return sliced ``initial_state`` for demo reset.

        Called from the reset event **before** :meth:`reset` / :meth:`_resample` so
        the command stack and scene share the same trajectory.
        """
        env_ids_tensor = self._env_ids_to_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return {}
        sampled = self._draw_sample_indices(env_ids_tensor)
        reservations: dict[int, TrajectoryState | None] = {}
        for e, traj_idx in zip(env_ids_tensor.tolist(), sampled.tolist()):
            ei = int(e)
            ti = int(traj_idx)
            self._reserved_traj_indices[ei] = ti
            timestep = 0
            state: TrajectoryState | None = None
            if 0 <= ti < len(self._traj_initial_states) and self._traj_initial_states[ti] is not None:
                state = _clone_trajectory_state_dict(self._traj_initial_states[ti])
                time_len = self._infer_state_time_length(state) if state is not None else None
                if (
                    state is not None
                    and time_len is not None
                    and time_len > 1
                    and getattr(self.cfg, "randomize_initial_state_timestep", False)
                ):
                    timestep = int(torch.randint(0, time_len, (), device=self.device).item())
                    state = self._slice_state_at_timestep(state, timestep)
                elif state is not None and time_len is not None and time_len > 0:
                    state = self._slice_state_at_timestep(state, 0)
            if timestep == 0:
                # Only at t=0: past that the state is mid-trajectory, where an object's
                # pose is whatever the demo motion put there, not a placement to redraw.
                state = self._randomize_state_xy(state, int(self._assignment_ids[ei].item()))
            self._reserved_start_steps[ei] = timestep
            reservations[ei] = state
        return reservations

    def compute(self, dt: float):  # noqa: ARG002
        """Refresh the progress metric, then emit the next demo row and advance cursors."""
        self._update_metrics()
        self._update_command()

    def reset(self, env_ids: Sequence[int] | None):
        """Resample demos for the given envs (all envs when ``None``)."""
        if env_ids is None:
            env_ids = slice(None)
        env_ids_tensor = self._env_ids_to_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return
        self._resample(env_ids_tensor)

    def get_command(self, semantic_key: str) -> torch.Tensor:
        """Return the ``(num_envs, command_dim)`` buffer for one semantic view."""
        return self._commands[semantic_key]

    def _infer_state_time_length(self, state: TrajectoryState) -> int | None:
        """Infer a state dict's time length as the min leading dim (>1) over tensor leaves."""
        lengths: list[int] = []

        def visit(v):
            if isinstance(v, dict):
                for x in v.values():
                    visit(x)
            elif torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] > 1:
                lengths.append(int(v.shape[0]))

        visit(state)
        return min(lengths) if lengths else None

    def _slice_state_at_timestep(self, state: TrajectoryState, timestep: int) -> TrajectoryState:
        """Slice every time-varying tensor leaf to a single-frame ``[timestep:timestep+1]`` copy."""
        def slice_val(v):
            if isinstance(v, dict):
                return {k: slice_val(x) for k, x in v.items()}
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] > 1:
                return v[timestep : timestep + 1].clone()
            return v

        return slice_val(state)  # type: ignore[return-value]


def _get_unified_backend(env: ManagerBasedEnv, backend_id: str, cfg: SourceLiberoCommandCfg):
    """Return the env-cached backend for ``backend_id``, constructing it on first use."""
    backends = getattr(env, "_unified_libero_backends", None)
    if backends is None:
        backends = {}
        setattr(env, "_unified_libero_backends", backends)
    if backend_id not in backends:
        backends[backend_id] = _UnifiedLiberoCommandBackend(cfg, env)
    return backends[backend_id]


class SourceLiberoCommand(CommandTerm):
    """Single command term exposing demo views via :meth:`get_command_view`."""

    cfg: SourceLiberoCommandCfg

    def __init__(self, cfg: SourceLiberoCommandCfg, env: ManagerBasedEnv):
        """Attach to the shared backend (created on first use) and share its metrics dict."""
        super().__init__(cfg, env)
        self._backend = _get_unified_backend(env, cfg.backend_id, cfg)
        self.metrics = self._backend.metrics
        self._traj_pose_buffer_cache: dict[str, torch.Tensor] | None = None
        self._traj_pose_buffer_cache_key: tuple[object, int, int] | None = None

    @property
    def command(self) -> torch.Tensor:
        """Return the primary semantic view (default ``source_action``), ``(num_envs, 7)``."""
        return self._backend.get_command(self.cfg.primary_semantic_key)

    def get_command_view(self, semantic_key: str) -> torch.Tensor:
        """Return the ``(num_envs, command_dim)`` buffer for the named semantic view."""
        return self._backend.get_command(semantic_key)

    def get_per_env_progress(self) -> torch.Tensor:
        """Delegate to the shared backend's normalized demo cursor (the ``demo_phase`` input)."""
        return self._backend.get_per_env_progress()

    def get_per_env_start_progress(self) -> torch.Tensor:
        """Delegate to the shared backend's normalized reset cursor."""
        return self._backend.get_per_env_start_progress()

    @property
    def _frame_counters(self) -> torch.Tensor:
        """Delegate to the shared backend's per-env demo frame counters."""
        return self._backend._frame_counters

    @property
    def _active_traj(self) -> torch.Tensor:
        """Delegate to the shared backend's per-env active trajectory indices."""
        return self._backend._active_traj

    @property
    def _traj_initial_states(self) -> list[TrajectoryState | None]:
        """Delegate to the shared backend's per-trajectory ``initial_state`` dicts."""
        return self._backend._traj_initial_states

    @property
    def _traj_lengths(self) -> torch.Tensor | None:
        """Delegate to the shared backend's per-trajectory time lengths."""
        return self._backend._traj_lengths

    def _infer_state_time_length(self, state: TrajectoryState) -> int | None:
        """Delegate time-length inference to the shared backend."""
        return self._backend._infer_state_time_length(state)

    def _slice_state_at_timestep(self, state: TrajectoryState, timestep: int) -> TrajectoryState:
        """Delegate single-frame state slicing to the shared backend."""
        return self._backend._slice_state_at_timestep(state, timestep)

    def reserve_trajectories_for_envs(
        self, env_ids: Sequence[int] | torch.Tensor | slice
    ) -> dict[int, TrajectoryState | None]:
        """Delegate to the unified backend (used by demo-reset events).

        When an ``RFCLEnvWrapper`` has registered its curriculum scheduler on
        this term (``_rfcl_scheduler``), the reservation is driven by the
        curriculum instead of random/sequential sampling — so every
        per-episode reset injects the reverse/forward-curriculum start state.
        """
        if getattr(self, "_rfcl_scheduler", None) is not None:
            return self._backend.reserve_trajectories_for_envs_rfcl(env_ids, self)
        return self._backend.reserve_trajectories_for_envs(env_ids)

    def _update_metrics(self):
        """Delegate the progress-metric refresh to the shared backend."""
        self._backend._update_metrics()

    def _resample_command(self, env_ids: Sequence[int]):  # noqa: ARG002
        """No-op hook; :meth:`reset` drives the backend resample directly."""
        pass

    def _update_command(self):
        """No-op hook; :meth:`compute` drives the backend command update directly."""
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Run the base reset, then resample the backend's demos for the given envs."""
        extras = super().reset(env_ids)
        self._backend.reset(env_ids)
        return extras

    def compute(self, dt: float):
        """Step the shared backend: refresh metrics, emit the next row, advance cursors."""
        self._backend.compute(dt)


@configclass
class SourceLiberoCommandCfg(CommandTermCfg):
    """Configuration for :class:`SourceLiberoCommand`."""

    class_type: type = SourceLiberoCommand

    semantic_keys: tuple[str, ...] = UNIFIED_LIBERO_SEMANTIC_KEYS
    primary_semantic_key: str = "source_action"
    backend_id: str = "libero_demo"
    datasets_root: str = ""
    libero_config: object | None = None
    command_dim: int = 7
    resampling_time_range: tuple[float, float] = (1e6, 1e6)
    randomize_initial_state_timestep: bool = False
    randomize_initial_state_xy: bool = False
    """Resample every object's reset XY uniformly from the demo-derived range.

    LIBERO's own demo collection draws each object's initial placement from a
    per-task region, so the 50 demos of a task already scatter each object over a
    box (typically ~5 cm per axis). Enabling this reconstructs that box --
    per (assignment, entity) min/max over the demos' ``initial_state`` frame 0 --
    and draws a fresh XY inside it at reset, instead of replaying whichever demo
    the env happens to have reserved.

    Z, orientation, and the robot are untouched, and entities whose demo range
    collapses to a point (fixed fixtures such as ``wine_rack_1``) stay put.

    This decouples the scene from the reserved demo, so the demo-tracking reward
    and the privileged object-target diffs no longer describe the scene in front
    of the policy. Intended for OOD *evaluation*; leave it off for
    ``LIBERO_REWARD_MODE=world_state_tracking_reward`` training.
    """
    randomize_initial_state_xy_scale: float = 1.0
    """Multiply each demo-derived reset box about its centre before sampling.

    ``1.0`` reproduces the demos' own placement region, which is *in* distribution:
    the demos already scatter each object across that box and every reset reserves
    a random one, so re-drawing inside it interpolates the training support rather
    than leaving it. Values above 1 push placements outside anything the demos
    cover, which is what an OOD init-state sweep needs (``2.0`` doubles each axis,
    so half the draws land beyond the demo range).

    Only entities that already move between demos are scaled -- a fixed fixture has
    a zero-width box, and scaling zero stays zero.
    """
    cache_pose_buffer_states: bool = True
    pose_buffer_cache_device: str | None = None
    """Device holding the demo object-pose cache. ``None`` = the sim device.

    The cache is read every step by the privileged ``ObjectTargetPoseDiff`` obs.
    Keeping it on the host costs a GPU->CPU->GPU round trip per step (measured
    134 us vs 22 us on an L20 at 1000 envs), so it belongs on the sim device.
    Residency is ``n_traj * max_T * n_slots * (7 + joint_dim) * 4`` bytes —
    ~0.7 GB for 2000 demos x T=500 x 26 slots. Set ``"cpu"`` to trade the
    per-step transfer back for host memory when that does not fit.
    """
    track_metrics: bool = True
    sample_mode: str = "random"
    max_demos_per_task: int | None = None
    """If set, load at most this many episodes from each task HDF5 file."""

    demo_quat_order: QuatOrder = DEFAULT_DEMO_QUAT_ORDER
    """Quaternion order of demo ``obs/ee_states`` before conversion to sim XYZW."""

    def __post_init__(self):
        """Validate required fields (``libero_config``, semantic keys, demo count, quat order)."""
        if self.libero_config is None:
            raise ValueError("libero_config must be provided for SourceLiberoCommandCfg.")
        if not self.semantic_keys:
            raise ValueError("semantic_keys must be non-empty.")
        if self.primary_semantic_key not in self.semantic_keys:
            raise ValueError(f"primary_semantic_key must be in semantic_keys, got {self.primary_semantic_key!r}.")
        if self.max_demos_per_task is not None and int(self.max_demos_per_task) <= 0:
            raise ValueError(f"max_demos_per_task must be positive, got {self.max_demos_per_task}.")
        if self.demo_quat_order not in ("wxyz", "xyzw"):
            raise ValueError(f"demo_quat_order must be 'wxyz' or 'xyzw', got {self.demo_quat_order!r}.")


@configclass
class DgpoCommandsCfg:
    """Commands cfg: the single demo term registers as ``libero_demo`` on the stock manager.

    Semantic views (``ee_pose`` / ``source_action`` / ...) are resolved via
    :mod:`isaaclab_contrib.tasks.manipulation.utils.commands`.
    """

    libero_demo: SourceLiberoCommandCfg | None = None


# Short-lived aliases.
CompatCommandsCfg = DgpoCommandsCfg


def make_dgpo_commands_cfg(
    bindings: Sequence[TaskBinding],
    num_envs: int,
    *,
    datasets_root: str | None = None,
    require_demos: bool | None = None,
    demo_quat_order: QuatOrder = DEFAULT_DEMO_QUAT_ORDER,
) -> DgpoCommandsCfg | None:
    """Build demo commands or return ``None`` when demos are unavailable.

    Returns:
        A filled :class:`DgpoCommandsCfg`, or ``None`` when demos are missing
        and not required (caller keeps privileged obs as zeros).
    """
    root = datasets_root if datasets_root is not None else resolve_libero_demos_root(require=require_demos)
    if root is None:
        return None
    eval_mode = settings.truthy("LIBERO_EVALUATION")
    # Object placement: redraw each object's XY from the range its demos span.
    # Deliberately not gated on eval_mode -- OOD evaluation is the point.
    random_init_xy = settings.truthy("LIBERO_RANDOM_INIT_STATE")
    # Start timestep: begin the episode part-way along the demo. Training-only
    # (it makes episodes strictly easier, so it would inflate an eval SR).
    random_init_timestep = settings.truthy("LIBERO_RANDOM_INIT_TIMESTEP")
    xy_scale = settings.number("LIBERO_RANDOM_INIT_XY_SCALE", 1.0)
    max_demos: int | None = settings.integer("LIBERO_COMPAT_MAX_DEMOS_PER_TASK")
    assignments = build_env_task_assignments(bindings, num_envs)
    shifts = build_assignment_workspace_shifts(bindings)
    unique_assignments = tuple(dict.fromkeys(assignments))
    cfg = DgpoCommandsCfg()
    cfg.libero_demo = SourceLiberoCommandCfg(
        libero_config=DgpoDemoTaskConfig(
            env_task_assignments=assignments,
            assignment_workspace_shifts=shifts,
            task_sequence=unique_assignments,
            full_task_sequence=unique_assignments,
        ),
        datasets_root=root,
        command_dim=7,
        resampling_time_range=(1e6, 1e6),
        track_metrics=True,
        randomize_initial_state_timestep=random_init_timestep and not eval_mode,
        randomize_initial_state_xy=random_init_xy,
        randomize_initial_state_xy_scale=xy_scale,
        cache_pose_buffer_states=True,
        # None = sim device; see SourceLiberoCommandCfg.pose_buffer_cache_device.
        pose_buffer_cache_device=None,
        sample_mode="random",
        max_demos_per_task=max_demos,
        demo_quat_order=demo_quat_order,
    )
    return cfg


make_compat_commands_cfg = make_dgpo_commands_cfg
