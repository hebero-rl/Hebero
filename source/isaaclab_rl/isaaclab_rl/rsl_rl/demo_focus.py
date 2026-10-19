"""Helpers for filtering Libero demo HDF5 files by LIBERO_FOCUS_TASKS.

When ``LIBERO_FOCUS_TASKS`` is set, ``libero_config.task_sequence`` becomes the
deduplicated focused subset (e.g. ``("libero_10::0", "libero_10::1")``) while
``full_task_sequence`` always stays the canonical 40-task list. The RFCL
curriculum must consume only files matching the focused subset so its reset
states belong to tasks the policy is training on.

Demo files follow the naming convention ``{suite_name}_task{task_id}_*.hdf5``
produced by ``preprocess_libero_demos.py`` and the original assembled demos.
"""
from __future__ import annotations

import os
import re
from typing import Iterable


_DEMO_BASENAME_RE = re.compile(r"^(?P<suite>[a-zA-Z][a-zA-Z0-9_]*?)_task(?P<task_id>\d+)_.*\.hdf5$")


def parse_demo_basename(basename: str) -> tuple[str, int] | None:
    """Parse ``{suite}_task{id}_*.hdf5`` → ``(suite, id)``; return None if no match."""
    m = _DEMO_BASENAME_RE.match(basename)
    if m is None:
        return None
    return m.group("suite"), int(m.group("task_id"))


def _decode_assignment(assignment) -> tuple[str, int] | None:
    """Decode ``"suite::id"`` or ``(suite, id)`` → ``(suite, int(id))``."""
    if isinstance(assignment, tuple) and len(assignment) == 2:
        suite, task_id = assignment
        return str(suite), int(task_id)
    if isinstance(assignment, str) and "::" in assignment:
        suite, _, task_id_str = assignment.rpartition("::")
        if suite and task_id_str:
            try:
                return suite, int(task_id_str)
            except ValueError:
                return None
    return None


def resolve_focus_task_set(env) -> frozenset[tuple[str, int]] | None:
    """Return the focused ``(suite, task_id)`` set, or None when no filtering is needed.

    Reads ``libero_config.task_sequence`` (focused subset) and
    ``libero_config.full_task_sequence`` (canonical 40) from the env cfg.

    Returns:
        - ``None`` when the env has no ``libero_config`` (non-Libero env).
        - ``None`` when ``task_sequence`` is empty / missing.
        - ``None`` when ``len(task_sequence) == len(full_task_sequence)`` —
          training on all canonical tasks, no filtering needed.
        - Otherwise, the deduplicated set of ``(suite, task_id)`` to keep.
    """
    env_cfg = getattr(getattr(env, "unwrapped", env), "cfg", None)
    libero_cfg = getattr(env_cfg, "libero_config", None) if env_cfg is not None else None
    if libero_cfg is None:
        return None

    focused = getattr(libero_cfg, "task_sequence", None)
    full = getattr(libero_cfg, "full_task_sequence", None)
    if not focused:
        return None
    if full is not None and len(focused) == len(full):
        return None  # no filtering needed

    keys: set[tuple[str, int]] = set()
    for assignment in focused:
        decoded = _decode_assignment(assignment)
        if decoded is not None:
            keys.add(decoded)
    return frozenset(keys) if keys else None


def filter_demo_paths(paths: Iterable[str], focus_set: frozenset[tuple[str, int]]) -> list[str]:
    """Return paths whose basenames match the focused ``(suite, task_id)`` set.

    Paths whose basename does not parse as ``{suite}_task{id}_*.hdf5`` are
    silently dropped — callers should validate the resulting count.
    """
    out: list[str] = []
    for p in paths:
        parsed = parse_demo_basename(os.path.basename(p))
        if parsed is None:
            continue
        if parsed in focus_set:
            out.append(p)
    return out


def resolve_task_sequence(env) -> tuple[tuple[str, int], ...] | None:
    """Return the env's ``task_sequence`` as a tuple of ``(suite, task_id)`` labels.

    Prefers ``libero_config.task_sequence`` (focused subset under
    ``LIBERO_FOCUS_TASKS``) over ``full_task_sequence`` (canonical 40), so the
    returned order matches the env's per-env task assignments.

    Returns ``None`` when no ``libero_config`` is reachable or the sequence is
    empty — callers should treat that as "do not reorder."
    """
    env_cfg = getattr(getattr(env, "unwrapped", env), "cfg", None)
    libero_cfg = getattr(env_cfg, "libero_config", None) if env_cfg is not None else None
    if libero_cfg is None:
        return None
    seq = getattr(libero_cfg, "task_sequence", None) or getattr(libero_cfg, "full_task_sequence", None)
    if not seq:
        return None
    out: list[tuple[str, int]] = []
    for entry in seq:
        decoded = _decode_assignment(entry)
        if decoded is not None:
            out.append(decoded)
    return tuple(out) if out else None


def sort_demo_paths_by_task_sequence(
    paths: Iterable[str],
    task_sequence: tuple[tuple[str, int], ...] | None,
    require_all_tasks: bool = False,
) -> list[str]:
    """Reorder demo HDF5 paths so element ``i`` corresponds to ``task_sequence[i]``.

    Why: ``sorted(glob(*.hdf5))`` gives **alphabetical** order
    (``libero_10 < libero_goal < libero_object < libero_spatial``), but the env's
    ``task_sequence`` is ``[libero_10, libero_object, libero_spatial, libero_goal]``.
    Downstream code that indexes loaded demos by env ``task_id`` then injects
    states from the wrong suite into the wrong env (silent, multi-task-only,
    very confusing).

    Behaviour:
      - ``task_sequence`` is None / empty → returns ``list(paths)`` unchanged.
      - All paths parseable → reorder by ``task_sequence``.  Duplicate labels
        raise ``ValueError``.  Missing task labels are:
          - ``require_all_tasks=True``: raises ``FileNotFoundError``.
          - ``require_all_tasks=False`` (default): skipped with a warning.
        Extra parseable demos whose
        label is **not** in ``task_sequence`` are dropped (they belong to
        tasks outside the focused set).
      - Some paths unparseable (custom-named single-task HDF5s) → keep
        original order, no reordering applied (warns once).
      - All paths unparseable → returns ``list(paths)`` unchanged.

    The returned order maps demo index → ``task_sequence`` index 1-to-1, so
    schedulers can use ``_task_states[i]`` directly with the env's task_id ``i``.
    """
    paths = list(paths)
    if not task_sequence:
        return paths

    label_to_path: dict[tuple[str, int], str] = {}
    n_unparseable = 0
    for p in paths:
        lab = parse_demo_basename(os.path.basename(p))
        if lab is None:
            n_unparseable += 1
            continue
        if lab in label_to_path:
            raise ValueError(
                f"sort_demo_paths_by_task_sequence: duplicate demo for {lab}:\n"
                f"  {label_to_path[lab]}\n  {p}"
            )
        label_to_path[lab] = p

    if not label_to_path:
        return paths  # nothing parseable → custom-named single-task workflow
    if n_unparseable > 0:
        import warnings as _w
        _w.warn(
            f"sort_demo_paths_by_task_sequence: {n_unparseable} of {len(paths)} demo "
            f"path(s) do not match the '{{suite}}_task{{id}}_*.hdf5' convention; "
            f"skipping task_sequence reorder.",
            stacklevel=2,
        )
        return paths

    ordered: list[str] = []
    missing: list[tuple[str, int]] = []
    for lab in task_sequence:
        path = label_to_path.pop(lab, None)
        if path is not None:
            ordered.append(path)
        else:
            missing.append(lab)
    if missing:
        head = ", ".join(f"{s}::{i}" for s, i in missing[:5])
        tail = "..." if len(missing) > 5 else ""
        msg = (
            f"sort_demo_paths_by_task_sequence: missing demo HDF5 for "
            f"{len(missing)} task(s): [{head}{tail}].  task_sequence has "
            f"{len(task_sequence)} entries but only "
            f"{len(task_sequence) - len(missing)} matched."
        )
        if require_all_tasks:
            raise FileNotFoundError(msg)
        import warnings as _w
        _w.warn(f"{msg} Skipping missing tasks and continuing with matched demos.", stacklevel=2)
    # Note: any label_to_path entries left are demos outside the focus set —
    # silently dropped (consistent with filter_demo_paths behaviour).
    return ordered
