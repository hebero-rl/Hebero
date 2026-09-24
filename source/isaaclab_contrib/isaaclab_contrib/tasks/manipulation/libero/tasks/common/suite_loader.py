# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data-driven loader for the ``libero_object`` and ``libero_10`` suites.

Unlike ``libero_spatial`` / ``libero_goal`` (whose 10 tasks all share one object
set), the ``libero_object`` and ``libero_10`` suites use *different objects per
task*.  Each task therefore becomes its own clone group, and its object layout is
read from the LIBERO per-task JSON configs (the single source of truth, also used
by the upstream benchmark) instead of being hand-written.

Source of truth
---------------
``$LIBERO_CONFIG_DIR/<suite>.json`` (see :func:`...assets.libero_config_dir`).
Each ``tasks[i]`` entry provides:

* ``objects``: ``name -> {type, scale, initial_region}`` (USD lives at
  ``<type>/<type>.usd`` under the asset root).
* ``regions``: ``region -> pose_range`` with ``x/y/z`` [m] and ``roll/pitch/yaw``
  [rad] min/max; the initial pose is the range midpoint.
* ``obj_of_interest`` / ``targets``: the manipulated object and its goal object.
* ``goals``: BDDL goal predicates; the first relational (``on`` / ``in``) goal
  supplies the ref/target objects and the ``xy_threshold`` / ``height_threshold``
  used for the geometric success proxy (see :mod:`...mdp.terminations`).
* ``fixtures``, ``workspace_name``, ``robot_base_pos``.

Workspace normalisation
-----------------------
Tasks span several workspaces (``floor``, ``living_room_table``, ``study_table``,
``kitchen_table``) whose authored robot base differs.  The shared LIBERO Franka is
fixed at the kitchen base :data:`ROBOT_BASE_KITCHEN`.  Every object and fixture
pose is rigidly translated by ``ROBOT_BASE_KITCHEN - task_robot_base`` so the
robot base coincides with the shared robot; this preserves each task's authored
robot-relative geometry exactly.  The same ``workspace_shift`` is stored on
:class:`TaskLayout` / harvest :class:`TaskBinding` and applied to demo
``initial_state`` poses when loading the DGPO trajectory bank (otherwise
floor/living-room demos write objects at z≈0 under the raised fixture).
"""

from __future__ import annotations

import functools
import json
import math
import os
from dataclasses import dataclass, field

from ...assets import libero_config_dir

# Shared Franka base pose (kitchen workspace); mirrors ``assets.LIBERO_FRANKA_PANDA_CFG``.
ROBOT_BASE_KITCHEN: tuple[float, float, float] = (-0.66, 0.0, 0.912)

# Internal suite name -> on-disk JSON filename.  The dataset ships the
# long-horizon suite as ``libero_10.json``; we expose it under the friendlier
# public name ``libero_long`` and map it back to the real file here (the on-disk
# data file is never renamed).
_SUITE_FILES: dict[str, str] = {"libero_long": "libero_10.json"}

# Fixture type -> USD directory.  LIBERO's generic ``table`` fixture is the
# kitchen-table asset (matches the reference framework's mapping).
_FIXTURE_USD_DIRS: dict[str, str] = {"table": "kitchen_table"}

# Per-fixture-type scale multiplier applied on top of the authored JSON scale.
# The ``floor`` fixture ships as a thin flat slab with a 6 m x 6 m footprint
# (scale 1.0), which overruns the per-env spacing (2.5 m) and spills into
# neighbouring envs.  Because it is flat and its objects rest within ~0.26 m of
# its centre, shrinking only its XY footprint (Z left at 1.0 to keep the surface
# height, and therefore object rest heights, unchanged) both fixes the overrun
# and keeps a comfortable margin: 6 m x 0.35 = 2.1 m < 2.5 m.  Keyed per fixture
# type (not per task) so the fix stays data-driven.
_FIXTURE_SCALE_MULTIPLIERS: dict[str, tuple[float, float, float]] = {"floor": (0.35, 0.35, 1.0)}

# Object types spawned as articulations (everything else is a rigid object).
ARTICULATION_TYPES: frozenset[str] = frozenset({"flat_stove", "microwave", "white_cabinet", "wooden_cabinet"})

# Default joint positions [rad] for each articulation type (from the LIBERO assets).
ARTICULATION_JOINT_STATES: dict[str, dict[str, float]] = {
    "flat_stove": {"button": 0.0},
    "microwave": {"microjoint": -1.79},
    "white_cabinet": {"top_level": 0.0, "middle_level": 0.0, "bottom_level": -0.1523},
    "wooden_cabinet": {"top_level": 0.0, "middle_level": 0.0, "bottom_level": 0.0},
}


@dataclass
class ObjectSpec:
    """A single object/articulation in a task layout (poses in the shared env frame)."""

    key: str
    """Scene-unique object name (task-prefixed, e.g. ``"object_task0_basket_1"``)."""
    obj_name: str
    """Raw LIBERO object name within the task (e.g. ``"basket_1"``)."""
    obj_type: str
    """LIBERO object type; USD path is ``<obj_type>/<obj_type>.usd``."""
    scale: tuple[float, float, float]
    """USD spawn scale [--]."""
    pos: tuple[float, float, float]
    """Initial position in the env frame [m] (region midpoint + workspace shift)."""
    rot: tuple[float, float, float, float]
    """Initial orientation quaternion ``(x, y, z, w)`` [--]."""
    is_articulation: bool
    """Whether the object spawns as an articulation."""
    joint_pos: dict[str, float]
    """Default joint positions [rad] (empty for rigid objects)."""


@dataclass
class FixtureSpec:
    """A static workspace fixture (floor / table)."""

    obj_type: str
    """Fixture type as authored in the JSON (e.g. ``table``, ``living_room_table``)."""
    scale: tuple[float, float, float]
    """USD spawn scale [--]."""
    pos: tuple[float, float, float]
    """Fixture position in the env frame [m] (authored pose + workspace shift)."""
    usd_dir: str = ""
    """USD directory/name (``<usd_dir>/<usd_dir>.usd``); ``table`` maps to ``kitchen_table``."""
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Initial orientation quaternion ``(x, y, z, w)`` [--] (from the JSON ``initial_quat``)."""


@dataclass
class TaskLayout:
    """The full object layout for one LIBERO task (one clone group)."""

    name: str
    """Task identifier, e.g. ``"object_task0"`` (used for group / scene keys)."""
    fixture: FixtureSpec
    """The workspace fixture for this task."""
    objects: list[ObjectSpec] = field(default_factory=list)
    """All objects/articulations in the scene."""
    primary_key: str | None = None
    """Scene key of the manipulated object (``obj_of_interest``)."""
    target_key: str | None = None
    """Scene key of the goal object (``targets``); may be ``None``."""
    interest_names: tuple[str, ...] = ()
    """Raw LIBERO ``obj_of_interest`` names (e.g. ``alphabet_soup_1``) for pose-buffer activation."""
    target_names: tuple[str, ...] = ()
    """Raw LIBERO ``targets`` names for pose-buffer activation."""
    success_xy_threshold: float = 0.10
    """Planar tolerance [m] for the geometric success proxy (from the BDDL goal)."""
    success_height_threshold: float = 0.10
    """Vertical tolerance [m] for the geometric success proxy (from the BDDL goal)."""
    workspace_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """``ROBOT_BASE_KITCHEN - task_robot_base`` [m]; applied to fixtures/objects and demo poses."""
    goals: tuple[dict, ...] = ()
    """Raw BDDL goal predicate dicts (relationship + articulation operation goals)."""

    @property
    def rigid_object_keys(self) -> list[str]:
        """Scene keys of every non-articulation object."""
        return [o.key for o in self.objects if not o.is_articulation]

    @property
    def articulation_keys(self) -> list[str]:
        """Scene keys of every articulation object."""
        return [o.key for o in self.objects if o.is_articulation]

    @property
    def primary_rest_z(self) -> float:
        """Env-frame resting height [m] of the primary object (0.9 fallback)."""
        for o in self.objects:
            if o.key == self.primary_key:
                return o.pos[2]
        return 0.9

    @property
    def asset_signature(self) -> frozenset[tuple[str, str]]:
        """The suite's asset-key set: ``(obj_name, obj_type)`` for every object.

        Ignores per-env poses; used to decide whether a suite is homogeneous
        (all tasks share the identical asset set).
        """
        return frozenset((o.obj_name, o.obj_type) for o in self.objects)


def suite_is_homogeneous(layouts: list[TaskLayout]) -> bool:
    """Return ``True`` when every task in a suite shares the identical asset set.

    Homogeneous suites (``libero_spatial`` / ``libero_goal``) are realised as a
    single clone group with a per-env sub-task layer; heterogeneous suites
    (``libero_object`` / ``libero_long``) get one clone group per task.
    """
    return len({layout.asset_signature for layout in layouts}) == 1


def _midpoint(values) -> float:
    """Return the midpoint of a ``[min, max]`` range (0.0 when missing)."""
    if values is None:
        return 0.0
    if isinstance(values, (list, tuple)):
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        return float((values[0] + values[1]) * 0.5)
    return float(values)


def _quat_xyzw_from_pose_range(pose_range: dict) -> tuple[float, float, float, float]:
    """Convert a region's midpoint roll/pitch/yaw [rad] to an ``(x, y, z, w)`` quaternion."""
    roll = _midpoint(pose_range.get("roll"))
    pitch = _midpoint(pose_range.get("pitch"))
    yaw = _midpoint(pose_range.get("yaw"))
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


def _quat_xyzw_from_wxyz(quat) -> tuple[float, float, float, float]:
    """Convert a LIBERO ``initial_quat`` (``w, x, y, z``) to ``(x, y, z, w)``.

    LIBERO/robosuite author fixture orientations in ``wxyz`` order; IsaacLab's
    current ``InitialStateCfg.rot`` convention is ``xyzw``.  For example the
    ``living_room_table`` ``initial_quat = [0.707, 0, 0, 0.707]`` (a +90° yaw
    about Z) becomes ``(0, 0, 0.707, 0.707)``.
    """
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    return (x, y, z, w)


def _scale_tuple(scale) -> tuple[float, float, float]:
    """Coerce a JSON scale (scalar or list) to a 3-tuple."""
    if isinstance(scale, (list, tuple)):
        if len(scale) == 1:
            return (float(scale[0]),) * 3
        return (float(scale[0]), float(scale[1]), float(scale[2]))
    return (float(scale),) * 3


def load_suite(suite_name: str, task_prefix: str) -> list[TaskLayout]:
    """Load and normalise every task layout of a LIBERO suite.

    Args:
        suite_name: Suite key, e.g. ``"libero_object"`` or ``"libero_long"``.
            ``"libero_long"`` maps to the on-disk ``libero_10.json`` (see
            :data:`_SUITE_FILES`).
        task_prefix: Prefix for per-task names / scene keys, e.g. ``"object"`` →
            ``"object_task0"``.  Keeps scene keys unique across suites.

    Returns:
        One :class:`TaskLayout` per task, ordered by ``task_id``.

    Raises:
        FileNotFoundError: If the suite JSON cannot be located (config dir unset
            or file missing).
    """
    config_dir = libero_config_dir()
    suite_path = os.path.join(config_dir, _SUITE_FILES.get(suite_name, f"{suite_name}.json"))
    if not config_dir or not os.path.isfile(suite_path):
        raise FileNotFoundError(
            f"LIBERO suite config '{suite_path}' not found. Set LIBERO_CONFIG_DIR (or LIBERO_ASSETS_DATA_DIR "
            f"so its sibling 'config' dir is discoverable) to the LIBERO config directory."
        )
    with open(suite_path) as fh:
        suite_data = json.load(fh)

    layouts: list[TaskLayout] = []
    for task in suite_data.get("tasks", []):
        task_id = int(task["task_id"])
        name = f"{task_prefix}_task{task_id}"
        base = task["robot_base_pos"]
        dx = ROBOT_BASE_KITCHEN[0] - float(base[0])
        dy = ROBOT_BASE_KITCHEN[1] - float(base[1])
        dz = ROBOT_BASE_KITCHEN[2] - float(base[2])
        workspace_shift = (dx, dy, dz)

        # ── Fixture (one per task) ────────────────────────────────────────
        fx_name, fx_meta = next(iter(task["fixtures"].items()))
        fx_type = fx_meta["type"]
        fx_pose = fx_meta.get("initial_pose", [0.0, 0.0, 0.0])
        fx_quat = fx_meta.get("initial_quat", [1.0, 0.0, 0.0, 0.0])
        fx_scale = _scale_tuple(fx_meta.get("scale", [1.0, 1.0, 1.0]))
        fx_mult = _FIXTURE_SCALE_MULTIPLIERS.get(fx_type, (1.0, 1.0, 1.0))
        fixture = FixtureSpec(
            obj_type=fx_type,
            scale=(fx_scale[0] * fx_mult[0], fx_scale[1] * fx_mult[1], fx_scale[2] * fx_mult[2]),
            pos=(float(fx_pose[0]) + dx, float(fx_pose[1]) + dy, float(fx_pose[2]) + dz),
            usd_dir=_FIXTURE_USD_DIRS.get(fx_type, fx_type),
            rot=_quat_xyzw_from_wxyz(fx_quat),
        )

        # ── Objects ───────────────────────────────────────────────────────
        regions = task.get("regions", {})
        objects: list[ObjectSpec] = []
        key_by_object: dict[str, str] = {}
        for obj_name, obj_meta in task["objects"].items():
            obj_type = obj_meta["type"]
            pr = regions.get(obj_meta.get("initial_region"), {}).get("pose_range", {})
            pos = (_midpoint(pr.get("x")) + dx, _midpoint(pr.get("y")) + dy, _midpoint(pr.get("z")) + dz)
            is_arti = obj_type in ARTICULATION_TYPES
            scene_key = f"{name}_{obj_name}"
            key_by_object[obj_name] = scene_key
            objects.append(
                ObjectSpec(
                    key=scene_key,
                    obj_name=obj_name,
                    obj_type=obj_type,
                    scale=_scale_tuple(obj_meta.get("scale", 1.0)),
                    pos=pos,
                    rot=_quat_xyzw_from_pose_range(pr),
                    is_articulation=is_arti,
                    joint_pos=dict(ARTICULATION_JOINT_STATES.get(obj_type, {})),
                )
            )

        interest_names = tuple(n for n in (task.get("obj_of_interest") or []) if n in key_by_object)
        target_names = tuple(n for n in (task.get("targets") or []) if n in key_by_object)
        primary_obj = interest_names[0] if interest_names else None
        target_obj = target_names[0] if target_names else None

        # ── Goal predicate → geometric success thresholds ────────────────
        # Prefer the first relational ("on" / "in") goal: its ref/target names
        # and xy/height tolerances define the per-task success proxy. When only
        # operation goals ("open"/"turnon") exist, fall back to obj_of_interest
        # + targets with default tolerances.
        xy_thr, hz_thr = 0.10, 0.10
        for goal in task.get("goals") or []:
            if goal.get("relationship") in ("on", "in") and goal.get("ref_obj") and goal.get("target"):
                if goal["ref_obj"] in key_by_object:
                    primary_obj = goal["ref_obj"]
                    if goal["ref_obj"] not in interest_names:
                        interest_names = interest_names + (goal["ref_obj"],)
                # a goal may name a body of the target ("white_cabinet_1/drawer_bottom",
                # "flat_stove_1/burnerplate"); the object key is the leading segment
                target_base = goal["target"].split("/")[0]
                if target_base in key_by_object:
                    target_obj = target_base
                    if target_base not in target_names:
                        target_names = target_names + (target_base,)
                xy_thr = float(goal.get("xy_threshold", xy_thr))
                hz_thr = float(goal.get("height_threshold", hz_thr))
                break

        layouts.append(
            TaskLayout(
                name=name,
                fixture=fixture,
                objects=objects,
                primary_key=key_by_object.get(primary_obj),
                target_key=key_by_object.get(target_obj),
                interest_names=interest_names,
                target_names=target_names,
                goals=tuple(task.get("goals") or []),
                success_xy_threshold=xy_thr,
                success_height_threshold=hz_thr,
                workspace_shift=workspace_shift,
            )
        )
    return layouts


# ---------------------------------------------------------------------------
# Goal-evaluation tables (suite-independent, keyed by asset)
# ---------------------------------------------------------------------------

_GOAL_THRESHOLDS_FILE = "goal_thresholds.json"


@functools.lru_cache(maxsize=1)
def load_goal_thresholds() -> dict:
    """Load ``goal_thresholds.json`` from the LIBERO config directory.

    Holds the articulation joint windows and the relationship orientation
    offsets used by the success predicates and the dense reward. Unlike the
    suite files these are keyed by asset rather than by task, so they live in a
    single shared file.

    Cached: the file is read once per process, and the tables are constants of
    the scene, not of a run.

    Raises:
        FileNotFoundError: If the config dir is unset or the file is missing.
            These tables gate task success, so a missing file is a hard error
            rather than a fall back to built-in values -- two sources of truth
            for a success threshold is how they drift apart.
    """
    config_dir = libero_config_dir()
    path = os.path.join(config_dir, _GOAL_THRESHOLDS_FILE)
    if not config_dir or not os.path.isfile(path):
        raise FileNotFoundError(
            f"LIBERO goal thresholds '{path}' not found. Set LIBERO_CONFIG_DIR (or LIBERO_ASSETS_DATA_DIR "
            f"so its sibling 'config' dir is discoverable) to the LIBERO config directory."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def articulation_joint_spec(target: str, operation: str, layer: str | None) -> tuple[float, float, str] | None:
    """Return ``(low, high, joint_pattern)`` for an articulation goal, else ``None``.

    ``target`` is the goal's canonical asset name. The wooden cabinet is layered,
    so its joint comes from ``layer`` and its window from ``operation``; every
    other asset is keyed directly by ``(asset, operation)``.
    """
    tables = load_goal_thresholds()
    cabinet = tables["wooden_cabinet"]
    if target == cabinet["asset"]:
        joint = cabinet["layer_joint"].get(layer or "")
        window = cabinet["operation_range"].get(operation)
        if joint is None or window is None:
            return None
        return float(window["low"]), float(window["high"]), joint
    entry = tables["articulation_joint_targets"].get(target, {}).get(operation)
    if entry is None:
        return None
    return float(entry["low"]), float(entry["high"]), str(entry["joint"])


def orientation_offset(orientation: str) -> tuple[float, float, float] | None:
    """Positional offset [m] for a relationship goal's side, or ``None``."""
    raw = load_goal_thresholds()["orientation_offsets"].get(orientation)
    return (float(raw[0]), float(raw[1]), float(raw[2])) if raw else None
