# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-level prototype harvesting for the combined LIBERO environment.

The combined ``Isaac-Hebero-All-*`` environment trains every LIBERO task
(4 suites x 10 tasks) in one scene.  Rather than namespacing each task's objects
per clone group -- which spawns the *same* USD once per task and wastes an
:class:`~isaaclab.assets.AssetView` per copy -- this module mirrors the
prototype-sharing model of ``scripts/demos/heterogeneous_scene_cloner_selector``:

* **prototype** -- one distinct object model (USD path + scale), spawned a single
  time and cloned into the union of the envs whose task uses it.  Identical models
  across tasks (e.g. ``alphabet_soup`` in several ``object`` / ``long`` tasks)
  collapse to ONE prototype = ONE scene entity = ONE :class:`AssetView`.
* **task binding** -- one LIBERO task: the prototype names it uses, each with this
  task's authored init pose (for per-task reset), plus its primary/target
  prototypes and success tolerances (for the per-env reward gather).

De-duplication follows the demo's :func:`prototype_key`: resettable objects share
by *model* (type + scale), with an ``occurrence`` index keeping same-model
duplicates *within* one task distinct (e.g. two ``akita_black_bowl`` share their
0th / 1st slot across tasks but stay two prototypes).  Static fixtures also key on
pose, since they cannot be re-posed after spawn.

The harvest is pure data (no simulator), so it can be inspected and unit-checked
with ``./isaaclab.sh -p`` without launching Isaac Sim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..common.suite_loader import load_suite

# Internal suite name -> per-task scene-key prefix, harvested in this order.  The
# order fixes the task index (and therefore the ``env i -> task i % n_tasks``
# assignment).  DGPO+ABC eval uses :data:`~...dgpo_layout.DGPO_ABC_HARVEST_SUITES`
# instead (libero_10 / object / spatial / goal) to match playground assignment order.
LIBERO_SUITES: tuple[tuple[str, str], ...] = (
    ("libero_spatial", "spatial"),
    ("libero_goal", "goal"),
    ("libero_object", "object"),
    ("libero_long", "long"),
)


def _round_scale(scale: tuple[float, float, float]) -> tuple[float, float, float]:
    """Round a scale tuple so float noise does not split otherwise-equal models."""
    return tuple(round(float(x), 4) for x in scale)


@dataclass
class Prototype:
    """One distinct object model, spawned once and cloned into the envs that use it."""

    name: str
    """Unique scene-entity name (prim leaf), e.g. ``"akita_black_bowl"``."""
    key: str
    """De-duplication key (model signature + occurrence, or + pose for fixtures)."""
    obj_type: str
    """LIBERO object/fixture type; USD lives at ``<obj_type>/<obj_type>.usd``."""
    usd_rel_path: str
    """USD path relative to the LIBERO asset root."""
    scale: tuple[float, float, float]
    """USD spawn scale [--]."""
    is_articulation: bool
    """Whether the prototype spawns as an articulation."""
    is_fixture: bool
    """Whether the prototype is a static workspace fixture (floor / table)."""
    joint_pos: dict[str, float]
    """Default joint positions [rad] (empty for rigid objects / fixtures)."""
    canonical_pos: tuple[float, float, float]
    """Spawn position in the env frame [m] (the first task to register it)."""
    canonical_rot: tuple[float, float, float, float]
    """Spawn orientation quaternion ``(x, y, z, w)`` [--] (first registering task)."""
    kinematic: bool = False
    """Whether this instance spawns kinematic (unpushable).

    True for the receptacle side of a relationship goal; see
    :func:`~..harvest.combined.resolve_static_receptacle_protos`.  It is part of
    the de-duplication key, so a model that is a receptacle in some tasks and a
    carried object in others yields *two* prototypes -- one kinematic, one not --
    rather than forcing a single shared choice on every task that uses it.
    """
    task_names: list[str] = field(default_factory=list)
    """Names of the tasks that use this prototype."""

    @property
    def shared(self) -> bool:
        """Whether more than one task uses this prototype."""
        return len(self.task_names) > 1


@dataclass
class ObjectBinding:
    """One object slot of a task: which prototype, at which per-task init pose."""

    proto_name: str
    """Name of the shared prototype this slot maps to."""
    canonical_name: str
    """Raw LIBERO object name (e.g. ``alphabet_soup_1``); matches pose-buffer columns."""
    pos: tuple[float, float, float]
    """This task's init position in the env frame [m]."""
    rot: tuple[float, float, float, float]
    """This task's init orientation quaternion ``(x, y, z, w)`` [--]."""
    is_articulation: bool
    """Whether the bound prototype is an articulation."""


@dataclass
class TaskBinding:
    """One LIBERO task expressed over shared prototypes."""

    name: str
    """Task identifier, e.g. ``"object_task3"``."""
    suite: str
    """Owning suite key, e.g. ``"libero_object"``."""
    fixture_proto: str
    """Name of this task's workspace-fixture prototype."""
    object_bindings: list[ObjectBinding]
    """Per-object prototype bindings (rigid + articulation; excludes the fixture)."""
    primary_proto: str | None
    """Prototype name of the manipulated object (``obj_of_interest``)."""
    target_proto: str | None
    """Prototype name of the goal object (falls back to the primary object)."""
    interest_names: tuple[str, ...] = ()
    """Raw ``obj_of_interest`` names for DGPO pose-buffer activation."""
    target_names: tuple[str, ...] = ()
    """Raw ``targets`` names for DGPO pose-buffer activation."""
    success_xy_threshold: float = 0.10
    """Planar success tolerance [m] (from the BDDL goal)."""
    success_height_threshold: float = 0.10
    """Vertical success tolerance [m] (from the BDDL goal)."""
    primary_rest_z: float = 0.9
    """Env-frame resting height [m] of the primary object."""
    workspace_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """``ROBOT_BASE_KITCHEN - task_robot_base`` [m]; added to demo ``initial_state`` poses on load."""
    goals: tuple[dict, ...] = ()
    """Raw BDDL goal predicate dicts (relationship + articulation operation goals)."""

    @property
    def active_canonical_names(self) -> tuple[str, ...]:
        """Ordered unique interest+target names used to activate pose-buffer slots."""
        seen: list[str] = []
        for name in (*self.interest_names, *self.target_names):
            if name not in seen:
                seen.append(name)
        return tuple(seen)


def _static_receptacle_names(goals: tuple[dict, ...]) -> set[str]:
    """Return the objects a task places something onto and never itself moves.

    The freeze set is this task's relationship-goal ``target`` operands minus every
    operand appearing as any of its goals' ``ref_obj`` -- a container that is itself
    carried later in a long-horizon task stays dynamic.  Region suffixes
    (``plate_1/top``) are stripped.
    """
    targets: set[str] = set()
    refs: set[str] = set()
    for goal in goals or ():
        ref = goal.get("ref_obj") or ""
        if ref:
            refs.add(ref.split("/")[0])
        if "relationship" in goal:
            target = goal.get("target") or ""
            if target:
                targets.add(target.split("/")[0])
    return targets - refs


def harvest_libero_prototypes(
    suites: tuple[tuple[str, str], ...] = LIBERO_SUITES,
) -> tuple[list[Prototype], list[TaskBinding]]:
    """Harvest every LIBERO task into shared prototypes + per-task bindings.

    Args:
        suites: ``(suite_name, prefix)`` pairs to harvest, in task-index order.

    Returns:
        ``(prototypes, tasks)`` where ``prototypes`` are the de-duplicated object
        models (stable registration order == scene-field order) and ``tasks`` are
        the per-task bindings (one per LIBERO task, in ``suites`` order).

    Raises:
        FileNotFoundError: If a suite JSON config cannot be located.
    """
    prototypes: dict[str, Prototype] = {}
    taken_names: set[str] = set()
    tasks: list[TaskBinding] = []

    def _unique_name(stem: str) -> str:
        base = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_").lower() or "asset"
        name, i = base, 1
        while name in taken_names:
            i += 1
            name = f"{base}_{i}"
        taken_names.add(name)
        return name

    for suite_name, prefix in suites:
        for layout in load_suite(suite_name, prefix):
            # ── Fixture prototype (static: keyed on model + pose) ──────────
            fx = layout.fixture
            fx_pos = tuple(round(v, 3) for v in fx.pos)
            fx_rot = tuple(round(v, 3) for v in fx.rot)
            fx_key = f"fixture|{fx.usd_dir}|{_round_scale(fx.scale)}|pos={fx_pos}|rot={fx_rot}"
            fixture_proto = prototypes.get(fx_key)
            if fixture_proto is None:
                fixture_proto = Prototype(
                    name=_unique_name(fx.usd_dir),
                    key=fx_key,
                    obj_type=fx.usd_dir,
                    usd_rel_path=f"{fx.usd_dir}/{fx.usd_dir}.usd",
                    scale=fx.scale,
                    is_articulation=False,
                    is_fixture=True,
                    joint_pos={},
                    canonical_pos=fx.pos,
                    canonical_rot=fx.rot,
                )
                prototypes[fx_key] = fixture_proto
            if layout.name not in fixture_proto.task_names:
                fixture_proto.task_names.append(layout.name)

            # ── Object prototypes (resettable: keyed on model + occurrence) ─
            # Receptacles -- objects this task's relationship goals place something
            # *onto*, and which this task never asks the robot to move -- spawn
            # kinematic so the sparse task_success term cannot be satisfied by
            # shoving the receptacle under the object.  The rule is per task, and
            # ``kinematic`` is part of the prototype key, so a model that is a
            # receptacle here and a carried object elsewhere gets two prototypes.
            static_receptacles = _static_receptacle_names(layout.goals)
            per_model_count: dict[tuple[str, tuple[float, float, float]], int] = {}
            object_bindings: list[ObjectBinding] = []
            proto_by_object_key: dict[str, str] = {}
            for obj in layout.objects:
                sig = (obj.obj_type, _round_scale(obj.scale))
                occ = per_model_count.get(sig, 0)
                per_model_count[sig] = occ + 1
                # Articulations take the ArticulationCfg branch and are never frozen,
                # so drawers, doors and stove knobs keep working.
                kinematic = not obj.is_articulation and obj.obj_name in static_receptacles
                key = f"obj|{obj.obj_type}|{_round_scale(obj.scale)}#{occ}{'|kin' if kinematic else ''}"
                proto = prototypes.get(key)
                if proto is None:
                    proto = Prototype(
                        name=_unique_name(f"{obj.obj_type}_static" if kinematic else obj.obj_type),
                        key=key,
                        obj_type=obj.obj_type,
                        usd_rel_path=f"{obj.obj_type}/{obj.obj_type}.usd",
                        scale=obj.scale,
                        is_articulation=obj.is_articulation,
                        is_fixture=False,
                        joint_pos=dict(obj.joint_pos),
                        canonical_pos=obj.pos,
                        canonical_rot=obj.rot,
                        kinematic=kinematic,
                    )
                    prototypes[key] = proto
                if layout.name not in proto.task_names:
                    proto.task_names.append(layout.name)
                object_bindings.append(
                    ObjectBinding(
                        proto_name=proto.name,
                        canonical_name=obj.obj_name,
                        pos=obj.pos,
                        rot=obj.rot,
                        is_articulation=obj.is_articulation,
                    )
                )
                proto_by_object_key[obj.key] = proto.name

            primary_proto = proto_by_object_key.get(layout.primary_key)
            target_proto = proto_by_object_key.get(layout.target_key) or primary_proto
            tasks.append(
                TaskBinding(
                    name=layout.name,
                    suite=suite_name,
                    fixture_proto=fixture_proto.name,
                    object_bindings=object_bindings,
                    primary_proto=primary_proto,
                    target_proto=target_proto,
                    interest_names=layout.interest_names,
                    target_names=layout.target_names,
                    success_xy_threshold=layout.success_xy_threshold,
                    success_height_threshold=layout.success_height_threshold,
                    primary_rest_z=layout.primary_rest_z,
                    workspace_shift=layout.workspace_shift,
                    goals=layout.goals,
                )
            )

    return list(prototypes.values()), tasks
