# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prototype-sharing task modules for the combined LIBERO environment.

Each :class:`LiberoPrototypeTaskCfg` is a thin task registration whose only job
is to contribute its task's **shared** prototype scene assets (see
:mod:`.prototypes`).  Because identical models across tasks reuse the *same*
scene-entity name and cfg object, the registry's per-registration
:class:`~isaaclab.cloner.InclusionSet` clones each shared prototype into the union
of the envs whose task uses it -- one spawn, one
:class:`~isaaclab.assets.AssetView`.

All task-dependent MDP (rewards, terminations, task observations, ``task_onehot``,
and object reset) is intentionally left empty here and supplied instead by the
per-env task-gather terms in :mod:`...mdp.combined`, because a shared AssetView
spans envs of different tasks and cannot be scoped per clone group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_contrib.tasks.manipulation.multitask.tasks._base import TaskModuleCfg

from ..common.base import fixture, flat_stove, microwave, rigid_object, white_cabinet, wooden_cabinet
from .prototypes import LIBERO_SUITES, Prototype, TaskBinding, harvest_libero_prototypes

if TYPE_CHECKING:
    from isaaclab_contrib.tasks.manipulation.multitask.robots._base import RobotModuleCfg

_ARTICULATION_BUILDERS = {
    "wooden_cabinet": wooden_cabinet,
    "flat_stove": flat_stove,
    "microwave": microwave,
    "white_cabinet": white_cabinet,
}


def build_prototype_cfgs(prototypes: list[Prototype]) -> dict[str, object]:
    """Build one scene-asset cfg per prototype, keyed by its unique scene name.

    Args:
        prototypes: The de-duplicated prototypes from :func:`.harvest_libero_prototypes`.
            Those flagged ``kinematic`` spawn unpushable; see
            :data:`~...assets.OBJECT_RECEPTACLE_RIGID_PROPS`.

    Returns:
        Mapping from prototype name to its (fixture / rigid / articulation) cfg.

    Note:
        The authored JSON scale is applied to fixtures and rigid objects but *not*
        to articulations: the four articulation USDs already bake their JSON scale
        into the asset (``wooden_cabinet`` / ``white_cabinet``: 0.05 on the root
        link, ``microwave``: 0.5, ``flat_stove``: 1.0), while the rigid-object USDs
        spawn at scale 1.0 and need it.  Passing it again shrank the cabinets to
        1/20 and the microwave to 1/2 of their authored size.  The reference
        framework likewise spawns these four types with no ``scale``.
    """
    cfgs: dict[str, object] = {}
    for proto in prototypes:
        if proto.is_fixture:
            cfgs[proto.name] = fixture(
                proto.name, proto.usd_rel_path, scale=proto.scale, pos=proto.canonical_pos, rot=proto.canonical_rot
            )
        elif proto.is_articulation:
            cfgs[proto.name] = _ARTICULATION_BUILDERS[proto.obj_type](
                proto.name, pos=proto.canonical_pos, rot=proto.canonical_rot
            )
        else:
            cfgs[proto.name] = rigid_object(
                proto.name,
                proto.usd_rel_path,
                scale=proto.scale,
                pos=proto.canonical_pos,
                rot=proto.canonical_rot,
                kinematic=proto.kinematic,
            )
    return cfgs


@dataclass
class LiberoPrototypeTaskCfg(TaskModuleCfg):
    """A single combined-env task that contributes only its shared prototype assets."""

    task_name: str = ""
    """Task identifier, e.g. ``"object_task3"`` (unique clone-group name)."""
    proto_cfgs: dict[str, object] = field(default_factory=dict)
    """The subset of prototype cfgs this task clones (shared across tasks)."""

    @property
    def name(self) -> str:
        """Return the unique task / clone-group name (:attr:`task_name`)."""
        return self.task_name

    def scene_assets(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        """Return the shared prototype cfgs this task clones (:attr:`proto_cfgs`)."""
        return dict(self.proto_cfgs)

    def command_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        """Contribute no command terms; the task MDP is supplied by ``mdp/combined.py`` gather terms."""
        return {}

    def task_obs_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        """Contribute no task observations; the task MDP is supplied by ``mdp/combined.py`` gather terms."""
        return {}

    def scatter_obs_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        """Contribute no scattered observations; the task MDP is supplied by ``mdp/combined.py`` gather terms."""
        return {}

    def reward_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        """Contribute no rewards; the task MDP is supplied by per-env gather terms in ``mdp/combined.py``."""
        return {}

    def termination_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        """Contribute no terminations; the task MDP is supplied by per-env gather terms in ``mdp/combined.py``."""
        return {}

    def reset_events(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        """Contribute no reset events; object reset is supplied by per-env gather terms in ``mdp/combined.py``."""
        return {}


def build_combined_tasks(
    suites: tuple[tuple[str, str], ...] = LIBERO_SUITES,
) -> tuple[list[LiberoPrototypeTaskCfg], list[Prototype], list[TaskBinding]]:
    """Harvest the selected suites and build one prototype-sharing task per LIBERO task.

    Args:
        suites: ``(suite_name, prefix)`` pairs to include, in task-index order.
            Defaults to all four suites; pass a subset to build a smaller combo.

    Returns:
        ``(task_cfgs, prototypes, bindings)`` -- the per-task registrations (in
        harvest / task-index order), the de-duplicated prototypes (scene-field
        order, shared across the *selected* suites), and the per-task bindings
        consumed by the per-env gather MDP.
    """
    prototypes, bindings = harvest_libero_prototypes(suites)
    frozen = sorted(p.name for p in prototypes if p.kinematic)
    if frozen:
        print(f"[libero] static (kinematic) receptacles: {frozen}")
    proto_cfgs = build_prototype_cfgs(prototypes)
    task_cfgs: list[LiberoPrototypeTaskCfg] = []
    for binding in bindings:
        used = [binding.fixture_proto, *[b.proto_name for b in binding.object_bindings]]
        task_cfgs.append(
            LiberoPrototypeTaskCfg(task_name=binding.name, proto_cfgs={name: proto_cfgs[name] for name in used})
        )
    return task_cfgs, prototypes, bindings
