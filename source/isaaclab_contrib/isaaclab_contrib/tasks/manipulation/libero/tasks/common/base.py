# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared base class and asset builders for the LIBERO task modules.

The LIBERO suites share the same kitchen-table fixture, the same articulation
types (wooden cabinet + flat stove), and the same reward structure (reach the
primary object, lift it, and bring it toward a target object).  The concrete
per-suite object layouts are loaded from JSON by :mod:`..common.suite_loader`
and assembled into shared prototypes by :mod:`..harvest.prototypes`.
"""

from __future__ import annotations

import os
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg as TermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaaclab_contrib.tasks.manipulation.multitask.tasks._base import TaskModuleCfg

from ... import mdp
from ...assets import (
    OBJECT_RECEPTACLE_RIGID_PROPS,
    OBJECT_RIGID_PROPS,
    libero_assets_dir,
    spawn_receptacle_from_usd,
)
from .... import settings

if TYPE_CHECKING:
    from isaaclab_contrib.tasks.manipulation.multitask.robots._base import RobotModuleCfg

# Fixed scene names for the single LIBERO robot (see ``robots/franka_osc.py``).
_ROBOT_ASSET = "franka_robot"
_EE_FRAME_ASSET = "franka_ee_frame"

# Selectable reward modes (see :meth:`LiberoManipulationTaskCfg.reward_terms`).
REWARD_MODES: tuple[str, ...] = ("goal", "metaworld_dense", "world_state_tracking_reward")

# Emitted once per process when ``world_state_tracking_reward`` is requested without a demo
# reference source (see :meth:`LiberoManipulationTaskCfg._world_state_tracking_reward_terms`).
_WORLD_STATE_TRACKING_WARNED = False


def resolve_reward_mode(cfg_value: str) -> str:
    """Resolve the active LIBERO reward mode.

    The ``LIBERO_REWARD_MODE`` environment variable (values ``goal``,
    ``metaworld_dense``, ``world_state_tracking_reward``) overrides the ``cfg_value`` field
    when set; otherwise ``cfg_value`` is used.

    Args:
        cfg_value: The task cfg's ``reward_mode`` field value (the default).

    Returns:
        The resolved reward mode, one of :data:`REWARD_MODES`.

    Raises:
        ValueError: If the resolved mode is not a recognised reward mode.
    """
    mode = settings.text("LIBERO_REWARD_MODE").lower() or cfg_value
    if mode not in REWARD_MODES:
        raise ValueError(f"LIBERO reward mode '{mode}' is invalid; expected one of {REWARD_MODES}.")
    return mode


# ---------------------------------------------------------------------------
# Asset builders
# ---------------------------------------------------------------------------


def kitchen_table(scene_key: str) -> AssetBaseCfg:
    """Return the LIBERO kitchen-table fixture spawned at ``scene_key``.

    Args:
        scene_key: Env-relative prim/scene name (e.g. ``"spatial_main_table"``).

    Returns:
        The kitchen-table asset config.
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{scene_key}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.845)),
        spawn=UsdFileCfg(
            usd_path=f"{libero_assets_dir()}/kitchen_table/kitchen_table.usd",
            scale=(0.02, 0.02, 0.02),
        ),
    )


def rigid_object(
    scene_key: str,
    usd_rel_path: str,
    scale: tuple[float, float, float],
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    kinematic: bool = False,
) -> RigidObjectCfg:
    """Build a LIBERO rigid object with the shared solver properties.

    Args:
        scene_key: Env-relative prim/scene name.
        usd_rel_path: USD path relative to the LIBERO asset root.
        scale: Uniform/per-axis USD scale [--].
        pos: Initial position relative to the env origin [m].
        rot: Initial orientation quaternion ``(x, y, z, w)`` [--].
        kinematic: Spawn the object kinematic (unpushable).  Used for the
            receptacle side of a relationship goal; see
            :data:`~...assets.OBJECT_RECEPTACLE_RIGID_PROPS`.

    Returns:
        The rigid-object config.
    """
    spawn = UsdFileCfg(
        usd_path=f"{libero_assets_dir()}/{usd_rel_path}",
        scale=scale,
        rigid_props=OBJECT_RECEPTACLE_RIGID_PROPS if kinematic else OBJECT_RIGID_PROPS,
    )
    if kinematic:
        # PhysX rejects per-body CCD on a kinematic body; the receptacle spawner drops
        # the flag the asset authors instead of logging one error per body per env.
        spawn.func = spawn_receptacle_from_usd
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{scene_key}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=spawn,
    )


def fixture(
    scene_key: str,
    usd_rel_path: str,
    scale: tuple[float, float, float],
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> AssetBaseCfg:
    """Build a static workspace fixture (floor / table) as a non-interactive asset.

    Args:
        scene_key: Env-relative prim/scene name.
        usd_rel_path: USD path relative to the LIBERO asset root.
        scale: Uniform/per-axis USD scale [--].
        pos: Fixture position relative to the env origin [m].
        rot: Fixture orientation quaternion ``(x, y, z, w)`` [--].

    Returns:
        The fixture asset config.
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{scene_key}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=UsdFileCfg(
            usd_path=f"{libero_assets_dir()}/{usd_rel_path}",
            scale=scale,
        ),
    )


def wooden_cabinet(
    scene_key: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ArticulationCfg:
    """Build the LIBERO wooden cabinet (three drawer joints).

    The USD bakes the authored 0.05 scale into its root link, so this spawns at
    1.0 like the other articulation builders -- passing the JSON scale here
    applies it twice (see :func:`...harvest.combined.build_prototype_cfgs`).
    """
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{scene_key}",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos={"top_level": 0.0, "middle_level": 0.0, "bottom_level": 0.0},
        ),
        spawn=UsdFileCfg(
            usd_path=f"{libero_assets_dir()}/wooden_cabinet/wooden_cabinet.usd",
            scale=scale,
        ),
        actuators={
            "drawer_actuator": ImplicitActuatorCfg(
                joint_names_expr=["top_level", "middle_level", "bottom_level"],
                effort_limit_sim=10.0,
                velocity_limit_sim=2.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.01,
            ),
        },
    )


def flat_stove(
    scene_key: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ArticulationCfg:
    """Build the LIBERO flat stove (single knob joint)."""
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{scene_key}",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos={"button": 0.0},
        ),
        spawn=UsdFileCfg(
            usd_path=f"{libero_assets_dir()}/flat_stove/flat_stove.usd",
            scale=scale,
        ),
        actuators={
            "knob_actuator": ImplicitActuatorCfg(
                joint_names_expr=["button"],
                effort_limit_sim=10.0,
                velocity_limit_sim=2.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.5,
            ),
        },
    )


def microwave(
    scene_key: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ArticulationCfg:
    """Build the LIBERO microwave (single door joint)."""
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{scene_key}",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos={"microjoint": -1.79},
        ),
        spawn=UsdFileCfg(
            usd_path=f"{libero_assets_dir()}/microwave/microwave.usd",
            scale=scale,
        ),
        actuators={
            "door_actuator": ImplicitActuatorCfg(
                joint_names_expr=["microjoint"],
                effort_limit_sim=0.01,
                velocity_limit_sim=20.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.2,
            ),
        },
    )


def white_cabinet(
    scene_key: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ArticulationCfg:
    """Build the LIBERO white cabinet (three drawer joints)."""
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{scene_key}",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos={"top_level": 0.0, "middle_level": 0.0, "bottom_level": -0.1523},
        ),
        spawn=UsdFileCfg(
            usd_path=f"{libero_assets_dir()}/white_cabinet/white_cabinet.usd",
            scale=scale,
        ),
        actuators={
            "drawer_bottom_actuator": ImplicitActuatorCfg(
                joint_names_expr=["bottom_level"],
                effort_limit_sim=0.0,
                velocity_limit_sim=20.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
            ),
            "drawer_top_middle_actuator": ImplicitActuatorCfg(
                joint_names_expr=["top_level", "middle_level"],
                effort_limit_sim=10.0,
                velocity_limit_sim=20.0,
                stiffness=1000.0,
                damping=0.0,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Base task module
# ---------------------------------------------------------------------------


@dataclass
class LiberoManipulationTaskCfg(TaskModuleCfg):
    """Base class for the LIBERO manipulation suites.

    Subclasses declare the per-suite object layout by implementing
    :meth:`object_assets`, :meth:`rigid_object_keys`,
    :meth:`articulation_keys`, :attr:`primary_object_key`, and
    :attr:`target_object_key`.

    The reward is selected by :attr:`reward_mode` (overridable via the
    ``LIBERO_REWARD_MODE`` environment variable):

    * ``goal`` (default): sparse success bonus only (weight
      :attr:`success_weight`); episodes end on success.  Fair baseline signal.
    * ``metaworld_dense``: demo-free dense shaping (reach the primary object,
      lift it, bring it toward the target) plus the success bonus.
    * ``world_state_tracking_reward``: dense demo/trajectory tracking.  This repo has no
      per-step demo reference stream, so the dense tracking terms are gated off
      with a warning and only the success bonus remains (see
      :meth:`_world_state_tracking_reward_terms`).

    Task success is a geometric proxy (:func:`...mdp.object_reached_target_mask`):
    the primary object placed within :attr:`success_xy_threshold` and
    :attr:`success_height_threshold` of the target and nearly at rest.  No BDDL
    predicate evaluator, demo replay, or command streams are involved.
    """

    reward_mode: str = "goal"
    """Reward mode; one of :data:`REWARD_MODES`. ``LIBERO_REWARD_MODE`` overrides it."""

    success_weight: float = 10.0
    """Reward weight for the sparse task-success bonus."""

    success_xy_threshold: float = 0.08
    """Planar tolerance [m] for the geometric success proxy."""

    success_height_threshold: float = 0.06
    """Vertical tolerance [m] for the geometric success proxy."""

    success_speed_threshold: float | None = 0.4
    """Max object speed [m/s] for success; ``None`` disables the velocity check."""

    lift_height: float = 0.95
    """World height above which the primary object counts as lifted [m]."""

    drop_height: float = 0.5
    """World height below which the episode terminates (object fell off) [m]."""

    reach_std: float = 0.1
    """Tanh kernel width for the ee-to-object reach reward [m]."""

    reach_weight: float = 1.0
    """Reward weight for the ee-to-object reach shaping."""

    lift_weight: float = 5.0
    """Reward weight for the object-lifted bonus."""

    place_std: float = 0.1
    """Tanh kernel width for the object-to-target place reward [m]."""

    place_weight: float = 3.0
    """Reward weight for the object-to-target place shaping."""

    object_spawn_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)}
    )
    """Uniform randomisation range for the primary object at reset [m]."""

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Suite identifier (e.g. ``"spatial"``, ``"goal"``)."""
        ...

    @abstractmethod
    def object_assets(self) -> dict[str, object]:
        """Return per-suite object assets keyed by their scene names."""
        ...

    @abstractmethod
    def rigid_object_keys(self) -> list[str]:
        """Return the scene names of every rigid object in this suite."""
        ...

    @abstractmethod
    def articulation_keys(self) -> list[str]:
        """Return the scene names of every task articulation in this suite."""
        ...

    @property
    @abstractmethod
    def primary_object_key(self) -> str:
        """Scene name of the object the reward shaping manipulates."""
        ...

    @property
    @abstractmethod
    def target_object_key(self) -> str:
        """Scene name of the object the primary object should be brought to."""
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _table_key(self) -> str:
        """Return the scene name of the suite's main kitchen table (``{name}_main_table``)."""
        return f"{self.name}_main_table"

    # ------------------------------------------------------------------
    # Scene assets
    # ------------------------------------------------------------------

    def fixture_assets(self) -> dict[str, object]:
        """Return the workspace fixtures for this task.

        Defaults to a single kitchen table; suites on other workspaces (floor,
        living room, study) override this.
        """
        return {self._table_key(): kitchen_table(self._table_key())}

    def scene_assets(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        """Return this task's scene assets: workspace fixtures plus the per-suite objects."""
        assets: dict[str, object] = dict(self.fixture_assets())
        assets.update(self.object_assets())
        return assets

    # ------------------------------------------------------------------
    # Commands (sparse reward — no goal command stream)
    # ------------------------------------------------------------------

    def command_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        """Contribute no command terms: the sparse-reward LIBERO tasks use no goal command stream."""
        return {}

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def task_obs_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, ObsTerm]:
        """Return one ``{key}_pos`` observation term per rigid object (position in robot frame [m])."""
        terms: dict[str, ObsTerm] = {}
        for key in self.rigid_object_keys():
            terms[f"{key}_pos"] = ObsTerm(
                func=mdp.object_pos_in_robot_frame,
                params={
                    "robot_cfg": SceneEntityCfg(_ROBOT_ASSET, selector=group),
                    "object_cfg": SceneEntityCfg(key, selector=group),
                },
            )
        return terms

    def scatter_obs_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, tuple[int | None, TermCfg]]:
        """Contribute no scattered (per-env-slice) observation terms for this task."""
        return {}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _success_params(self, group: str) -> dict[str, object]:
        """Shared params for the group's success reward / termination."""
        return {
            "object_cfg": SceneEntityCfg(self.primary_object_key, selector=group),
            "target_object_cfg": SceneEntityCfg(self.target_object_key, selector=group),
            "xy_threshold": self.success_xy_threshold,
            "height_threshold": self.success_height_threshold,
            "speed_threshold": self.success_speed_threshold,
        }

    def _success_reward_term(self, group: str) -> RewTerm:
        """Sparse task-success bonus (weight :attr:`success_weight`)."""
        return RewTerm(
            func=mdp.object_reached_target_bonus,
            weight=self.success_weight,
            params=self._success_params(group),
        )

    def _shaping_reward_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, RewTerm]:
        """Dense reach → lift → place shaping (demo-free, goal-derived)."""
        primary_cfg = SceneEntityCfg(self.primary_object_key, selector=group)
        target_cfg = SceneEntityCfg(self.target_object_key, selector=group)
        ee_frame_cfg = SceneEntityCfg(_EE_FRAME_ASSET, selector=group)
        return {
            f"{group}_reach_object": RewTerm(
                func=mdp.object_ee_distance,
                weight=self.reach_weight,
                params={"std": self.reach_std, "object_cfg": primary_cfg, "ee_frame_cfg": ee_frame_cfg},
            ),
            f"{group}_lift_object": RewTerm(
                func=mdp.object_is_lifted,
                weight=self.lift_weight,
                params={"minimal_height": self.lift_height, "object_cfg": primary_cfg},
            ),
            f"{group}_place_object": RewTerm(
                func=mdp.object_to_object_distance,
                weight=self.place_weight,
                params={"std": self.place_std, "object_cfg": primary_cfg, "target_object_cfg": target_cfg},
            ),
        }

    def _goal_reward_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, RewTerm]:
        """Sparse ``goal`` reward: success bonus only (plus global penalties)."""
        return {f"{group}_task_success": self._success_reward_term(group)}

    def _metaworld_dense_reward_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, RewTerm]:
        """Demo-free dense reward: reach/lift/place shaping + success bonus."""
        terms = self._shaping_reward_terms(group, robot)
        terms[f"{group}_task_success"] = self._success_reward_term(group)
        return terms

    def _world_state_tracking_reward_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, RewTerm]:
        """Dense demo/trajectory-tracking reward (gated).

        The reference implementation tracks a per-step demo trajectory: EE
        pose + gripper residuals and object/articulation pose errors in the
        robot base frame, driven by a demo command stream sourced from replayed
        HDF5 demonstrations.  ``isaaclab_contrib`` ships no such demo/reference
        source, so the dense tracking terms cannot be built without fabricating
        target data.  They are therefore gated off (a warning is emitted once)
        and only the sparse success bonus is kept, so the mode still constructs
        and trains.  Wire a per-step reference command stream (EE/gripper/object
        target poses) to enable the full tracking reward.
        """
        global _WORLD_STATE_TRACKING_WARNED
        if not _WORLD_STATE_TRACKING_WARNED:
            warnings.warn(
                "LIBERO reward mode 'world_state_tracking_reward' needs a per-step demo reference trajectory "
                "(EE pose, gripper, and object/articulation target poses in the robot base frame, "
                "e.g. from replayed HDF5 demonstrations). No such source is available in "
                "isaaclab_contrib, so the dense tracking terms are disabled and only the sparse "
                "task-success bonus is used. Set LIBERO_REWARD_MODE=metaworld_dense for a demo-free "
                "dense reward.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WORLD_STATE_TRACKING_WARNED = True
        return {f"{group}_task_success": self._success_reward_term(group)}

    def reward_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, RewTerm]:
        """Return this group's reward terms for the resolved :attr:`reward_mode`.

        Dispatches to the ``goal`` / ``metaworld_dense`` / ``world_state_tracking_reward``
        builders; empty when the task has no rigid primary object.
        """
        # Relational / articulation goals (open drawer, turn on stove) have no
        # manipulated rigid object (``primary_object_key is None``); every task
        # reward here is keyed on a rigid primary object, so contribute no
        # task-specific reward for them (only the global penalties apply). This
        # mirrors the harvest/combined path, which skips ``primary_proto is None``
        # tasks from its per-env reward gather.
        if self.primary_object_key is None:
            return {}
        mode = resolve_reward_mode(self.reward_mode)
        if mode == "metaworld_dense":
            return self._metaworld_dense_reward_terms(group, robot)
        if mode == "world_state_tracking_reward":
            return self._world_state_tracking_reward_terms(group, robot)
        return self._goal_reward_terms(group, robot)

    # ------------------------------------------------------------------
    # Terminations
    # ------------------------------------------------------------------

    def termination_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, DoneTerm]:
        """Return the group's object-dropped and task-success terminations.

        Empty when the task has no rigid primary object (time-out only).
        """
        # Both terminations track the rigid primary object; a relational goal
        # (``primary_object_key is None``) has none, so it terminates only on
        # time-out. Consistent with the harvest path (no primary -> no drop /
        # success proxy for that env).
        if self.primary_object_key is None:
            return {}
        return {
            f"{group}_object_dropped": DoneTerm(
                func=mdp.object_height_below_minimum,
                params={
                    "minimum_height": self.drop_height,
                    "object_cfg": SceneEntityCfg(self.primary_object_key, selector=group),
                },
            ),
            f"{group}_task_success": DoneTerm(
                func=mdp.object_reached_target,
                params=self._success_params(group),
            ),
        }

    # ------------------------------------------------------------------
    # Reset events
    # ------------------------------------------------------------------

    def reset_events(self, group: str, robot: RobotModuleCfg) -> dict[str, EventTerm]:
        """Return the group's reset events.

        Resets every rigid object / articulation to its default pose, then applies a
        uniform :attr:`object_spawn_range` jitter to the primary object (if any).
        """
        object_cfgs = [SceneEntityCfg(key, selector=group) for key in self.rigid_object_keys()]
        object_cfgs += [SceneEntityCfg(key, selector=group) for key in self.articulation_keys()]
        events = {
            f"{group}_reset_objects_default": EventTerm(
                func=mdp.reset_to_default,
                mode="reset",
                params={"asset_cfgs": object_cfgs},
            ),
        }
        # Only jitter a primary object when the task has one. Relational /
        # articulation goals (``primary_object_key is None``) are fully reset by
        # ``reset_to_default`` above; adding an object-uniform reset with a
        # ``None`` object would build ``SceneEntityCfg(name=None)`` and raise a
        # ``KeyError`` at reset time.
        if self.primary_object_key is not None:
            events[f"{group}_reset_primary_uniform"] = EventTerm(
                func=mdp.reset_object_uniform,
                mode="reset",
                params={
                    "pose_range": self.object_spawn_range,
                    "velocity_range": {},
                    "object_cfg": SceneEntityCfg(self.primary_object_key, selector=group),
                },
            )
        return events
