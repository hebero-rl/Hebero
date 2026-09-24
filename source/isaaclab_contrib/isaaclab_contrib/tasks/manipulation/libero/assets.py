# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared asset definitions for the LIBERO manipulation suites.

The two LIBERO suites (``libero_spatial`` and ``libero_goal``) share the same
Franka Panda robot and the same physics presets for their manipulable objects.
Centralising these here avoids duplicating the ~50-line robot config across the
scene configs, the robot module, and the task modules.

USD assets are loaded from the directory pointed to by the
``LIBERO_ASSETS_DATA_DIR`` environment variable.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from .. import settings

if TYPE_CHECKING:
    from pxr import Usd


def libero_assets_dir() -> str:
    """Return the LIBERO USD asset directory from ``LIBERO_ASSETS_DATA_DIR``.

    Returns:
        The asset root path, or an empty string when the variable is unset.
    """
    return settings.text("LIBERO_ASSETS_DATA_DIR")


def libero_config_dir() -> str:
    """Return the LIBERO per-task JSON config directory.

    Uses ``LIBERO_CONFIG_DIR`` when set; otherwise falls back to the ``config``
    directory sitting next to the USD asset directory (``LIBERO_ASSETS_DATA_DIR``
    points at ``.../libero/USD``, so the configs live in ``.../libero/config``).

    Returns:
        The config directory path, or an empty string when neither the config
        variable nor the asset variable is set.
    """
    explicit = settings.text("LIBERO_CONFIG_DIR")
    if explicit:
        return explicit
    assets = settings.text("LIBERO_ASSETS_DATA_DIR")
    if assets:
        return os.path.join(os.path.dirname(assets.rstrip("/")), "config")
    return ""


# ---------------------------------------------------------------------------
# Shared physics properties for manipulable rigid objects
# ---------------------------------------------------------------------------

OBJECT_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)
"""Rigid-body solver properties shared by all LIBERO manipulable objects."""

OBJECT_RECEPTACLE_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
    kinematic_enabled=True,
)
"""Same solver settings, but kinematic: the object cannot be pushed.

Applied to receptacles -- the "target" side of a relationship goal (plate, basket,
caddy, tray).  Without it the sparse ``task_success`` term is trivially hackable:
the predicate for "bowl on plate" is satisfied just as well by shoving the plate
under the bowl as by picking the bowl up, and shoving is far easier to discover.
The accumulated ``*_goal`` payouts used to mask this because they only paid along
a demo-shaped trajectory; replacing them with a single sparse payout exposed it.

Articulated targets (drawers, cabinets, stoves) take the ``ArticulationCfg`` branch
and are unaffected, so doors and drawers still open.

Spawn these through :func:`spawn_receptacle_from_usd`, which strips the per-body CCD
flag that PhysX refuses to combine with a kinematic body.
"""

LIBERO_COMPLIANT_MATERIAL_CFG = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.1,
    dynamic_friction=0.2,
    restitution=0.10,
    friction_combine_mode="average",
    restitution_combine_mode="average",
    compliant_contact_stiffness=10.0,
    compliant_contact_damping=1.0,
)
"""Opt-in scene material approximating MuJoCo LIBERO's soft contact model.

Off unless ``LIBERO_COMPLIANT_CONTACT=1``: it changes contact dynamics for every
object in the benchmark, so it is a property of the run, not a silent default.

MuJoCo LIBERO authors most scene and object geoms with ``solref="0.001 1"`` /
``solimp="0.998 0.998 0.001"``, i.e. a contact that gives slightly instead of
resolving rigidly. PhysX's default rigid contact instead spikes the contact force
whenever the gripper closes on an object, which shows up as jittering objects and
exploding fingertip forces. Setting a finite ``compliant_contact_stiffness`` turns
the contact into an implicit spring, which is the closest PhysX analogue.

When enabled it is bound as ``sim.physics_material``, i.e. the scene's *fallback*
material: any prim that spawns its own ``physics_material`` (e.g. a per-object
friction override) still wins.

Tuning guidance:

* ``stiffness`` -- higher is closer to rigid; lower is softer / more forgiving.
* ``damping`` -- energy dissipation, roughly ``2 * zeta * sqrt(stiffness * mass)``.
* friction -- these values are carried over verbatim from the playground repo,
  where they were tuned empirically; note they are *not* MuJoCo's ``0.95`` sliding
  friction, and static < dynamic here.

PhysX-only: the compliant-contact fields have no Newton consumer and are ignored
under ``presets=newton``.
"""


@sim_utils.clone
def spawn_receptacle_from_usd(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a kinematic receptacle, clearing any per-body CCD flag the asset authors.

    Some LIBERO assets ship ``physxRigidBody:enableCCD = True`` (``basket``, the
    receptacle of 13 tasks, and ``white_yellow_mug``).  Combined with the
    ``kinematic_enabled`` of :data:`OBJECT_RECEPTACLE_RIGID_PROPS`, PhysX errors once
    per body per env while parsing the stage::

        PxRigidBody::setRigidBodyFlag(): kinematic bodies with CCD enabled are not
        supported! CCD will be ignored.

    Nothing is lost by clearing it: CCD is meaningless on a body the solver never
    integrates, and scene-level CCD is off in this project anyway
    (``PhysxManagerCfg.enable_ccd`` defaults to False and is force-disabled on GPU).

    The override is authored inside the ``@clone`` body, i.e. on the prototype prim
    before it is copied to the other envs, so every env gets it.

    Args:
        prim_path: The prim path (or regex) to spawn the asset at.
        cfg: The USD spawner config.
        translation: Translation w.r.t. the parent prim, or None to keep the USD's.
        orientation: Orientation quaternion w.r.t. the parent prim, or None to keep the USD's.
        **kwargs: Forwarded by the ``clone`` decorator; unused here.

    Returns:
        The spawned source prim.
    """
    from pxr import Usd  # noqa: PLC0415

    # ``spawn_from_usd`` is itself ``@clone``-decorated; call the undecorated function so
    # the CCD override lands on the prototype rather than after the clone.
    prim = sim_utils.spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation)
    for body in Usd.PrimRange(prim):
        ccd_attr = body.GetAttribute("physxRigidBody:enableCCD")
        if ccd_attr and ccd_attr.IsAuthored() and ccd_attr.Get():
            ccd_attr.Set(False)
    return prim


# ---------------------------------------------------------------------------
# Franka Panda robot (LIBERO PD gains + ready pose)
# ---------------------------------------------------------------------------

# The Isaac 6.0 asset bucket does not ship panda_instanceable.usd (404); pin this
# one asset to the 5.0 tree, which still hosts it. Overridable via env var.
_FRANKA_NUCLEUS_DIR = os.getenv(
    "LIBERO_FRANKA_NUCLEUS_DIR",
    ISAACLAB_NUCLEUS_DIR.replace("/Isaac/6.0/", "/Isaac/5.0/"),
)

LIBERO_FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_FRANKA_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.66, 0.0, 0.912),
        joint_pos={
            "panda_joint1": -0.019882839432642387,
            "panda_joint2": -0.18734066496238144,
            "panda_joint3": 0.0076694004538321505,
            "panda_joint4": -2.4034025985475256,
            "panda_joint5": 0.004964681607500244,
            "panda_joint6": 2.2453365042123963,
            "panda_joint7": 0.7948478983158621,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=8000.0,
            damping=800.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=8000.0,
            damping=800.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Franka Panda articulation with LIBERO-specific PD gains and ready-pose joints."""
