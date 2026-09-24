# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka robot module using Operational Space Control (OSC) for DGPO training.

DGPO / Adaptive-BC expects OSC ``pose_rel`` (6-D) + binary gripper (1-D) with arm
actuator stiffness/damping zeroed so OSC owns torque control.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    OperationalSpaceControllerActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg as TermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.robots._base import RobotModuleCfg

from ..assets import LIBERO_FRANKA_PANDA_CFG
from ..dgpo_layout import (
    DGPO_ABC_GRIPPER_CLOSE,
    DGPO_ABC_GRIPPER_OPEN,
    DGPO_ABC_OSC_MOTION_STIFFNESS,
    DGPO_ABC_OSC_NULLSPACE_STIFFNESS,
    DGPO_ABC_OSC_ORIENTATION_SCALE,
    DGPO_ABC_OSC_POSITION_SCALE,
)


@dataclass
class LiberoFrankaOscRobotCfg(RobotModuleCfg):
    """Franka with OSC pose_rel arm control for the DGPO harvest training path."""

    osc_type: str = "pose_rel"
    """OSC target type (``pose_rel`` for DGPO layout)."""

    position_scale: float = DGPO_ABC_OSC_POSITION_SCALE
    """Position command scale for relative OSC [--]."""

    orientation_scale: float = DGPO_ABC_OSC_ORIENTATION_SCALE
    """Orientation command scale for relative OSC [--]."""

    @property
    def name(self) -> str:
        """Return the robot module identifier (``"franka"``)."""
        return "franka"

    @property
    def all_joint_names(self) -> list[str]:
        """Return joint-name regexes covering the arm and gripper (``panda_joint.*``, ``panda_finger.*``)."""
        return ["panda_joint.*", "panda_finger.*"]

    def scene_assets(self, group: str) -> dict[str, object]:
        """Return the shared Franka scene assets.

        Contains the ``franka_robot`` articulation (arm PD zeroed so OSC owns torque),
        the ``franka_ee_frame`` FrameTransformer with three target frames (``ee_tcp``
        plus the two fingertips).
        that feeds the 24-d gripper_force proprio history.
        """
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = f"/Visuals/{group}_FrankaEEFrame"

        robot_cfg = LIBERO_FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # OSC owns arm torque; zero PD on shoulder/forearm like playground OscPose cfg.
        # Replace actuator cfgs (do not mutate the shared module-level defaults).
        actuators = dict(robot_cfg.actuators)
        for arm_group in ("panda_shoulder", "panda_forearm"):
            actuators[arm_group] = actuators[arm_group].replace(stiffness=0.0, damping=0.0)
        robot_cfg.actuators = actuators

        return {
            "franka_robot": robot_cfg,
            "franka_ee_frame": FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
                debug_vis=False,
                visualizer_cfg=marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                        name="ee_tcp",
                        offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
                    ),
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                        name="tool_leftfinger",
                        offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                    ),
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                        name="tool_rightfinger",
                        offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                    ),
                ],
            ),
        }

    def action_specs(self, group: str | None = None) -> dict[str, tuple[int, object]]:
        """Return the DGPO action layout: OSC ``pose_rel`` arm (6-D) + binary gripper (1-D)."""
        arm = OperationalSpaceControllerActionCfg(
            asset_name="franka_robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=[self.osc_type],
                impedance_mode="fixed",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=tuple(DGPO_ABC_OSC_MOTION_STIFFNESS),
                motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                nullspace_control="position",
                nullspace_stiffness=DGPO_ABC_OSC_NULLSPACE_STIFFNESS,
            ),
            nullspace_joint_pos_target="default",
            position_scale=self.position_scale,
            orientation_scale=self.orientation_scale,
            body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034)),
        )
        # Plain binary gripper; policy outputs use BinaryJointPositionAction sign
        # (open >= 0). The training env factory swaps this for the SR-gated
        # CurriculumBinaryJointPositionActionCfg when GRIPPER_CURRICULUM is on
        # (see manipulation/utils/gripper_curriculum.py); play/eval keep the plain action.
        gripper = BinaryJointPositionActionCfg(
            asset_name="franka_robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": DGPO_ABC_GRIPPER_OPEN},
            close_command_expr={"panda_finger_.*": DGPO_ABC_GRIPPER_CLOSE},
        )
        return {
            "arm": (6, arm),
            "gripper": (1, gripper),
        }

    def scatter_obs_terms(self, group: str) -> dict[str, tuple[int | None, TermCfg]]:
        """Contribute no scattered observation terms; DGPO obs groups are installed by the env factory."""
        # DGPO obs groups are installed by the env factory; robot scatter terms unused.
        return {}

    def reset_events(self, group: str) -> dict[str, EventTerm]:
        """Return the group's robot reset event: restore the Franka to its default joint state and targets."""
        return {
            f"{group}_franka_reset_to_default": EventTerm(
                func=mdp.reset_to_default,
                mode="reset",
                params={
                    "reset_joint_targets": True,
                    "asset_cfgs": [SceneEntityCfg("franka_robot", selector=group)],
                },
            ),
        }


FRANKA_OSC = LiberoFrankaOscRobotCfg()
"""Franka Panda with OSC pose_rel arm control (6-D) and a binary gripper (1-D)."""
