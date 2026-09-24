# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SR-gated gripper curriculum (playground ``GRIPPER_CURRICULUM`` port).

For tasks whose per-task success-rate EMA is below ``cfg.sr_threshold``, the
binary gripper action REPLAYS the demo gripper command (one column of the
``source_action`` demo command view) instead of the policy output; once the
task's SR EMA crosses the threshold, the gripper hands over to the policy.

The action term is self-contained: each ``process_actions`` (every step,
pre-physics) it reads :meth:`TaskSRTracker.replay_mask` through the documented
``env.task_sr_tracker`` handle maintained by the ``task_success_rate``
curriculum term (see :mod:`.task_sr`). No env subclass or per-step
mixin refresh is involved; without a tracker or with ``sr_threshold=None`` the
mask stays ``False`` (pure policy control), so eval/play configs simply omit
the curriculum term or the threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.utils.configclass import configclass

from .commands import resolve_command

if TYPE_CHECKING:
    from isaaclab.managers.action_manager import ActionTerm


# ---------------------------------------------------------------------------
# Action terms
# ---------------------------------------------------------------------------


class ReplayBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary gripper action that replays the demo command stream (ignores the policy)."""

    cfg: ReplayBinaryJointPositionActionCfg

    def __init__(self, cfg: ReplayBinaryJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._source_command = torch.zeros(self.num_envs, 1, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        """Ignore policy inputs and apply the replayed demo command to the gripper."""
        _ = actions
        gripper_command = self._read_source_command().unsqueeze(-1)
        self._source_command[:] = gripper_command
        super().process_actions(gripper_command)

    def _read_source_command(self) -> torch.Tensor:
        """Return the ``(num_envs,)`` demo gripper column named by the cfg."""
        if getattr(self._env, "command_manager", None) is None:
            raise RuntimeError(
                "Command manager is not available. Enable the demo command before using"
                f" {type(self).__name__}."
            )
        try:
            command = resolve_command(self._env, self.cfg.source_command_name)
        except KeyError as exc:
            raise RuntimeError(
                f"Command term '{self.cfg.source_command_name}' is not active. Wire the demo command"
                " (assembled demos) before enabling gripper replay."
            ) from exc
        if command.ndim != 2:
            raise ValueError(
                f"Command '{self.cfg.source_command_name}' must be a 2-D tensor; got shape {tuple(command.shape)}."
            )
        column = self.cfg.command_index
        if column < 0:
            column += command.shape[1]
        if column < 0 or column >= command.shape[1]:
            raise ValueError(
                f"Command index {self.cfg.command_index} is out of bounds for command"
                f" '{self.cfg.source_command_name}' with dimension {command.shape[1]}."
            )
        return command[:, column]

    @property
    def recordable_source_actions(self) -> torch.Tensor:
        """Return the replayed gripper command used for control."""
        return self._source_command


class CurriculumBinaryJointPositionAction(ReplayBinaryJointPositionAction):
    """Gripper action that replays demos for low-SR tasks and uses the policy otherwise.

    Per-env selection comes from ``env.task_sr_tracker.replay_mask(cfg.sr_threshold)``
    (tasks that have not finished ``min_episodes`` yet also replay, so a single
    fluke success cannot graduate a task). Without a tracker or a threshold the
    mask stays ``False`` → pure policy control.
    """

    cfg: CurriculumBinaryJointPositionActionCfg

    def __init__(self, cfg: CurriculumBinaryJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._policy_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._replay_mask = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
        self._step_count: int = 0

    def process_actions(self, actions: torch.Tensor):
        """Replay the demo gripper command until the curriculum promotes policy control."""
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        self._policy_actions[:] = actions

        self._step_count += 1
        replay_mask = self._read_replay_mask()
        source_command = self._read_source_command().unsqueeze(-1)
        if self.cfg.invert_source_command_sign:
            source_command = -source_command

        self._replay_mask[:] = replay_mask
        self._source_command[:] = source_command
        mixed_actions = torch.where(replay_mask, source_command, actions)
        BinaryJointPositionAction.process_actions(self, mixed_actions)

    def _read_replay_mask(self) -> torch.Tensor:
        """Return the per-env replay mask from the task-SR tracker (zeros when absent)."""
        threshold = self.cfg.sr_threshold
        tracker = getattr(self._env, "task_sr_tracker", None)
        if threshold is None or tracker is None:
            return self._replay_mask.zero_()
        max_steps = self.cfg.max_env_steps
        if max_steps is not None and self._step_count >= int(max_steps):
            # Expired: the policy owns the gripper for every env from here on.
            return self._replay_mask.zero_()
        return tracker.replay_mask(float(threshold)).view(self.num_envs, 1)

    @property
    def recordable_policy_actions(self) -> torch.Tensor:
        """Return the raw policy gripper action before curriculum mixing."""
        return self._policy_actions

    @property
    def recordable_replay_mask(self) -> torch.Tensor:
        """Return the per-env replay mask used by the gripper curriculum."""
        return self._replay_mask


# ---------------------------------------------------------------------------
# Action cfgs
# ---------------------------------------------------------------------------


@configclass
class ReplayBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Binary gripper action that replays commands from the demo command stream."""

    class_type: type[ActionTerm] = ReplayBinaryJointPositionAction
    source_command_name: str = "source_action"
    """Command term (or semantic view) whose buffer carries the demo action (gripper in one column)."""
    command_index: int = -1
    """Column of the command buffer holding the gripper action (default: last)."""


@configclass
class CurriculumBinaryJointPositionActionCfg(ReplayBinaryJointPositionActionCfg):
    """Binary gripper action that switches between policy control and demo replay."""

    class_type: type[ActionTerm] = CurriculumBinaryJointPositionAction
    invert_source_command_sign: bool = True
    """Negate the demo gripper column before applying it.

    ``BinaryJointPositionAction`` treats ``action >= 0`` as OPEN. Set ``True``
    when the demo stores the opposite sign (LIBERO assembled demos), ``False``
    when the demo already matches (+1 open / -1 close).
    """
    sr_threshold: float | None = None
    """Per-task SR EMA below which the gripper replays the demo command.

    ``None`` disables the curriculum (pure policy control). Requires the
    ``task_success_rate`` curriculum term (``env.task_sr_tracker``).
    """
    max_env_steps: int | None = None
    """Hard cap on how long the curriculum may stay engaged, in env steps.

    The SR threshold alone makes the override asymmetric across methods: a method
    whose SR climbs past the threshold graduates, while a method stuck below it
    keeps the demonstration gripper for the whole run, leaving a gripper-matching
    reward term action-independent. Set an integer to force graduation regardless
    of SR; ``None`` keeps the SR-only behaviour.
    """
