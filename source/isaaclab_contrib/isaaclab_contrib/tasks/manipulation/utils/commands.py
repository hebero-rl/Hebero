# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic command-view resolution on the stock :class:`~isaaclab.managers.CommandManager`.

The LIBERO demo command is ONE term (registered under its cfg attribute name,
e.g. ``libero_demo``) that exposes several semantic views (``ee_pose``,
``gripper_state``, ``joint_state``, ``joint_velocity``, ``source_action``) via
``get_command_view`` — one backend, one frame-cursor advance per step. MDP
terms keep addressing views by name (``command_name="ee_pose"``); these
resolvers map a view name onto the owning term so no custom (aliasing)
``CommandManager`` subclass is needed:

* a name registered directly on the manager resolves the stock way
  (plain commands);
* otherwise the manager's terms are scanned for one whose cfg lists the name
  in ``semantic_keys``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CommandTerm


def resolve_command_term(env: ManagerBasedRLEnv, command_name: str) -> CommandTerm:
    """Return the command term serving ``command_name`` (direct name or semantic view).

    Raises:
        KeyError: If no term is registered under the name and no term's
            ``semantic_keys`` contain it.
    """
    manager = env.command_manager
    if command_name in manager.active_terms:
        return manager.get_term(command_name)
    for term_name in manager.active_terms:
        term = manager.get_term(term_name)
        keys = getattr(getattr(term, "cfg", None), "semantic_keys", None)
        if keys and command_name in keys:
            return term
    raise KeyError(
        f"Command '{command_name}' is neither a registered term nor a semantic view of one"
        f" (active terms: {list(manager.active_terms)})."
    )


def resolve_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return the command tensor for ``command_name`` (direct name or semantic view)."""
    manager = env.command_manager
    if command_name in manager.active_terms:
        return manager.get_command(command_name)
    term = resolve_command_term(env, command_name)
    return term.get_command_view(command_name)
