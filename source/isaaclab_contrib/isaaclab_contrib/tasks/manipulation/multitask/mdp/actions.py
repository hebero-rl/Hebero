# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scattered action terms for multi-task environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.scene.env_view_index import filter_to_group

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.scene.env_view_index import EnvToViewMap

    from . import actions_cfg


class ScatteredActionTerm(ActionTerm):
    """Scattters multiple action terms that share the same action dimension.

    The policy outputs a single set of actions (e.g. 6 IK dims or 1 gripper
    dim) and the scattered term broadcasts them to every sub-term.  Each
    sub-term applies to its own asset/group independently.

    All terms must have the same ``action_dim``; unequal dimensions raise
    :class:`ValueError` at init time.
    """

    cfg: actions_cfg.ScatteredActionTermCfg

    def __init__(self, cfg: actions_cfg.ScatteredActionTermCfg, env: ManagerBasedEnv):
        """Instantiate one sub-term per registration and validate a shared ``action_dim``.

        Each sub-term is paired with its clone group's env-to-view map so it
        only ever touches its own group's envs.  Sub-terms whose asset has no
        selector or whose group is disabled (``weight=0`` → empty ``env_ids``)
        are skipped.  All active sub-terms must agree on ``action_dim``;
        ``cfg.dim`` is the fallback when no sub-term is active.
        """
        ManagerTermBase.__init__(self, cfg, env)
        self._IO_descriptor = GenericActionIODescriptor()
        self._export_IO_descriptor = True
        self._debug_vis_handle = None

        selector = env.scene.selector
        self._sub_terms: list[tuple[EnvToViewMap, ActionTerm]] = []
        for term_cfg in cfg.terms:
            asset_selectors = selector.assets.get(term_cfg.asset_name)
            if not asset_selectors:
                continue
            env_to_view_map = selector.get(list(asset_selectors), asset=term_cfg.asset_name)
            # Skip disabled combinations (weight=0): empty tensor means no active envs.
            if isinstance(env_to_view_map.env_ids, torch.Tensor) and env_to_view_map.env_ids.numel() == 0:
                continue
            term = term_cfg.class_type(term_cfg, env)
            self._sub_terms.append((env_to_view_map, term))

        dims = {t.action_dim for _, t in self._sub_terms}
        if len(dims) == 0:
            if cfg.dim is not None:
                self._child_action_dim = cfg.dim
            else:
                raise ValueError("No active sub-terms and no fallback 'dim' on cfg.")
        elif len(dims) != 1:
            raise ValueError(f"All terms must have the same action_dim. Got: {sorted(dims)}")
        else:
            self._child_action_dim = next(iter(dims))

        self._raw_buf = torch.zeros(env.num_envs, self._child_action_dim, device=self.device)
        self._proc_buf = torch.zeros_like(self._raw_buf)

    @property
    def action_dim(self) -> int:
        """Action dimension shared by all sub-terms (the policy's column width)."""
        return self._child_action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw actions scattered from each sub-term into a global buffer.

        Each sub-term writes only its clone group's rows; envs outside every
        group stay zero.  Returns shape ``(num_envs, action_dim)``.
        """
        self._raw_buf.zero_()
        for env_to_view_map, term in self._sub_terms:
            self._raw_buf[env_to_view_map.env_ids] = term.raw_actions
        return self._raw_buf

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed actions scattered from each sub-term into a global buffer.

        Each sub-term writes only its clone group's rows; envs outside every
        group stay zero.  Returns shape ``(num_envs, action_dim)``.
        """
        self._proc_buf.zero_()
        for env_to_view_map, term in self._sub_terms:
            self._proc_buf[env_to_view_map.env_ids] = term.processed_actions
        return self._proc_buf

    def process_actions(self, actions: torch.Tensor):
        """Slice the shared ``(num_envs, action_dim)`` column per clone group.

        Every sub-term receives only its own group's rows, so the same policy
        output column drives each group's asset independently.
        """
        for env_to_view_map, term in self._sub_terms:
            term.process_actions(actions[env_to_view_map.env_ids])

    def apply_actions(self):
        """Apply each sub-term's processed actions to its own group's asset."""
        for _, term in self._sub_terms:
            term.apply_actions()

    def reset(self, env_ids=None):
        """Reset sub-terms, forwarding only the env ids inside each sub-term's group.

        Global env ids are translated to per-group view ids via
        ``filter_to_group``; sub-terms with no matching envs are skipped.
        ``env_ids=None`` resets every sub-term fully.
        """
        if env_ids is None:
            for _, term in self._sub_terms:
                term.reset(None)
            return
        for env_to_view_map, term in self._sub_terms:
            view_ids, filtered_env_ids = filter_to_group(env_to_view_map.layout, env_ids)
            if filtered_env_ids.numel() > 0:
                term.reset(view_ids)
