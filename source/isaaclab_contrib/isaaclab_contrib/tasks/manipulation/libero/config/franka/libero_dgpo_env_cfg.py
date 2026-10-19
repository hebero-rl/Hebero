# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym entry-point classes for the DGPO OSC LIBERO env (harvest + cloner).

Importing this module is cheap; cfg classes are built on first attribute access
(requires ``LIBERO_CONFIG_DIR`` / ``LIBERO_ASSETS_DATA_DIR``). The factories live
in :mod:`~...envs.dgpo_env_cfg` so layout tests can import helpers without
harvesting suites.

Gym registration uses ``Dgpo`` names (train: All only; play: All + suite
subsets). ``LiberoAllCompatOscEnvCfg`` remains a thin class alias for older
imports / tests.
"""

from __future__ import annotations

from typing import Any

from ...dgpo_layout import DGPO_ABC_HARVEST_SUITES
from ...envs.dgpo_env_cfg import make_libero_dgpo_env_cfg, make_libero_dgpo_play_cfg

# Single-suite / combo tuples (relative order matches DGPO: long → object → spatial → goal).
_LONG: tuple[tuple[str, str], ...] = (("libero_long", "long"),)
_OBJECT: tuple[tuple[str, str], ...] = (("libero_object", "object"),)
_SPATIAL: tuple[tuple[str, str], ...] = (("libero_spatial", "spatial"),)
_GOAL: tuple[tuple[str, str], ...] = (("libero_goal", "goal"),)
_SPATIAL_GOAL: tuple[tuple[str, str], ...] = _SPATIAL + _GOAL
_OBJECT_LONG: tuple[tuple[str, str], ...] = _LONG + _OBJECT

# name → (suites, num_envs | None for play)
_CFG_SPECS: dict[str, tuple[tuple[tuple[str, str], ...], int | None]] = {
    "LiberoAllDgpoOscEnvCfg": (DGPO_ABC_HARVEST_SUITES, 2560),
    "LiberoAllDgpoOscEnvCfg_PLAY": (DGPO_ABC_HARVEST_SUITES, None),
    "LiberoLongDgpoOscEnvCfg": (_LONG, 640),
    "LiberoLongDgpoOscEnvCfg_PLAY": (_LONG, None),
    "LiberoObjectDgpoOscEnvCfg": (_OBJECT, 640),
    "LiberoObjectDgpoOscEnvCfg_PLAY": (_OBJECT, None),
    "LiberoSpatialDgpoOscEnvCfg": (_SPATIAL, 640),
    "LiberoSpatialDgpoOscEnvCfg_PLAY": (_SPATIAL, None),
    "LiberoGoalDgpoOscEnvCfg": (_GOAL, 640),
    "LiberoGoalDgpoOscEnvCfg_PLAY": (_GOAL, None),
    "LiberoSpatialGoalDgpoOscEnvCfg": (_SPATIAL_GOAL, 1280),
    "LiberoSpatialGoalDgpoOscEnvCfg_PLAY": (_SPATIAL_GOAL, None),
    "LiberoObjectLongDgpoOscEnvCfg": (_OBJECT_LONG, 1280),
    "LiberoObjectLongDgpoOscEnvCfg_PLAY": (_OBJECT_LONG, None),
}

# Compat aliases → primary All names.
_ALIASES: dict[str, str] = {
    "LiberoAllCompatOscEnvCfg": "LiberoAllDgpoOscEnvCfg",
    "LiberoAllCompatOscEnvCfg_PLAY": "LiberoAllDgpoOscEnvCfg_PLAY",
}

__all__ = list(_CFG_SPECS) + list(_ALIASES)


def __getattr__(name: str) -> Any:
    """Lazily build the requested env cfg class on first access."""
    resolved = _ALIASES.get(name, name)
    if resolved not in _CFG_SPECS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    suites, num_envs = _CFG_SPECS[resolved]
    if num_envs is None:
        cfg_cls = make_libero_dgpo_play_cfg(suites=suites)
    else:
        cfg_cls = make_libero_dgpo_env_cfg(suites=suites, num_envs=num_envs)
    globals()[resolved] = cfg_cls
    if name != resolved:
        globals()[name] = cfg_cls
    return cfg_cls


def __dir__() -> list[str]:
    """List module attributes including lazily-built cfg classes not yet materialised."""
    return sorted(set(globals()) | set(__all__))
