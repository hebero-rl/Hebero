# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym entry-point classes for the vision (Theia) LIBERO DGPO env.

Mirrors :mod:`.libero_dgpo_env_cfg` -- same suites, same lazy build -- with cameras and
the frozen-Theia ``perception`` group installed on top; see
:mod:`~...envs.vision_cfg` for what changes in the observation layout.
"""

from __future__ import annotations

from typing import Any

from ...dgpo_layout import DGPO_ABC_HARVEST_SUITES
from ...envs.vision_cfg import make_libero_dgpo_vision_env_cfg, make_libero_dgpo_vision_play_cfg

_LONG: tuple[tuple[str, str], ...] = (("libero_long", "long"),)

# name → (suites, num_envs | None for play)
_CFG_SPECS: dict[str, tuple[tuple[tuple[str, str], ...], int | None]] = {
    # Rendering two cameras per env is the binding cost here, so the train default is a
    # quarter of the state-based env's 2560.  Override with ``--num_envs``.
    "LiberoAllDgpoOscVisionEnvCfg": (DGPO_ABC_HARVEST_SUITES, 640),
    "LiberoAllDgpoOscVisionEnvCfg_PLAY": (DGPO_ABC_HARVEST_SUITES, None),
    "LiberoLongDgpoOscVisionEnvCfg": (_LONG, 160),
    "LiberoLongDgpoOscVisionEnvCfg_PLAY": (_LONG, None),
}

__all__ = list(_CFG_SPECS)


def __getattr__(name: str) -> Any:
    """Lazily build the requested vision env cfg class on first access."""
    if name not in _CFG_SPECS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    suites, num_envs = _CFG_SPECS[name]
    if num_envs is None:
        cfg_cls = make_libero_dgpo_vision_play_cfg(suites=suites)
    else:
        cfg_cls = make_libero_dgpo_vision_env_cfg(suites=suites, num_envs=num_envs)
    globals()[name] = cfg_cls
    return cfg_cls


def __dir__() -> list[str]:
    """List module attributes including lazily-built cfg classes not yet materialised."""
    return sorted(set(globals()) | set(__all__))
