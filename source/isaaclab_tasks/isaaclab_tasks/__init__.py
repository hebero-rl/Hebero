# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task utilities for the Hebero extract of Isaac Lab.

This is a slimmed copy of the upstream ``isaaclab_tasks`` package. Only the
``utils`` sub-package (hydra helpers, config parsing, and the preset CLI) is
shipped here, because the RSL-RL entry scripts depend solely on those utilities.
The upstream ``core`` / ``direct`` / ``manager_based`` / ``contrib`` task suites
are intentionally omitted -- the LIBERO environments register themselves through
:mod:`isaaclab_contrib.tasks`. Consequently the eager ``import_packages`` gym
registration performed by the upstream ``__init__`` is removed below.
"""

import importlib.metadata
import os
import tomllib

ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

_ext_toml = os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml")
if os.path.exists(_ext_toml):
    with open(_ext_toml, "rb") as _f:
        ISAACLAB_TASKS_METADATA = tomllib.load(_f)
else:
    ISAACLAB_TASKS_METADATA = {}
"""Extension metadata dictionary parsed from the extension.toml file."""

try:
    __version__ = importlib.metadata.version("isaaclab_tasks")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# NOTE: Upstream this module eagerly walks and imports every task suite via
# ``utils.import_packages`` to run their ``gym.register`` side effects. This
# extract ships no such suites (only ``utils``), so that registration step is
# omitted. LIBERO environments are registered by importing
# :mod:`isaaclab_contrib.tasks`.
