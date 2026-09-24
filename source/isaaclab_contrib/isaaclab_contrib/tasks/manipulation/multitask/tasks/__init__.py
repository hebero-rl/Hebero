# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task module definitions for multitask environments.

This is the vendored subset shipped with Hebero: only the abstract
:class:`TaskModuleCfg` base is retained, since LIBERO provides its own task
modules (:mod:`...libero.tasks`). The upstream concrete tasks
(reach / lift / cabinet) are not part of this extract.
"""

from ._base import TaskModuleCfg
