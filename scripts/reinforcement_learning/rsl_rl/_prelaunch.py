# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit launch-order helper shared by the RSL-RL train/play entry points."""

from __future__ import annotations

import argparse


def prelaunch_kit_before_cfg(args_cli: argparse.Namespace, argv: list[str]):
    """Launch Isaac Sim/Kit before the task config is built, when Kit will be used.

    Building the LIBERO DGPO configs harvests every task and assembles the demo
    command config, which loads native numeric libraries (``libgomp``/``libtorch``
    and friends) into the base linker namespace. If that heavy work runs before
    carb creates its private ``dlmopen`` namespace, Kit aborts with
    ``_dl_find_dso_for_object: Assertion 'ns == l->l_ns'``. Launching Kit first
    establishes the namespace so the config build is safe;
    :func:`~isaaclab.app.launch_simulation` then becomes a no-op relaunch via its
    ``has_kit()`` guard.

    The decision uses only the CLI (the config is not built yet). Kit-based
    backends (PhysX/ovphysx) and the Kit visualizer need Isaac Sim, while a Newton
    backend without a Kit visualizer is kitless; only genuinely kitless runs skip
    the early launch.

    Args:
        args_cli: Parsed command-line arguments.
        argv: Raw command-line tokens used to sniff backend/preset selections.

    Returns:
        The :class:`~isaaclab.app.AppLauncher` when Kit was launched, else ``None``.
    """
    from isaaclab.utils import has_kit

    if has_kit():
        return None

    visualizer = getattr(args_cli, "visualizer", None)
    visualizer_types: set[str] = set()
    if visualizer:
        parts = visualizer.split(",") if isinstance(visualizer, str) else visualizer
        visualizer_types = {str(v).strip().lower() for v in parts if str(v).strip()}

    kit_requested = "kit" in visualizer_types
    livestream_on = int(getattr(args_cli, "livestream", -1) or -1) >= 0

    tokens = " ".join(argv).lower()
    selects_newton = "newton" in tokens
    selects_kit_backend = "physx" in tokens or "ovphysx" in tokens
    kitless_run = (
        not kit_requested
        and not livestream_on
        and selects_newton
        and not selects_kit_backend
        and visualizer_types <= {"newton", "none"}
    )
    if kitless_run:
        return None

    from isaaclab.app import AppLauncher

    return AppLauncher(args_cli)
