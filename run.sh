#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Lightweight launcher for Hebero.
#
# The standalone repo has no ``./isaaclab.sh`` wrapper. This script simply puts
# the vendored ``source/*`` package roots on PYTHONPATH and forwards all
# arguments to ``python``. It expects an Isaac Sim / IsaacLab Python environment
# (isaacsim, torch, gymnasium, rsl-rl-lib, warp, ...) to already be active.
#
# Usage:
#   LIBERO_ASSETS_DATA_DIR=<path>/libero/USD \
#     ./run.sh scripts/reinforcement_learning/rsl_rl/train.py --task ... presets=physx
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${REPO_ROOT}/source"

PKG_ROOTS=(
  "${SRC}/isaaclab"
  "${SRC}/isaaclab_rl"
  "${SRC}/isaaclab_tasks"
  "${SRC}/isaaclab_contrib"
  "${SRC}/isaaclab_newton"
  "${SRC}/isaaclab_physx"
  "${SRC}/isaaclab_ovphysx"
)

JOINED="$(IFS=:; echo "${PKG_ROOTS[*]}")"
export PYTHONPATH="${JOINED}${PYTHONPATH:+:${PYTHONPATH}}"

exec python "$@"
