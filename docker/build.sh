#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Build the Hebero training image.
#
# Usage:
#   ./docker/build.sh [extra docker build args...]
#
# Note: the image is large (~40 GB) — the pip-installed Isaac Sim wheels alone
# are tens of GB. First build downloads everything and takes a while.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec docker build -f docker/Dockerfile -t hebero:latest "$@" .
