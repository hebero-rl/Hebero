#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Launch LIBERO training / evaluation inside the hebero container
# (headless). Mounts the LIBERO USD assets, the assembled demo dataset, and the
# logs dir; for eval it also mounts the checkpoint.
#
# Usage:
#   ./docker/launch.sh train [extra train.py args...]
#   CKPT=<host-path>/model_xxx.pt ./docker/launch.sh eval [extra eval args...]
#   ./docker/launch.sh shell                     # interactive bash in the image
#
# Examples:
#   ./docker/launch.sh train                                       # full 40-task run
#   MAX_ITERATIONS=5 LIBERO_NUM_ENVS=16 ./docker/launch.sh train   # smoke run
#   CKPT=logs/rsl_rl/libero_all_dgpo/<run>/model_4000.pt \
#     ./docker/launch.sh eval --num_envs 40 --max_episode_per_task 50
#
# Host data locations (override via env):
#   LIBERO_USD_DIR    — LIBERO USD assets (.../libero/USD)
#   LIBERO_DEMO_DIR   — assembled demo dataset (.../datasets/demos)
#   LOGS_DIR          — host dir that receives logs/checkpoints
#   CKPT              — host path to the checkpoint to evaluate (eval only)
#
# Behaviour knobs (override via env):
#   LIBERO_REWARD_MODE — goal | world_state_tracking_reward | metaworld_dense
#                        (default world_state_tracking_reward: DGPO dense demo tracking)
#   GPUS               — docker --gpus value.  Defaults to one device for a
#                        DISTRIBUTED=0 run and all for DISTRIBUTED=1; set e.g.
#                        GPUS='"device=3"' to pin a single-GPU run elsewhere.
#   MOUNT_SRC=1        — bind-mount the host source/ + scripts/ over the copies
#                        baked into the image (fast iteration without rebuild;
#                        remember the image itself stays stale until
#                        ./docker/build.sh is rerun)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Host-side data, mounted read-only below.  Defaults to the Hugging Face pull at
# <repo>/datasets (README §2.3); both must be absolute for `docker -v`.
LIBERO_USD_DIR="${LIBERO_USD_DIR:-$(pwd)/datasets/USD}"
LIBERO_DEMO_DIR="${LIBERO_DEMO_DIR:-$(pwd)/datasets/demos}"
LOGS_DIR="${LOGS_DIR:-$(pwd)/logs}"
IMAGE="${IMAGE:-hebero:latest}"

mode="${1:-train}"
shift || true

mkdir -p "${LOGS_DIR}"

# Allocate a TTY only when we have one (keeps the script usable from CI/cron).
TTY_ARGS=()
if [ -t 0 ]; then
    TTY_ARGS=(-it)
fi

# Which GPUs the container may see.  A single-process run needs exactly one:
# Isaac Sim creates a CUDA context on every visible device, so exposing all of
# them to a DISTRIBUTED=0 run parks a few hundred MB on each idle GPU and keeps
# them from other work.  Override for a single-GPU run on a specific device,
# e.g. GPUS='"device=3"'.
if [ -n "${GPUS:-}" ]; then
    GPU_ARG="${GPUS}"
elif [ "${DISTRIBUTED:-0}" = "1" ]; then
    GPU_ARG="all"
else
    GPU_ARG='"device=0"'
fi

DOCKER_ARGS=(
    --rm
    --gpus "${GPU_ARG}"
    --shm-size=8g
    -e VIZ_ARGS=--headless
    -e LIBERO_NUM_ENVS="${LIBERO_NUM_ENVS:-160}"
    -e MAX_ITERATIONS="${MAX_ITERATIONS:-4000}"
    # Task subset: default is all four suites (40 tasks). Set LIBERO_TASK to
    # Isaac-Hebero-Long-State-v0 with LIBERO_NUM_TASKS=10 for libero_10 alone.
    -e LIBERO_TASK="${LIBERO_TASK:-Isaac-Hebero-All-State-v0}"
    -e LIBERO_NUM_TASKS="${LIBERO_NUM_TASKS:-40}"
    -e LIBERO_EXPERIMENT_NAME="${LIBERO_EXPERIMENT_NAME:-}"
    # Algorithm entry point: rsl_rl{,_iw_abc,_dapg,_mtppo,_ppo_rfcl}_cfg_entry_point
    -e AGENT="${AGENT:-rsl_rl_cfg_entry_point}"
    # DISTRIBUTED=1 + NPROC_PER_NODE=<gpus> -> torchrun multi-GPU inside the container
    -e DISTRIBUTED="${DISTRIBUTED:-0}"
    -e NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    # goal (sparse) | world_state_tracking_reward (DGPO dense demo tracking) | metaworld_dense
    -e LIBERO_REWARD_MODE="${LIBERO_REWARD_MODE:-world_state_tracking_reward}"
    -v "${LIBERO_USD_DIR}:/data/libero/USD:ro"
    -v "${LIBERO_DEMO_DIR}:/data/libero/demos:ro"
    -v "${LOGS_DIR}:/workspace/hebero/logs"
    -v hebero-kit-cache:/root/.cache
    -v hebero-ov-data:/root/.local/share/ov
)
if [ -n "${LIBERO_COMPAT_MAX_DEMOS_PER_TASK:-}" ]; then
    DOCKER_ARGS+=(-e LIBERO_COMPAT_MAX_DEMOS_PER_TASK="${LIBERO_COMPAT_MAX_DEMOS_PER_TASK}")
fi
# LIBERO_RANDOM_INIT_STATE=1 -> each object's reset XY is redrawn from the box
# its own demos span (OOD init states). LIBERO_RANDOM_INIT_TIMESTEP=1 -> episodes
# reset to a random timestep along the demo trajectory. Both default off.
for _libero_init_var in LIBERO_RANDOM_INIT_STATE LIBERO_RANDOM_INIT_XY_SCALE LIBERO_RANDOM_INIT_TIMESTEP; do
    if [ -n "${!_libero_init_var:-}" ]; then
        DOCKER_ARGS+=(-e "${_libero_init_var}=${!_libero_init_var}")
    fi
done
# MOUNT_SRC=1: overlay the host working tree over the baked-in copies for fast
# iteration (the image stays stale until ./docker/build.sh is rerun).
if [ "${MOUNT_SRC:-0}" = "1" ]; then
    DOCKER_ARGS+=(
        -v "$(pwd)/source:/workspace/hebero/source"
        -v "$(pwd)/scripts:/workspace/hebero/scripts"
        -v "$(pwd)/config:/workspace/hebero/config"
        # The launchers too: without these the container silently runs the image's
        # baked-in train.sh, so host-side changes to the task id, env-count guard
        # or agent selection are ignored and the run looks like it succeeded on
        # the wrong config.
        -v "$(pwd)/train.sh:/workspace/hebero/train.sh:ro"
        -v "$(pwd)/eval.sh:/workspace/hebero/eval.sh:ro"
        -v "$(pwd)/libero_common.sh:/workspace/hebero/libero_common.sh:ro"
    )
fi

case "${mode}" in
    train)
        exec docker run "${TTY_ARGS[@]}" "${DOCKER_ARGS[@]}" "${IMAGE}" ./train.sh libero "$@"
        ;;
    eval)
        if [ -z "${CKPT:-}" ]; then
            echo "eval requires CKPT=<host path to checkpoint .pt>" >&2
            exit 1
        fi
        if [ ! -f "${CKPT}" ]; then
            echo "checkpoint not found: ${CKPT}" >&2
            exit 1
        fi
        ckpt_dir="$(cd "$(dirname "${CKPT}")" && pwd)"
        ckpt_name="$(basename "${CKPT}")"
        # Mounted read-write: eval_libero.py writes the SR report to
        # <ckpt_dir>/plots/ next to the checkpoint, same as a local run.
        exec docker run "${TTY_ARGS[@]}" "${DOCKER_ARGS[@]}" \
            -v "${ckpt_dir}:/data/checkpoints" \
            -e LIBERO_CKPT="/data/checkpoints/${ckpt_name}" \
            "${IMAGE}" ./eval.sh libero "$@"
        ;;
    shell)
        exec docker run "${TTY_ARGS[@]}" "${DOCKER_ARGS[@]}" "${IMAGE}" bash
        ;;
    *)
        echo "usage: $0 [train|eval|shell] [extra args...]" >&2
        exit 1
        ;;
esac
