#!/usr/bin/env bash
# Training entry points for Hebero (rsl-rl).
#
# Usage:
#   ./train.sh            # LIBERO 40-task suite
#   ./train.sh libero     # explicit suite selection
#
# Algorithm selection (AGENT env var → train.py --agent):
#   rsl_rl_cfg_entry_point        IW-ABC — adaptive BC + SR importance weights (default)
#   rsl_rl_iw_abc_cfg_entry_point IW-ABC / Vis IW-ABC, selected by the task
#   rsl_rl_mtppo_cfg_entry_point  PPO base — IW enabled; see README for plain PPO
#   rsl_rl_dapg_cfg_entry_point   MT-DAPG — demo log-likelihood loss
#   rsl_rl_ppo_rfcl_cfg_entry_point PPO + RFCL reset-state curriculum (IW enabled)
#
# Distributed (single node, e.g. 8 GPUs):
#   Defaults to 8 GPUs x 2000 envs/GPU (50 envs/task/GPU, 400 envs/task total).
#   Single GPU: DISTRIBUTED=0 LIBERO_NUM_ENVS=200 ./train.sh libero
#
# Examples:
#   AGENT=rsl_rl_mtppo_cfg_entry_point MAX_ITERATIONS=30000 ./train.sh libero
#   DISTRIBUTED=0 LIBERO_NUM_ENVS=200 ./train.sh libero   # single-GPU debug
#
# Trains from scratch and writes checkpoints under logs/rsl_rl/<experiment_name>/.
# To resume add: --resume --load_run <run_dir> --checkpoint <model_xxx.pt>.
#
# Requires the new-gen IsaacLab python env, e.g.:
#   source .venv/bin/activate
#
# Swap --headless for --viz kit if you want a live viewer (slower).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Training hyper-params (override via env, e.g. `MAX_ITERATIONS=100 ./train.sh libero`)
# ---------------------------------------------------------------------------
# Paper configuration: 2000 envs/GPU x 8 GPUs = 16000 envs total.
# For 40 tasks: 50 envs/task/GPU, 400 envs/task globally.
# The per-GPU count must stay a multiple of the task count.
LIBERO_NUM_ENVS="${LIBERO_NUM_ENVS:-2000}"
# Which LIBERO env to train. Default is all four suites (40 tasks); set to
# Isaac-Hebero-Long-State-v0 for the libero_10 long-horizon suite alone.
# LIBERO_NUM_TASKS must track it -- it is what turns num_envs into the
# envs-per-task group size, and a wrong value silently misreports that.
#
# Vision (frozen Theia + attention pool, actor sees cameras instead of
# ground-truth object poses): Isaac-Hebero-{All,Long}-Vision-v0.
# --enable_cameras is added automatically for those; drop the env count
# accordingly, since two cameras per env are rendered every control step:
#   LIBERO_TASK=Isaac-Hebero-All-Vision-v0 LIBERO_NUM_ENVS=320 ./train.sh libero
LIBERO_TASK="${LIBERO_TASK:-Isaac-Hebero-All-State-v0}"
LIBERO_NUM_TASKS="${LIBERO_NUM_TASKS:-40}"
# Log dir under logs/rsl_rl/. Empty keeps the agent cfg's own experiment_name,
# which is shared across task subsets -- set it for a subset run so its
# checkpoints do not land in the 40-task run's directory.
LIBERO_EXPERIMENT_NAME="${LIBERO_EXPERIMENT_NAME:-}"
# Paper training horizon; override for shorter debug runs.
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
# Algorithm entry point (see the table above).
AGENT="${AGENT:-rsl_rl_cfg_entry_point}"
# Visualization: defaults to the live Kit viewer; set VIZ_ARGS=--headless for
# headless runs (e.g. inside the docker image).
VIZ_ARGS="${VIZ_ARGS:---viz kit}"
# Distributed: DISTRIBUTED=1 launches via torch.distributed.run with
# NPROC_PER_NODE processes (one GPU each; env count is PER PROCESS).
# LIBERO on-policy is an 8-GPU run by default (8 x 2000 envs). Set DISTRIBUTED=0
# for a single-GPU debug run; drop LIBERO_NUM_ENVS with it.
DISTRIBUTED="${DISTRIBUTED:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# Benchmark-level environment shared by training and evaluation: data paths,
# reward mode, privileged critic, gripper curriculum, seed.  Anything that must be
# identical across methods belongs there, not here.
# shellcheck source=libero_common.sh
source ./libero_common.sh

MAX_ITER_ARGS=()
[ -n "${MAX_ITERATIONS}" ] && MAX_ITER_ARGS=(--max_iterations "${MAX_ITERATIONS}")

# torchrun prefix + --distributed flag when DISTRIBUTED=1.
LAUNCH_PREFIX=()
DIST_ARGS=()
if [ "${DISTRIBUTED}" = "1" ]; then
    LAUNCH_PREFIX=(-m torch.distributed.run --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}")
    DIST_ARGS=(--distributed)
    # Distributed runs must be headless.
    VIZ_ARGS="--headless"
fi

train_libero() {
    require_env_multiple "${LIBERO_NUM_ENVS}" "${LIBERO_NUM_TASKS}"
    EXPERIMENT_ARGS=()
    [ -n "${LIBERO_EXPERIMENT_NAME}" ] && EXPERIMENT_ARGS=(--experiment_name "${LIBERO_EXPERIMENT_NAME}")
    # The vision envs mount cameras; without --enable_cameras the app launches
    # with rendering off and the Theia obs term reads an empty sensor buffer.
    CAMERA_ARGS=()
    case "${LIBERO_TASK}" in *-Vision-*) CAMERA_ARGS=(--enable_cameras) ;; esac
    ./run.sh "${LAUNCH_PREFIX[@]}" scripts/reinforcement_learning/rsl_rl/train.py \
        --task "${LIBERO_TASK}" \
        --agent "${AGENT}" \
        --num_envs "${LIBERO_NUM_ENVS}" \
        "${MAX_ITER_ARGS[@]}" \
        "${EXPERIMENT_ARGS[@]}" \
        --seed "${SEED}" \
        "${CAMERA_ARGS[@]}" \
        presets=physx \
        ${VIZ_ARGS} \
        "${DIST_ARGS[@]}" \
        "$@"
}

target="${1:-libero}"
shift || true
case "${target}" in
    libero) train_libero "$@" ;;
    *) echo "usage: $0 [libero] [extra train.py args...]" >&2; exit 1 ;;
esac
