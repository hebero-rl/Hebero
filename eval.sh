#!/usr/bin/env bash
# Per-task SR evaluation entry points for Hebero.
#
# Usage:
#   LIBERO_CKPT=/path/to/model.pt ./eval.sh          # all 40 LIBERO tasks
#   ./eval.sh libero --checkpoint /path/to/model.pt # equivalent CLI override
#
# LIBERO task subset (mirrors train.sh -- LIBERO_TASK and LIBERO_NUM_TASKS
# must track each other, same as there): to evaluate a libero_10 long-horizon
# checkpoint against its matching task bindings instead of the default
# 40-task suite:
#   LIBERO_TASK=Isaac-Hebero-Long-State-Play-v0 LIBERO_NUM_TASKS=10 \
#   LIBERO_CKPT=logs/rsl_rl/libero_10_dgpo/<run>/model_XXXX.pt \
#     ./eval.sh libero
#
# Loads rsl-rl 5 checkpoints produced by this source package.
# Requires the IsaacLab python env, e.g.:
#   source .venv/bin/activate
#
# Default: --viz none, preserving per-env camera isolation for evaluation.
# Optional VIZ_ARGS="--viz kit" disables that isolation for visual tasks, letting
# policy cameras see neighbouring envs; these altered-observation runs are tagged
# `_allenvs`. Add --keep_env_culling to preserve comparable visual observations.
#
# The libero run prints live SR as one line per report, the four suites side by
# side (~137 columns). Extra args pass straight through, so:
#   VIZ_ARGS="--viz none" ./eval.sh libero --report_tasks          # add a per-task block per report
#   VIZ_ARGS="--viz none" ./eval.sh libero --report_every 200      # less often
#   VIZ_ARGS="--viz none" ./eval.sh libero --report_every 0        # silence it
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Checkpoints under test (override via env, e.g. `LIBERO_CKPT=... ./eval.sh libero`)
# ---------------------------------------------------------------------------
LIBERO_CKPT="${LIBERO_CKPT:-}"

# Task + env count. LIBERO_NUM_TASKS must track LIBERO_TASK -- it turns
# num_envs into the envs-per-task group size (see require_env_multiple in
# libero_common.sh) and is also how the checkpoint's task bindings are sized;
# a 10-task checkpoint loaded against the 40-task suite will not line up.
LIBERO_TASK="${LIBERO_TASK:-Isaac-Hebero-All-State-Play-v0}"
LIBERO_NUM_ENVS="${LIBERO_NUM_ENVS:-160}"
LIBERO_NUM_TASKS="${LIBERO_NUM_TASKS:-40}"

# Episodes to roll out per task before a task's SR is finalized.
MAX_EPISODES_PER_TASK_LIBERO="${MAX_EPISODES_PER_TASK_LIBERO:-50}"

# Viewer-less evaluation by default; optional Kit viewer caveat is documented above.
VIZ_ARGS="${VIZ_ARGS:---viz none}"

# Data locations + benchmark-level config shared with training (reward mode,
# privileged critic, seed, ...) -- see libero_common.sh. Keeps an eval run
# honest about the benchmark config its checkpoint was actually trained under.
# shellcheck source=libero_common.sh
source ./libero_common.sh
# Isaac cloud-asset download cache; see the TMPDIR note in libero_common.sh.
# IsaacLab reuses an existing mirror without checking it is readable, so a
# root-owned /tmp/Assets tree turns into "No collision prim found at path:
# '/World/GroundPlane'". Keep the mirror under $HOME. The directory must exist:
# `tempfile` silently falls back to /tmp when TMPDIR points at a missing path.
export TMPDIR="${TMPDIR:-${HOME}/.cache/isaac_assets}"
mkdir -p "${TMPDIR}"

eval_libero() {
    # A `--num_envs N` in the caller's extra args overrides the one built below --
    # argparse takes the last occurrence -- so resolve it before the guard runs.
    # Otherwise the guard validates LIBERO_NUM_ENVS while the run uses the override,
    # and a non-multiple slips through.
    local _prev="" _arg
    for _arg in "$@"; do
        [ "${_prev}" = "--num_envs" ] && LIBERO_NUM_ENVS="${_arg}"
        [ "${_prev}" = "--checkpoint" ] && LIBERO_CKPT="${_arg}"
        case "${_arg}" in
            --num_envs=*) LIBERO_NUM_ENVS="${_arg#*=}" ;;
            --checkpoint=*) LIBERO_CKPT="${_arg#*=}" ;;
        esac
        _prev="${_arg}"
    done
    if [ -z "${LIBERO_CKPT}" ]; then
        echo "Set LIBERO_CKPT or pass --checkpoint /path/to/model.pt." >&2
        return 1
    fi
    require_env_multiple "${LIBERO_NUM_ENVS}" "${LIBERO_NUM_TASKS}"
    # Vision checkpoints need the cameras their actor was trained on; see train.sh.
    CAMERA_ARGS=()
    case "${LIBERO_TASK}" in *-Vision-*) CAMERA_ARGS=(--enable_cameras) ;; esac
    ./run.sh scripts/reinforcement_learning/rsl_rl/eval_libero.py \
        --task "${LIBERO_TASK}" \
        --num_envs "${LIBERO_NUM_ENVS}" \
        --checkpoint "${LIBERO_CKPT}" \
        --max_episode_per_task "${MAX_EPISODES_PER_TASK_LIBERO}" \
        --seed "${SEED}" \
        "${CAMERA_ARGS[@]}" \
        presets=physx \
        ${VIZ_ARGS} \
        "$@"
}

target="${1:-libero}"
shift || true
case "${target}" in
    libero) eval_libero "$@" ;;
    *) echo "usage: $0 [libero] [extra eval args...]" >&2; exit 1 ;;
esac
