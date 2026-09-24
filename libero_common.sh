#!/usr/bin/env bash
# Shared launch environment for every LIBERO method.
#
# Sourced by train.sh and eval.sh so data paths, rewards, observation groups,
# and curricula stay consistent across training and evaluation.
# Per-method settings (algorithm, num_envs, task subset) stay in the launchers.

# Data locations.  Default to the Hugging Face pull at <repo>/datasets (README §2.3);
# override either variable to point at a shared copy.
_LIBERO_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_ASSETS_DATA_DIR="${LIBERO_ASSETS_DATA_DIR:-${_LIBERO_REPO_ROOT}/datasets/USD}"
export LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-${_LIBERO_REPO_ROOT}/config}"
export LIBERO_ASSEMBLED_DATASET_DIR="${LIBERO_ASSEMBLED_DATASET_DIR:-${_LIBERO_REPO_ROOT}/datasets/demos}"

# Isaac cloud-asset download cache.  IsaacLab mirrors remote USDs (ground plane,
# Isaac props/materials) under `tempfile.gettempdir()` and *skips the download
# when the file already exists* -- it never checks that the file is readable.  A
# root-owned /tmp/Assets tree, left behind by a docker or sudo run on the same
# box, is therefore reused, the USD reference silently composes to nothing, and
# the failure surfaces far from its cause (e.g. "No collision prim found at
# path: '/World/GroundPlane'").  Keeping the mirror under $HOME makes it belong
# to whoever runs the job.  The directory must exist: `tempfile` silently falls
# back to /tmp when TMPDIR points at a missing path.
export TMPDIR="${TMPDIR:-${HOME}/.cache/isaac_assets}"
mkdir -p "${TMPDIR}"

# The demonstration-guided reward every method runs on.  Also the env-config
# default now, so this line is belt-and-braces; `goal` (sparse) and
# `metaworld_dense` (demo-free) are different benchmarks, not different algorithms.
export LIBERO_REWARD_MODE="${LIBERO_REWARD_MODE:-world_state_tracking_reward}"

# Asymmetric critic (privileged_proprio) for every method.
export LIBERO_PRIVILEGED_CRITIC="${LIBERO_PRIVILEGED_CRITIC:-1}"

# SR-gated demo replay of the gripper bit, for every method.
export GRIPPER_CURRICULUM="${GRIPPER_CURRICULUM:-1}"

# Seed.  Fixed rather than left to the runner default so a method's result is not
# a draw from an unrecorded distribution; sweep it with SEED=0/1/2 for error bars.
SEED="${SEED:-0}"

# env i -> task i % n_tasks, so anything that is not a multiple of n_tasks gives
# the first NUM_ENVS % n_tasks tasks one extra env each.
#
# n_tasks defaults to 40 (all four suites). A single-suite run is 10, so the
# count is a parameter rather than a literal -- with 40 hard-coded, a 10-task
# run either fails the check or silently reports the wrong group size, and group
# size is the quantity that has to match across methods.
require_env_multiple() {
  local n="$1"
  local n_tasks="${2:-40}"
  if [ "$((n % n_tasks))" -ne 0 ] || [ "${n}" -lt "${n_tasks}" ]; then
    echo "num_envs must be a positive multiple of ${n_tasks} (one env per LIBERO task); got ${n}" >&2
    exit 1
  fi
  # Print it: envs-per-task is the quantity that has to match across methods, and
  # it is not visible anywhere in the launch command.
  echo "[libero] ${n} envs / ${n_tasks} tasks = $((n / n_tasks)) envs per task (group size)"
}
