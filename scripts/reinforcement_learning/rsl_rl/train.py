# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

import argparse
import contextlib
import importlib.metadata as metadata
import logging
import os
import platform
import sys
import time
from datetime import datetime

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

import isaaclab_contrib.tasks  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip
from _prelaunch import prelaunch_kit_before_cfg  # isort: skip

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

RSL_RL_VERSION = "5.0.1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function. Both sides of this
# intersection share the same token vocabulary (the callback reads the user's
# original sys.argv), so preset tokens like ``physics=NAME`` compare correctly.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# -- check RSL-RL version ----------------------------------------------------
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

# Establish the Kit linker namespace before the hydra decorator builds the
# (potentially heavy) LIBERO DGPO config, so it cannot crash carb's dlmopen;
# launch_simulation() below relaunches as a no-op. main() mirrors the device.
_EARLY_APP_LAUNCHER = prelaunch_kit_before_cfg(args_cli, sys.argv)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        if args_cli.init_from_checkpoint and agent_cfg.resume:
            raise ValueError(
                "--init_from_checkpoint (actor-only warm-start) cannot be combined with --resume /"
                " agent.resume=true (full training-state resume). Pick one."
            )
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        # For distributed training, launch_simulation() already resolved the
        # correct per-rank device; only apply a CLI --device override for
        # non-distributed runs (the default "cuda:0" would clobber the
        # per-rank device otherwise).
        if not args_cli.distributed:
            if args_cli.device is not None:
                env_cfg.sim.device = args_cli.device
            elif _EARLY_APP_LAUNCHER is not None and hasattr(_EARLY_APP_LAUNCHER, "device"):
                # launch_simulation() skips device propagation when Kit is already running.
                env_cfg.sim.device = _EARLY_APP_LAUNCHER.device
        # check for invalid combination of CPU device with distributed training
        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError(
                "Distributed training is not supported when using CPU device. "
                "Please use GPU device (e.g., --device cuda) for distributed training."
            )

        # multi-gpu training configuration
        if args_cli.distributed:
            global_rank = int(os.getenv("RANK", "0"))
            # env_cfg.sim.device is resolved by launch_simulation() which
            # accounts for CUDA_VISIBLE_DEVICES restrictions.
            agent_cfg.device = env_cfg.sim.device

            # use global rank for seed diversity across all nodes
            seed = agent_cfg.seed + global_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        # specify directory for logging experiments.  LOGS_DIR (same variable the
        # docker launcher uses for the host mount) redirects the whole tree, so a
        # read-only or shared checkout can still write checkpoints somewhere else.
        log_root_path = os.path.join(os.getenv("LOGS_DIR", "logs"), "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        # specify directory for logging runs: {time-stamp}_{run_name}
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
        # change it (see PR #2346, comment-2819298849)
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        # set the IO descriptors export flag if requested
        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        else:
            logger.warning(
                "IO descriptors are only supported for manager based RL environments."
                " No IO descriptors will be exported."
            )

        # set the log directory for the environment (works for all environment types)
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # save resume path before creating a new log_dir
        if agent_cfg.resume or getattr(agent_cfg.algorithm, "class_name", "") == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()

        # RFCL: inject the curriculum wrapper BEFORE the RSL-RL vec-env wrapper
        # (activated when the runner cfg carries a populated rfcl_cfg field).
        # Retain the wrapper so the PPO curriculum logger can reach its scheduler.
        _rfcl_cfg = getattr(agent_cfg, "rfcl_cfg", None)
        _rfcl_env_wrapper = None
        if _rfcl_cfg is not None and (_rfcl_cfg.demo_hdf5_paths or _rfcl_cfg.demo_hdf5_dir):
            from isaaclab_rl.rsl_rl.rfcl_curriculum import RFCLEnvWrapper

            print("[INFO] RFCL curriculum wrapper enabled.")
            env = RFCLEnvWrapper(env, _rfcl_cfg, device=agent_cfg.device)
            _rfcl_env_wrapper = env

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # create runner from rsl-rl
        if agent_cfg.class_name == "OnPolicyRunner":
            # Same runner, plus checkpoint pruning: upstream saves every
            # save_interval and never deletes, which at 30k iterations is ~300
            # files nothing will read.  See isaaclab_rl.rsl_rl.checkpoint_pruning.
            from isaaclab_rl.rsl_rl.checkpoint_pruning import PrunedOnPolicyRunner

            runner = PrunedOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # per-task SR EMAs go to TensorBoard only; the console keeps the mean + per-suite lines
        from isaaclab_rl.rsl_rl.utils import install_rfcl_curriculum_logger, install_tb_only_console_filter

        # Add RFCL curriculum progress to the PPO runner's training metrics.
        if agent_cfg.class_name == "OnPolicyRunner" and _rfcl_env_wrapper is not None:
            install_rfcl_curriculum_logger(runner, _rfcl_env_wrapper.scheduler)

        install_tb_only_console_filter(runner)
        # configure_seed must be called after runner construction so that PyTorch deterministic settings
        # do not interfere with the runner's internal initialization.
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)
        # write git state to logs
        runner.add_git_repo_to_log(__file__)
        # load the checkpoint
        if args_cli.init_from_checkpoint:
            # Actor-only warm-start: critic, optimizer, and iteration count all stay at
            # their fresh init (unlike --resume, this is not the same training run
            # continuing -- it is a new PPO run seeded with a pretrained actor, most
            # commonly an offline-BC checkpoint that never had a critic to begin with).
            ckpt_path = retrieve_file_path(args_cli.init_from_checkpoint)
            print(f"[INFO]: Warm-starting actor from checkpoint: {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=agent_cfg.device, weights_only=False)
            actor_sd = loaded_dict.get("actor_state_dict")
            if actor_sd is None:
                raise ValueError(
                    f"No 'actor_state_dict' in checkpoint: {ckpt_path}. Expected an rsl-rl 5 checkpoint."
                )
            # Fill anything the checkpoint doesn't carry (e.g. an offline-BC actor has no
            # action-noise distribution parameter) from the freshly initialized model --
            # but only outside the weight/normalizer namespaces. A checkpoint missing an
            # ``mlp.*`` or ``obs_normalizer.*`` tensor is broken and must fail loudly
            # instead of silently warm-starting from an untrained tensor.
            live_sd = runner.alg.actor.state_dict()
            absent = [k for k in live_sd if k not in actor_sd]
            weights = [k for k in absent if k.startswith(("mlp.", "obs_normalizer."))]
            if weights:
                raise ValueError(f"Checkpoint {ckpt_path} is missing actor weight/normalizer tensors: {weights}")
            for key in absent:
                actor_sd[key] = live_sd[key].clone()
            if absent:
                print(f"[INFO]: Not in the checkpoint, kept at init: {absent}")
            load_cfg = {"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False}
            runner.alg.load(loaded_dict, load_cfg, strict=True)
            print("[INFO]: Warm-start complete — critic, optimizer, and iteration count are fresh.")
        elif agent_cfg.resume or getattr(agent_cfg.algorithm, "class_name", "") == "Distillation":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            # load previously trained model
            runner.load(resume_path)
            # `env.task_sr_tracker` is rebuilt from scratch by every gym.make() above
            # and isn't part of the checkpoint (only the algorithm's mirror of it is,
            # see isaaclab_rl.rsl_rl.dgpo_ppo's module docstring). Without this, a
            # resumed run's gripper curriculum and Metrics/task_sr/* logging cold-start
            # from zero for ~20 episodes/task even though the checkpointed policy and
            # its SR estimate are already warm.
            _task_sr_ema = getattr(runner.alg, "_task_sr_ema", None)
            _task_sr_initialized = getattr(runner.alg, "_task_sr_initialized", None)
            _tracker = getattr(env.unwrapped, "task_sr_tracker", None)
            if _task_sr_ema is not None and _task_sr_initialized is not None and _tracker is not None:
                _tracker.restore(_task_sr_ema, _task_sr_initialized)

        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        # run training
        try:
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
            # close the simulator
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
    # launch_simulation does not own the early Kit app; close it before Python's atexit teardown.
    if _EARLY_APP_LAUNCHER is not None:
        _EARLY_APP_LAUNCHER.app.close()
