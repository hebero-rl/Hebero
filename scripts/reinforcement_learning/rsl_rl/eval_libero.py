# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a Hebero checkpoint with per-task success-rate accounting.

Runs each of the LIBERO harvest tasks
(``env i → task i % n_tasks``) until ``--max_episode_per_task`` episodes
complete, counts the ``libero_success`` termination per task, prints a per-task
SR table (``suite::id`` labels) grouped by suite and writes it next to the
checkpoint.

While it runs, every ``--report_every`` steps (default 60) it prints a table with
**one column per suite** and one row per task id, so each suite reads down as its
own per-task SR list and the suites can be compared across.  ``*`` marks a task
that has reached ``--max_episode_per_task``, and the closing ``all`` row carries
``[tasks at the cap / tasks]`` -- an SR over three finished episodes should not
read like a final number.  Field widths come from the caps, so successive reports
line up.  ``--report_summary_only`` keeps just that ``all`` line.

"""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys

# Eval mode must be set before the env cfg is built.
os.environ.setdefault("LIBERO_EVALUATION", "1")

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

import isaaclab_contrib.tasks  # noqa: F401
from isaaclab_contrib.tasks.manipulation import settings as mt_settings
from isaaclab_contrib.tasks.manipulation.libero.dgpo_layout import harvest_task_to_assignment_key

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip
from _prelaunch import prelaunch_kit_before_cfg  # isort: skip

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate a Hebero checkpoint (per-task SR).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Hebero-All-State-Play-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--max_episode_per_task", type=int, default=50, help="Stop counting a task after this many episodes."
)
parser.add_argument("--max_episodes", type=int, default=None, help="Hard cap on total counted episodes.")
parser.add_argument("--max_steps", type=int, default=500_000, help="Hard cap on env steps (safety).")
parser.add_argument("--random_demos", action="store_true", help="Random demo sampling instead of sequential.")
parser.add_argument(
    "--report_every", type=int, default=60, help="Env steps between live SR reports; 0 disables them."
)
parser.add_argument(
    "--report_summary_only",
    action="store_true",
    help="Collapse each live report to its single 'all' line instead of the full task-id table.",
)
parser.add_argument(
    "--keep_env_culling",
    action="store_true",
    help=(
        "Keep RTX per-env culling even under --viz kit, so the Kit perspective camera shows only"
        " one env. Use when the SR of a camera-equipped run has to be comparable to a --viz none run."
    ),
)
parser.add_argument(
    "--report_path",
    type=str,
    default=None,
    help="Write the final report here instead of <log_dir>/plots/libero_eval_<ckpt><init-state suffix>.txt.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

# clear out sys.argv for Hydra (keep preset tokens like ``presets=physx``)
sys.argv = [sys.argv[0]] + list(remaining_args or [])

installed_version = metadata.version("rsl-rl-lib")


# -- reporting ---------------------------------------------------------------
def init_state_tag() -> tuple[str, list[str]]:
    """Filename suffix + header lines naming the init-state randomisation in effect.

    A randomised-reset run answers a different question than the in-distribution
    one, so it must not silently overwrite the plain report for the same
    checkpoint -- ``libero_eval_model_28200.txt`` and
    ``libero_eval_model_28200_oodxy2x.txt`` sit side by side.

    Returns ``("", [])`` when every knob is at its default, so an ordinary eval
    keeps the original path.
    """
    parts: list[str] = []
    lines: list[str] = []
    if viewer_shows_all_envs():
        parts.append("allenvs")
        lines.append(
            "render: RTX per-env culling off for the viewer -- agentviews include neighbouring"
            " envs, so this SR is not comparable to a --viz none run"
        )
    if mt_settings.truthy("LIBERO_RANDOM_INIT_STATE"):
        scale = mt_settings.number("LIBERO_RANDOM_INIT_XY_SCALE", 1.0)
        parts.append(f"oodxy{scale:g}x")
        lines.append(f"init state: object XY redrawn from the demo range, box scale {scale:g}x")
    noise = mt_settings.number("ROBOT_INIT_NOISE_STD", 0.0)
    if noise > 0.0:
        parts.append(f"jn{noise:g}")
        lines.append(f"init state: robot arm joints perturbed, sigma {noise:g} rad")
    # Mirrors make_dgpo_commands_cfg: LIBERO_EVALUATION force-disables this one, so
    # tagging it under eval would name a randomisation that never happened.
    if mt_settings.truthy("LIBERO_RANDOM_INIT_TIMESTEP") and not mt_settings.truthy("LIBERO_EVALUATION"):
        parts.append("randt")
        lines.append("init state: episodes start at a random demo timestep")
    return ("_" + "_".join(parts) if parts else "", lines)


def viewer_shows_all_envs() -> bool:
    """Should RTX per-env culling be turned off so the Kit viewport shows the whole grid?

    ``IsaacRtxRenderer.prepare_stage`` tags each ``/World/envs/env_i`` subtree with an
    inheriting ``omni:scenePartition`` primvar and each camera under it with the matching
    token, and RTX then shows a camera only the geometry carrying its own token.  That is
    what keeps one env's agentview off the next env's table.  It also applies to the Kit
    perspective camera, which sits under no env and therefore sees no env at all -- the
    viewport looks empty apart from the ground plane, which is the reason a state-based run
    (no ``Camera`` sensor, so ``prepare_stage`` never runs) shows all 40 envs and a vision
    run shows none.

    Only for the interactive viewer, and only when there are cameras to isolate in the
    first place.  ``--viz none`` and the state-based tasks are untouched.
    """
    if args_cli.keep_env_culling or not getattr(args_cli, "enable_cameras", False):
        return False
    return "kit" in (getattr(args_cli, "visualizer", None) or [])


def disable_rtx_env_culling() -> None:
    """No-op ``prepare_stage`` so no env subtree is ever tagged.

    Must run before the scene is built: the tags are authored during the first ``Camera``
    sensor's initialisation, are baked into the render products created right after, and a
    later USD edit does not propagate -- removing the attributes post-hoc changes nothing.

    Without partitioning, policy cameras also see neighbouring environments,
    changing the observations used for evaluation. :func:`init_state_tag` marks
    these reports separately to distinguish them from isolated-camera runs and
    prevent overwriting those results.
    """
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    def _no_partition(self, stage, num_envs):  # noqa: ARG001
        return None

    IsaacRtxRenderer.prepare_stage = _no_partition
    print(
        "[WARN] RTX per-env culling disabled so the Kit perspective camera shows all envs.\n"
        "       Every agentview now also renders its neighbours, so the observations differ\n"
        "       from a --viz none run and this eval's SR is NOT a clean benchmark number.\n"
        "       Pass --keep_env_culling for a comparable run."
    )


def suite_groups(task_names: list[str]) -> list[tuple[str, dict[str, int]]]:
    """``[(suite, {task id: task index}), ...]`` in first-appearance order.

    Task labels are ``suite::id`` (:func:`harvest_task_to_assignment_key`), so the
    suite is the part before the separator and the id the part after.  A label
    without one becomes its own single-task group rather than an error -- this is
    reporting, not validation.
    """
    groups: dict[str, dict[str, int]] = {}
    for i, name in enumerate(task_names):
        suite, _, task_id = name.partition("::")
        groups.setdefault(suite, {})[task_id or suite] = i
    return list(groups.items())


def task_ids(task_names: list[str]) -> list[str]:
    """Every task id across the suites, numerically when they all look numeric."""
    ids = {task_id for _, tasks in suite_groups(task_names) for task_id in tasks}
    return sorted(ids, key=int) if all(i.isdigit() for i in ids) else sorted(ids)


def sr_line(label: str, successes: int, episodes: int, width: int = 24) -> str:
    """One ``label  sr/ep  SR=x.xxx`` row; ``SR=  n/a`` until an episode lands."""
    rate = f"{successes / episodes:.3f}" if episodes > 0 else "  n/a"
    return f"{label:<{width}s} {successes:>4d}/{episodes:<4d} SR={rate}"


def steps_to_success_lines(
    task_names: list[str],
    succ_steps: "torch.Tensor",
    succ_demo_steps: "torch.Tensor",
    succ_ratio_sum: "torch.Tensor",
    succ_ratio_n: "torch.Tensor",
) -> list[str]:
    """Per-task episode length of the successful episodes, against the demos' length.

    ``ratio`` is the mean of the per-episode ``episode / its own demo`` ratios, not
    the ratio of the means: each episode is bound to one specific demo, and averaging
    the pairs keeps that pairing. Below 1.0 the policy reaches the goal in fewer steps
    than the demonstration it was trained from.

    Tasks with no successful episode are listed with ``n/a`` rather than dropped, so
    the block lines up with the SR table above it.
    """
    out = [
        "steps to success vs. the demo's own length (successful episodes only)",
        f"{'task':<24}{'episode':>9}{'demo':>9}{'ratio':>8}{'n':>6}",
    ]
    total_ratio, total_n = 0.0, 0
    for t, name in enumerate(task_names):
        n = int(succ_ratio_n[t])
        if n == 0:
            out.append(f"{name:<24}{'n/a':>9}{'n/a':>9}{'n/a':>8}{0:>6d}")
            continue
        total_ratio += float(succ_ratio_sum[t])
        total_n += n
        out.append(
            f"{name:<24}{float(succ_steps[t]) / n:>9.1f}{float(succ_demo_steps[t]) / n:>9.1f}"
            f"{float(succ_ratio_sum[t]) / n:>8.3f}{n:>6d}"
        )
    if total_n:
        out.append(f"{'ALL':<24}{'':>9}{'':>9}{total_ratio / total_n:>8.3f}{total_n:>6d}")
    return out


def _rate(successes: int, episodes: int) -> str:
    """``0.700``, or ``  n/a`` until an episode lands."""
    return f"{successes / episodes:.3f}" if episodes > 0 else "  n/a"


def _total_cell(successes: int, episodes: int, done: int, n_tasks: int, ep_w: int, task_w: int) -> str:
    """A suite's (or the run's) roll-up: ``SR sr/ep [tasks at the cap]``."""
    return f"{_rate(successes, episodes)} {successes:>{ep_w}d}/{episodes:<{ep_w}d} [{done:>{task_w}d}/{n_tasks}]"


def _task_cell(successes: int, episodes: int, cap: int, ep_w: int) -> str:
    """One task's ``SR sr/ep``, with ``*`` once it has reached the episode cap."""
    mark = "*" if episodes >= cap else " "
    return f"{_rate(successes, episodes)} {successes:>{ep_w}d}/{episodes:<{ep_w}d}{mark}"


def _report_columns(task_names: list[str], cap: int) -> list[dict]:
    """Column specs for the live table: the suites side by side, then the total.

    Field widths come from the caps (a group's episodes top out at
    ``cap * n_tasks``), not from the current counts, so every field stays in the
    same character position as the numbers grow and successive reports stack into
    a table you can read down.
    """
    groups = suite_groups(task_names)
    all_tasks = {name: i for i, name in enumerate(task_names)}
    columns = []
    for header, tasks in groups + [("TOTAL", all_tasks)]:
        n = len(tasks)
        ep_w, task_w = len(str(cap * n)), len(str(n))
        width = max(
            len(header),
            len(_total_cell(cap * n, cap * n, n, n, ep_w, task_w)),
            len(_task_cell(cap, cap, cap, len(str(cap)))),
        )
        columns.append(
            {"header": header, "tasks": tasks, "ep_w": ep_w, "task_w": task_w, "width": width,
             "task_ep_w": len(str(cap))}
        )
    return columns


def progress_table(
    task_names: list[str],
    successes: torch.Tensor,
    episodes: torch.Tensor,
    cap: int,
    steps: int,
    summary_only: bool = False,
) -> str:
    """Live report: one column per suite, one row per task id, then an ``all`` row.

    Each suite reads top to bottom as its own per-task SR list, and the suites sit
    side by side so they can be compared across at a glance.  ``*`` marks a task
    that has reached ``--max_episode_per_task``; the ``all`` row carries
    ``[tasks at the cap / tasks]``, because an SR over three finished episodes
    should not read like a final number.  ``summary_only`` drops the task rows,
    leaving the single ``all`` line.
    """
    columns = _report_columns(task_names, cap)
    ids = task_ids(task_names)
    # floored so the columns hold their positions as the step count gains digits
    label_w = max(6, len(str(steps)), max((len(i) + 1 for i in ids), default=0))

    def row(label: str, cells: list[str]) -> str:
        padded = [f"{c:<{col['width']}s}" for c, col in zip(cells, columns)]
        return ("[eval] " + " | ".join([f"{label:>{label_w}s}"] + padded)).rstrip()

    # the step goes in the header's label column, so the table is one block per
    # report rather than a header plus a stray step line
    lines = [row(str(steps), [col["header"] for col in columns])]
    if not summary_only:
        for task_id in ids:
            lines.append(
                row(
                    f"#{task_id}",
                    [
                        ""
                        if (t := col["tasks"].get(task_id)) is None or col["header"] == "TOTAL"
                        else _task_cell(int(successes[t]), int(episodes[t]), cap, col["task_ep_w"])
                        for col in columns
                    ],
                )
            )
    cells = []
    for col in columns:
        idxs = list(col["tasks"].values())
        cells.append(
            _total_cell(
                int(successes[idxs].sum()),
                int(episodes[idxs].sum()),
                int((episodes[idxs] >= cap).sum()),
                len(idxs),
                col["ep_w"],
                col["task_w"],
            )
        )
    lines.append(row("all", cells))
    return "\n".join(lines)


_EARLY_APP_LAUNCHER = prelaunch_kit_before_cfg(args_cli, sys.argv)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate with per-task success accounting."""
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        elif _EARLY_APP_LAUNCHER is not None and hasattr(_EARLY_APP_LAUNCHER, "device"):
            env_cfg.sim.device = _EARLY_APP_LAUNCHER.device

        # Visit demo init states round-robin unless random sampling was requested.
        libero_cmd = getattr(env_cfg.commands, "libero_demo", None)
        if libero_cmd is not None and not args_cli.random_demos:
            libero_cmd.sample_mode = "sequential"

        # checkpoint resolution
        log_root_path = os.path.abspath(os.path.join(os.getenv("LOGS_DIR", "logs"), "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        # Before gym.make: the partition tags are authored at Camera init and baked into
        # the render products, so this cannot be done once the scene exists.
        if viewer_shows_all_envs():
            disable_rtx_env_culling()

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        loaded_dict = torch.load(resume_path, map_location=agent_cfg.device, weights_only=False)
        if "actor_state_dict" not in loaded_dict:
            raise ValueError(
                f"No 'actor_state_dict' in checkpoint: {resume_path}. Expected an rsl-rl 5 checkpoint."
            )
        # Evaluation only needs model weights.
        load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False}
        # An offline-BC checkpoint trains an actor and nothing else, so it carries no
        # critic and no action-noise parameter.  Evaluation only ever calls the
        # inference policy, so the critic is genuinely unused -- skip it rather than
        # failing the strict load.  The missing actor keys are filled from the live
        # model, but only ones outside the weight/normalizer namespaces: a checkpoint
        # missing an ``mlp.*`` or ``obs_normalizer.*`` tensor is broken, and must still
        # fail loudly instead of being evaluated with that tensor left at init.
        if not loaded_dict.get("critic_state_dict"):
            load_cfg["critic"] = False
            print("[INFO]: Actor-only checkpoint (no critic) — evaluating the inference policy.")
        actor_sd = loaded_dict.get("actor_state_dict")
        if actor_sd is not None:
            live_sd = runner.alg.actor.state_dict()
            absent = [k for k in live_sd if k not in actor_sd]
            weights = [k for k in absent if k.startswith(("mlp.", "obs_normalizer."))]
            if absent and not weights:
                for key in absent:
                    actor_sd[key] = live_sd[key].clone()
                print(f"[INFO]: Not in the checkpoint, kept at init: {absent}")
        runner.alg.load(loaded_dict, load_cfg, strict=True)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # per-task bookkeeping (env i -> task i % n_tasks), suite::id labels
        bindings = env.unwrapped.cfg.dgpo_task_bindings
        n_tasks = len(bindings)
        task_names = [harvest_task_to_assignment_key(b.name, b.suite) for b in bindings]
        num_envs = env.unwrapped.num_envs
        env_task = torch.arange(num_envs, device=env.unwrapped.device) % n_tasks
        episodes = torch.zeros(n_tasks, dtype=torch.long)
        successes = torch.zeros(n_tasks, dtype=torch.long)
        cap = int(args_cli.max_episode_per_task)

        term_manager = env.unwrapped.termination_manager
        has_success_term = "libero_success" in term_manager.active_terms

        # Steps-to-success measured against the length of the demo the env was bound
        # to, so the question "did online PPO find a shorter route than the
        # demonstration it was trained from" has an answer per task: ratio < 1 is
        # faster than the demo. Only successful episodes count -- a failure runs to the
        # time limit, so its length measures the cap and not the behaviour.
        try:
            demo_cmd = env.unwrapped.command_manager.get_term("libero_demo")
        except KeyError:
            demo_cmd = None
        traj_lengths = getattr(demo_cmd, "_traj_lengths", None) if demo_cmd is not None else None
        ep_steps = torch.zeros(num_envs, dtype=torch.long, device=env.unwrapped.device)
        succ_steps = torch.zeros(n_tasks, dtype=torch.float64)
        succ_demo_steps = torch.zeros(n_tasks, dtype=torch.float64)
        succ_ratio_sum = torch.zeros(n_tasks, dtype=torch.float64)
        succ_ratio_n = torch.zeros(n_tasks, dtype=torch.long)

        # Re-seed through ``ManagerBasedEnv.seed`` (torch / numpy / random / warp /
        # replicator).  The env already seeded itself from ``env_cfg.seed`` at
        # construction, but building the rsl-rl runner afterwards draws from the torch
        # RNG for the network init, so the state at rollout start otherwise depends on
        # the checkpoint's architecture.  After this the rollout's RNG state is a pure
        # function of ``--seed``.
        env.unwrapped.seed(agent_cfg.seed)
        print(f"[INFO]: RNGs re-seeded before rollout: {agent_cfg.seed}")

        obs = env.get_observations()
        steps = 0
        try:
            while True:
                with torch.inference_mode():
                    # Read the binding before stepping: a reset inside step() rebinds
                    # the env to its next demo, so afterwards this is the length of the
                    # episode that is about to start, not the one that just ended.
                    demo_len = (
                        traj_lengths[demo_cmd._active_traj].to(torch.float64)
                        if traj_lengths is not None
                        else None
                    )
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    policy.reset(dones)
                steps += 1
                ep_steps += 1
                done_ids = dones.nonzero(as_tuple=True)[0]
                if done_ids.numel() > 0:
                    success_buf = (
                        term_manager.get_term("libero_success")
                        if has_success_term
                        else torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
                    )
                    for e in done_ids.tolist():
                        t = int(env_task[e].item())
                        if episodes[t] >= cap:
                            continue
                        episodes[t] += 1
                        if bool(success_buf[e]):
                            successes[t] += 1
                            length = float(ep_steps[e])
                            succ_steps[t] += length
                            reference = float(demo_len[e]) if demo_len is not None else 0.0
                            if reference > 0.0:
                                succ_demo_steps[t] += reference
                                succ_ratio_sum[t] += length / reference
                                succ_ratio_n[t] += 1
                    # Every env that just reset starts its next episode at zero, cap or no cap.
                    ep_steps[done_ids] = 0
                total = int(episodes.sum().item())
                if bool((episodes >= cap).all()):
                    break
                if args_cli.max_episodes is not None and total >= int(args_cli.max_episodes):
                    break
                if steps >= int(args_cli.max_steps):
                    print(f"[WARN] step cap {args_cli.max_steps} reached before all tasks finished.")
                    break
                if args_cli.report_every > 0 and steps % args_cli.report_every == 0:
                    print(
                        progress_table(
                            task_names, successes, episodes, cap, steps,
                            summary_only=args_cli.report_summary_only,
                        ),
                        flush=True,
                    )
        except KeyboardInterrupt:
            pass

        # ---- report ----
        tag, tag_lines = init_state_tag()
        lines = ["Hebero evaluation", f"checkpoint: {resume_path}"] + tag_lines + [""]
        for suite, tasks in suite_groups(task_names):
            idxs = list(tasks.values())
            for t in idxs:
                lines.append(sr_line(task_names[t], int(successes[t]), int(episodes[t])))
            lines.append(sr_line(f"  {suite} (all)", int(successes[idxs].sum()), int(episodes[idxs].sum())))
            lines.append("")
        lines.append(sr_line("OVERALL", int(successes.sum()), int(episodes.sum())))
        lines += ["", *steps_to_success_lines(task_names, succ_steps, succ_demo_steps,
                                              succ_ratio_sum, succ_ratio_n)]
        # same suites-side-by-side table the run printed as it went, so the file
        # can be diffed against a live tail or against another checkpoint's
        lines += ["", progress_table(task_names, successes, episodes, cap, steps)]
        report = "\n".join(lines)
        print("\n" + report)

        if args_cli.report_path:
            out_path = os.path.abspath(args_cli.report_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        else:
            out_dir = os.path.join(log_dir, "plots")
            os.makedirs(out_dir, exist_ok=True)
            ckpt_tag = os.path.splitext(os.path.basename(resume_path))[0]
            out_path = os.path.join(out_dir, f"libero_eval_{ckpt_tag}{tag}.txt")
        with open(out_path, "w") as f:
            f.write(report + "\n")
        print(f"[INFO] wrote {out_path}")

        env.close()


if __name__ == "__main__":
    main()
    # launch_simulation does not own the early Kit app; close it before Python's atexit teardown.
    if _EARLY_APP_LAUNCHER is not None:
        _EARLY_APP_LAUNCHER.app.close()
