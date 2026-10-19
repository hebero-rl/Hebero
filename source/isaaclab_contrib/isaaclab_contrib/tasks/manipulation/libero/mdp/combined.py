# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env task-gather MDP for the harvest (object-level prototype-sharing) LIBERO env.

The combined ``Isaac-Hebero-*-State-*`` scene shares identical object models as a
single :class:`~isaaclab.assets.AssetView` across every task that uses them (see
:mod:`..tasks.harvest.prototypes`), so a shared view spans envs of *different*
tasks and the usual per-clone-group ``SceneEntityCfg(selector=group)`` scoping
does not work.

Instead every task-dependent term gathers **per env by task id**, using the
deterministic clone map ``env i -> task i % n_tasks`` (:func:`~isaaclab.cloner.sequential`)
to scatter each env's own goal-object pose into a full ``(num_envs, ...)`` buffer.
Assets are located with ``selector.filter_reset_ids(proto_name, env_ids)``.
Quaternions are ``(x, y, z, w)``.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from ...utils.task_sr import stash_task_success
from ...utils.commands import resolve_command_term
from .observations import _to_torch
from ... import settings
from ..tasks.common.suite_loader import articulation_joint_spec, orientation_offset

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from ..tasks.harvest.prototypes import TaskBinding


# ---------------------------------------------------------------------------
# Per-env runtime (built lazily, cached on the env instance)
# ---------------------------------------------------------------------------


#: BDDL spatial relationships the success proxy knows how to evaluate.
RELATIONAL_GOAL_TYPES = ("on", "in")

#: Fallbacks for goals that omit a tolerance. Every LIBERO goal in ``config/``
#: carries both, so these only matter for hand-written or future configs.
DEFAULT_GOAL_XY_THRESHOLD = 0.10
DEFAULT_GOAL_HEIGHT_THRESHOLD = 0.10


def _is_relational(goal: dict) -> bool:
    """True for a goal the relational success predicate can evaluate."""
    return goal.get("relationship") in RELATIONAL_GOAL_TYPES and bool(goal.get("ref_obj")) and bool(goal.get("target"))


class _LiberoRuntime:
    """Cached per-env task assignment and prototype gather indices.

    Built once from the deterministic ``env i -> task i % n_tasks`` clone map and
    the harvested task bindings; reused by every task-dependent MDP term.
    """

    def __init__(self, env: ManagerBasedEnv, tasks: list[TaskBinding]) -> None:
        """Build the task map plus per-prototype primary/target env-id gather tensors."""
        device = env.device
        self.device = device
        self.n_tasks = len(tasks)
        n = env.num_envs
        self.env_task = torch.arange(n, device=device, dtype=torch.long) % self.n_tasks

        # Success tolerances are NOT cached here: they are per *goal*, not per
        # task, and are read straight off each goal dict in
        # `relational_goals_satisfied`. A task-level tolerance cannot express
        # "put both mugs on their own plates" with different tolerances per mug.

        # Backing stores for the run-constant lookups below.
        self._env_ids_by_task: dict[int, torch.Tensor] = {}
        self._view_ids: dict[tuple[str, int], torch.Tensor | None] = {}
        self._proto_names: dict[int, dict[str, str]] = {}
        self._joint_index: dict[tuple[str, str], int | None] = {}
        self._body_index: dict[tuple[str, str], int | None] = {}
        self._goal_consts: dict[int, dict[str, torch.Tensor | None]] = {}

        # prototype name -> global env ids for which it is the primary / target object
        self.primary_envs = self._group_by_proto(tasks, "primary_proto", device)
        self.target_envs = self._group_by_proto(tasks, "target_proto", device)

        # per-env masks: does the env's task define a primary rigid object / any
        # articulation operation goal? Pure-articulation tasks (open drawer,
        # turn on stove) have primary_proto=None and are judged on joints only.
        self.has_primary = torch.zeros(n, dtype=torch.bool, device=device)
        for envs in self.primary_envs.values():
            self.has_primary[envs] = True
        has_op = torch.tensor(
            [any("operation" in g for g in getattr(t, "goals", ())) for t in tasks], device=device
        )
        self.has_op_goal = has_op[self.env_task]
        has_rel = torch.tensor(
            [any(_is_relational(g) for g in getattr(t, "goals", ())) for t in tasks], device=device
        )
        self.has_rel_goal = has_rel[self.env_task]

    # -- Run-constant lookups -------------------------------------------------
    #
    # Everything below answers a question whose inputs are fixed once the scene
    # exists: which envs run a task, where a prototype's rows live in its asset
    # view, what a task calls its objects, which joint a goal names. The
    # goal-evaluation loops ask these once per goal operand per task per *step*,
    # so each is resolved on first ask and kept. They live together here rather
    # than as separate caches next to their callers, so there is one place to
    # look for "what does this run treat as constant".

    def env_ids_for(self, task_idx: int) -> torch.Tensor:
        """Global env ids assigned to ``task_idx``. Cached; callers must treat as read-only."""
        cached = self._env_ids_by_task.get(task_idx)
        if cached is None:
            cached = (self.env_task == task_idx).nonzero(as_tuple=True)[0]
            self._env_ids_by_task[task_idx] = cached
        return cached

    def view_ids_for(self, env, proto: str, task_idx: int) -> torch.Tensor | None:
        """Rows of ``proto``'s asset view for ``task_idx``'s envs, aligned with :meth:`env_ids_for`.

        ``None`` ("cannot resolve") if the sizes don't match 1:1, which would
        otherwise silently misalign per-env rewards. Cached either way.
        """
        key = (proto, task_idx)
        if key not in self._view_ids:
            env_ids = self.env_ids_for(task_idx)
            global_ids, view_ids = env.scene.selector.filter_reset_ids(proto, env_ids)
            self._view_ids[key] = None if global_ids.numel() != env_ids.numel() else view_ids
        return self._view_ids[key]

    def proto_names_for(self, binding: TaskBinding) -> dict[str, str]:
        """A task's canonical goal-object names -> prototype scene names."""
        key = id(binding)
        cached = self._proto_names.get(key)
        if cached is None:
            cached = {obj.canonical_name: obj.proto_name for obj in binding.object_bindings}
            self._proto_names[key] = cached
        return cached

    def joint_index_for(self, env, proto: str, joint_pattern: str) -> int | None:
        """Index of the single joint of ``proto`` matching ``joint_pattern``, else ``None``. Cached."""
        key = (proto, joint_pattern)
        if key not in self._joint_index:
            found, _ = env.scene[proto].find_joints(joint_pattern)
            self._joint_index[key] = int(found[0]) if len(found) == 1 else None
        return self._joint_index[key]

    def body_index_for(self, asset, proto: str, segment: str) -> int | None:
        """Row of ``proto``'s body buffer whose name contains ``segment`` (case-insensitive), else ``None``. Cached."""
        key = (proto, segment)
        if key not in self._body_index:
            found = None
            for i, body_name in enumerate(getattr(asset, "body_names", []) or []):
                if re.match(rf"(?i).*{re.escape(segment)}.*", body_name):
                    found = i
                    break
            self._body_index[key] = found
        return self._body_index[key]

    def goal_consts(self, goal: dict) -> dict[str, torch.Tensor | None]:
        """:func:`build_goal_consts` for one goal, built once and kept (keyed by ``id(goal)``;
        safe since the goal dicts hang off ``cfg.dgpo_task_bindings`` and outlive this runtime)."""
        cached = self._goal_consts.get(id(goal))
        if cached is None:
            cached = build_goal_consts(goal, self.device)
            self._goal_consts[id(goal)] = cached
        return cached

    def _group_by_proto(self, tasks: list[TaskBinding], attr: str, device: str) -> dict[str, torch.Tensor]:
        """Map each prototype name to the sorted global env ids whose task uses it in ``attr``'s role."""
        out: dict[str, list[torch.Tensor]] = {}
        for task_idx, task in enumerate(tasks):
            name = getattr(task, attr)
            if name is None:
                continue
            envs = (self.env_task == task_idx).nonzero(as_tuple=True)[0]
            out.setdefault(name, []).append(envs)
        return {name: torch.cat(envs).sort().values for name, envs in out.items()}


def _const_vec(values: Sequence[float] | None, device) -> torch.Tensor | None:
    """A float32 device tensor for an authored vector, or ``None`` when the goal omits it."""
    if values is None:
        return None
    return torch.tensor([float(v) for v in values], dtype=torch.float32, device=device)


def build_goal_consts(goal: dict, device) -> dict[str, torch.Tensor | None]:
    """Build the constant device vectors (``target_offset``, ``xy_half_extents``,
    ``upright_axis``, world-frame ``orientation`` offset) one relational goal needs every step.

    ``None`` means the goal does not carry that field (e.g. missing ``target_offset``
    means "use the target's own origin"), never "not cached" -- ``upright_axis`` always
    comes back set since it has a default. Cached per run via :meth:`_LiberoRuntime.goal_consts`;
    exposed as a free function for callers without a runtime.
    """
    return {
        "target_offset": _const_vec(goal.get("target_offset"), device),
        "world_offset": _const_vec(orientation_offset(str(goal.get("orientation", "")).lower()), device),
        "xy_half_extents": _const_vec(goal.get("xy_half_extents"), device),
        "upright_axis": _const_vec(goal.get("upright_axis", (0.0, 0.0, 1.0)), device),
    }


def _resolve_tasks(env: ManagerBasedEnv, tasks: list[TaskBinding] | None) -> list[TaskBinding]:
    """Resolve the task bindings: explicit param > ``env.cfg.dgpo_task_bindings``.

    Every combined MDP term accepts ``tasks=None`` so manager configs stay declarative.
    """
    if tasks is not None:
        return tasks
    bindings = getattr(env.unwrapped.cfg, "dgpo_task_bindings", None)
    if bindings is None:
        raise RuntimeError("Combined MDP terms require env.cfg.dgpo_task_bindings (set by make_libero_dgpo_env_cfg).")
    return bindings


def _runtime(env: ManagerBasedEnv, tasks: list[TaskBinding] | None) -> _LiberoRuntime:
    """Return the cached :class:`_LiberoRuntime`, building it on first use."""
    rt = getattr(env, "_libero_runtime", None)
    if rt is None:
        rt = _LiberoRuntime(env, _resolve_tasks(env, tasks))
        env._libero_runtime = rt
    return rt


def _gather_object_state(
    env: ManagerBasedEnv, envs_by_proto: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter each env's assigned object world pose + speed into full ``(num_envs, ...)`` buffers.

    ``envs_by_proto`` maps a prototype name to the global env ids using it in this role.
    Returns ``(pos_w [m], speed [m/s])``; rows without an assigned object stay zero.
    """
    selector = env.scene.selector
    pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    speed = torch.zeros(env.num_envs, device=env.device)
    for name, envs in envs_by_proto.items():
        global_ids, view_ids = selector.filter_reset_ids(name, envs)
        if global_ids.numel() == 0:
            continue
        asset = env.scene[name]
        pos_w[global_ids] = wp.to_torch(asset.data.root_pos_w)[view_ids, :3]
        speed[global_ids] = torch.linalg.norm(wp.to_torch(asset.data.root_lin_vel_w)[view_ids, :3], dim=-1)
    return pos_w, speed


# ---------------------------------------------------------------------------
# Goal geometry: resolving a BDDL goal operand to the sim rows it names
#
# Shared by the success predicates below and by the metaworld dense reward.
# These used to live in metaworld_rewards.py, which forced a cycle: that
# module imports the runtime from here, so this module had to import the
# predicates back from it inside function bodies. The predicates are success
# criteria, not rewards, so they belong next to relational_goals_satisfied.
# ---------------------------------------------------------------------------

def _resolve_articulation_joint_spec(goal: dict) -> tuple[float, float, str] | None:
    """Return ``(low, high, joint_pattern)`` for an articulation goal, or ``None``.

    See :func:`~...tasks.common.suite_loader.articulation_joint_spec` (windows live
    in ``config/goal_thresholds.json``).
    """
    return articulation_joint_spec(goal["target"], goal["operation"], goal.get("layer"))


class _Operand(NamedTuple):
    """A goal operand resolved to the rows of its prototype's asset view.

    ``asset is None`` means "cannot resolve" (unknown operand, or prototype doesn't
    line up 1:1 with the task's envs); callers must treat that as a failed goal.
    """

    asset: object | None
    view_ids: torch.Tensor | None
    proto: str
    body_path: str


def _locate(env: ManagerBasedRLEnv, binding: TaskBinding, task_idx: int, entity_name: str) -> _Operand:
    """Resolve a goal operand name to the asset rows for one task's envs.

    ``entity_name`` may descend into an articulation body (``flat_stove_1/burnerplate``);
    only the leading segment names the prototype, and the body path is carried through
    for the caller to interpret. Backed by :class:`~.combined._LiberoRuntime`'s cached
    lookups, so this is a dict hit after the first call.
    """
    base_name, _, body_path = entity_name.partition("/")
    rt = _runtime(env, None)
    proto = rt.proto_names_for(binding).get(base_name)
    if proto is None or proto not in env.scene.keys():
        return _Operand(None, None, "", "")
    view_ids = rt.view_ids_for(env, proto, task_idx)
    if view_ids is None:
        return _Operand(None, None, proto, body_path)
    return _Operand(env.scene[proto], view_ids, proto, body_path)


def _resolve_entity_pose(
    env: ManagerBasedRLEnv, binding: TaskBinding, entity_name: str, task_idx: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """World pose ``((n, 3), (n, 4))`` of a goal entity, or ``None`` if unresolvable.

    A body path (``flat_stove_1/burnerplate``) selects that body by matching its last
    segment against ``body_names``, falling back to the root when unmatched. Quaternions
    are ``(x, y, z, w)``.
    """
    op = _locate(env, binding, task_idx, entity_name)
    if op.asset is None:
        return None
    if op.body_path:
        segment = op.body_path.split("/")[-1]
        i = _runtime(env, None).body_index_for(op.asset, op.proto, segment)
        if i is not None:
            return (
                _to_torch(op.asset.data.body_pos_w)[op.view_ids, i, :3],
                _to_torch(op.asset.data.body_quat_w)[op.view_ids, i, :4],
            )
    return (
        _to_torch(op.asset.data.root_pos_w)[op.view_ids, :3],
        _to_torch(op.asset.data.root_quat_w)[op.view_ids, :4],
    )


def _resolve_entity_pos(
    env: ManagerBasedRLEnv,
    binding: TaskBinding,
    entity_name: str,
    task_idx: int,
    local_offset: Sequence[float] | None = None,
) -> torch.Tensor | None:
    """World position ``(n_task_envs, 3)`` of a goal entity, or ``None`` if unresolvable.

    ``local_offset`` is a displacement in the entity's **own frame**, rotated by its
    world orientation before being added. Needed because LIBERO USD link frames sit
    at the asset root (not on the surface a goal actually means), and several
    fixtures spawn yawed, so the offset must rotate with the body.
    """
    pose = _resolve_entity_pose(env, binding, entity_name, task_idx)
    if pose is None:
        return None
    pos, quat = pose
    if local_offset is None:
        return pos
    offset = pos.new_tensor(tuple(float(v) for v in local_offset)).expand(pos.shape[0], 3)
    return pos + math_utils.quat_apply(quat, offset)


def _goal_target_pose(
    env: ManagerBasedRLEnv, binding: TaskBinding, goal: dict, task_idx: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """World target point of a goal plus the target's orientation, or ``None``.

    Position is link pose + ``target_offset`` + ``orientation`` -- two distinct,
    non-interchangeable offsets: ``target_offset`` is **asset-local** (rotated by the
    target's orientation; the support surface or cavity floor the relationship means),
    while ``orientation`` ("right"/"left"/"front", as in the source BDDL) is
    **world-frame** (plates etc. spawn yawed, so applying it locally would get the
    side wrong). The quaternion is returned too since ``xy_half_extents`` is a box
    in the target's frame and must be unrotated before testing.
    """
    pose = _resolve_entity_pose(env, binding, goal["target"], task_idx)
    if pose is None:
        return None
    pos, quat = pose
    consts = _runtime(env, None).goal_consts(goal)
    local_offset = consts["target_offset"]
    if local_offset is not None:
        pos = pos + math_utils.quat_apply(quat, local_offset.expand(pos.shape[0], 3))
    world_offset = consts["world_offset"]
    if world_offset is not None:
        pos = pos + world_offset
    return pos, quat


def _goal_target_pos(
    env: ManagerBasedRLEnv, binding: TaskBinding, goal: dict, task_idx: int
) -> torch.Tensor | None:
    """World position of a goal's target point; see :func:`_goal_target_pose`."""
    pose = _goal_target_pose(env, binding, goal, task_idx)
    return None if pose is None else pose[0]


def _xy_margin(
    goal: dict,
    ref_pos: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    half_extents: torch.Tensor | None = None,
) -> torch.Tensor:
    """Signed slack [m] on the xy test: ``< 0`` is inside, ``> 0`` is how far outside.

    ``xy_half_extents``, when present, makes this a box in the **target's own frame**
    (the container's interior footprint) instead of the isotropic ``xy_threshold``
    circle -- needed for e.g. two objects placed in one basket, where a circle small
    enough to fit inside fails a correct corner placement. Returns a signed margin
    rather than a bool so callers can distinguish "barely outside" from "way off".

    ``half_extents`` is the pre-built device tensor from :meth:`_LiberoRuntime.goal_consts`;
    pass ``None`` to rebuild it from ``goal`` (for callers without a runtime).
    """
    delta = ref_pos - target_pos
    half = goal.get("xy_half_extents")
    if half is None:
        xy_dist = torch.linalg.norm(delta[:, :2], dim=-1)
        return xy_dist - float(goal.get("xy_threshold", DEFAULT_GOAL_XY_THRESHOLD))
    local = math_utils.quat_apply_inverse(target_quat, delta)
    limits = local.new_tensor(tuple(float(v) for v in half)) if half_extents is None else half_extents
    return (local[:, :2].abs() - limits).amax(dim=-1)


def _xy_satisfied(
    goal: dict, ref_pos: torch.Tensor, target_pos: torch.Tensor, target_quat: torch.Tensor
) -> torch.Tensor:
    """Per-env bool: the ref object is over the target's footprint."""
    return _xy_margin(goal, ref_pos, target_pos, target_quat) < 0.0


def _resolve_entity_speed(
    env: ManagerBasedRLEnv, binding: TaskBinding, entity_name: str, task_idx: int
) -> torch.Tensor | None:
    """Linear speed ``(n_task_envs,)`` [m/s] of a goal entity, or ``None`` if unresolvable.

    Root speed only -- the at-rest test is about the manipulated rigid object; a
    body path names a static fixture part.
    """
    op = _locate(env, binding, task_idx, entity_name)
    if op.asset is None:
        return None
    lin_vel = getattr(op.asset.data, "root_lin_vel_w", None)
    if lin_vel is None:
        return None
    return torch.linalg.norm(_to_torch(lin_vel)[op.view_ids, :3], dim=-1)


def _articulation_joint_pos(
    env: ManagerBasedRLEnv, binding: TaskBinding, target_name: str, joint_pattern: str, task_idx: int
) -> torch.Tensor | None:
    """Joint position ``(n_task_envs,)`` of an articulation goal's joint, or ``None``."""
    op = _locate(env, binding, task_idx, target_name)
    if op.asset is None:
        return None
    joint_index = _runtime(env, None).joint_index_for(env, op.proto, joint_pattern)
    if joint_index is None:
        return None
    joint_pos = _to_torch(op.asset.data.joint_pos)[op.view_ids, joint_index]
    return joint_pos.squeeze(-1) if joint_pos.ndim > 1 else joint_pos


def _tilt_satisfied(goal: dict, ref_quat: torch.Tensor, upright_axis: torch.Tensor | None = None) -> torch.Tensor:
    """Per-env bool: the angle between the ref's ``upright_axis`` and world +z is in range.

    Position alone can't tell "bowl on plate" from "bowl upside-down on plate". ``max_tilt_deg``
    bounds the angle from above; ``min_tilt_deg`` bounds it from below for poses that are
    intentionally *not* upright (e.g. a bottle lying along a wine rack's rails). Either bound
    may be omitted; a goal with neither skips the test (right for "in the basket").

    ``upright_axis`` is the object-local axis pointing to world +z when upright. Defaults to
    local +z (right for bowl/plate/mug/book/bottle); HOPE cans/bottles are authored Y-up and
    stood up with ``roll = pi/2``, so for those it is ``[0, 1, 0]``.
    """
    max_tilt_deg, min_tilt_deg = goal.get("max_tilt_deg"), goal.get("min_tilt_deg")
    ok = torch.ones(ref_quat.shape[0], dtype=torch.bool, device=ref_quat.device)
    if max_tilt_deg is None and min_tilt_deg is None:
        return ok
    if upright_axis is None:
        upright_axis = ref_quat.new_tensor(tuple(float(v) for v in goal.get("upright_axis", (0.0, 0.0, 1.0))))
    up_local = upright_axis.expand(ref_quat.shape[0], 3)
    cos_tilt = math_utils.quat_apply(ref_quat, up_local)[:, 2]
    if max_tilt_deg is not None:
        ok &= cos_tilt >= math.cos(math.radians(float(max_tilt_deg)))
    if min_tilt_deg is not None:
        ok &= cos_tilt <= math.cos(math.radians(float(min_tilt_deg)))
    return ok


#: Scene keys the release test reads.  Absent (a layout without them) means the
#: test is skipped rather than failing, matching the other resolvers here.
LIBERO_ROBOT_KEY = "franka_robot"
LIBERO_EE_FRAME_KEY = "franka_ee_frame"


def _resolve_gripper_state(env: ManagerBasedRLEnv) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """``(jaw opening (n_envs,), TCP position (n_envs, 3))``; either element may be ``None``.

    Opening is the sum of the two finger joints (span between the pads: 0.0 closed,
    0.08 fully open), taken as magnitudes so either finger's sign convention works.
    """
    # Cached once per sim step: whole-robot, but up to 16 relational goals ask per step.
    step_key = getattr(env, "_sim_step_counter", None)
    cache = getattr(env, "_libero_gripper_state_cache", None)
    if cache is not None and step_key is not None and cache[0] == step_key:
        return cache[1]

    opening = None
    robot = getattr(env.scene, "articulations", {}).get(LIBERO_ROBOT_KEY)
    if robot is not None:
        joint_ids = getattr(env, "_libero_finger_joint_ids", None)
        if joint_ids is None:
            # find_joints is a regex scan over the articulation's joint names.
            joint_ids, _ = robot.find_joints(["panda_finger.*"], preserve_order=True)
            env._libero_finger_joint_ids = joint_ids
        if len(joint_ids) >= 2:
            joint_pos = _to_torch(robot.data.joint_pos)
            opening = joint_pos[:, joint_ids[0]].abs() + joint_pos[:, joint_ids[1]].abs()
    ee_pos = None
    frame = getattr(env.scene, "sensors", {}).get(LIBERO_EE_FRAME_KEY)
    if frame is not None:
        ee_pos = _to_torch(frame.data.target_pos_w)[:, 0, :3]

    result = (opening, ee_pos)
    if step_key is not None:
        env._libero_gripper_state_cache = (step_key, result)
    return result


def _released_satisfied(
    env: ManagerBasedRLEnv, goal: dict, ref_pos: torch.Tensor, env_ids: torch.Tensor
) -> torch.Tensor:
    """Per-env bool: the gripper has let go of the ref object.

    ``max_speed`` alone can't separate "placed and settled" from "held motionless
    in the jaws". ``require_released`` fills that gap with ``min_opening`` (jaw span
    too wide to be gripping) and/or ``min_clearance`` (TCP too far from the object);
    either is sufficient. Skipped (not failed) when the robot/EE frame isn't in the scene.
    """
    spec = goal.get("require_released")
    if not spec:
        return torch.ones(ref_pos.shape[0], dtype=torch.bool, device=ref_pos.device)
    opening, ee_pos = _resolve_gripper_state(env)
    released = None
    min_opening = spec.get("min_opening")
    if opening is not None and min_opening is not None:
        released = opening[env_ids] > float(min_opening)
    min_clearance = spec.get("min_clearance")
    if ee_pos is not None and min_clearance is not None:
        clear = torch.linalg.norm(ee_pos[env_ids] - ref_pos, dim=-1) > float(min_clearance)
        released = clear if released is None else (released | clear)
    if released is None:
        return torch.ones(ref_pos.shape[0], dtype=torch.bool, device=ref_pos.device)
    return released


# ---------------------------------------------------------------------------
# Operand resolution guard
# ---------------------------------------------------------------------------


#: Operand names already warned about, so an unresolvable goal says so once
#: rather than every step.
_UNRESOLVED_WARNED: set[tuple[str, str]] = set()


def _warn_unresolved(binding: TaskBinding, entity_name: str) -> None:
    """Say once that a goal operand does not resolve in the scene.

    An unresolvable operand scores ``False`` forever, which looks like a policy that
    never solves the task -- hard to distinguish from a real misconfiguration without
    this warning.
    """
    key = (getattr(binding, "name", "?"), entity_name)
    if key in _UNRESOLVED_WARNED:
        return
    _UNRESOLVED_WARNED.add(key)
    logger.warning(
        "libero goal operand '%s' of task '%s' does not resolve in the scene; "
        "every goal naming it scores False for the whole run",
        entity_name,
        key[0],
    )


def articulation_goals_satisfied(
    env: ManagerBasedRLEnv, tasks: list[TaskBinding] | None = None
) -> torch.Tensor:
    """Per-env bool: ALL articulation operation goals of the env's task hold.

    Vacuously ``True`` for envs whose task has no operation goal. Joint predicates use
    the same ``(low, high)`` ranges as the original success evaluation (open drawer,
    close microwave, turn on stove, ...).
    """
    tasks = _resolve_tasks(env, tasks)
    rt = _runtime(env, tasks)
    ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for task_idx, binding in enumerate(tasks):
        op_goals = [g for g in getattr(binding, "goals", ()) if "operation" in g]
        if not op_goals:
            continue
        env_ids = rt.env_ids_for(task_idx)
        if env_ids.numel() == 0:
            continue
        for goal in op_goals:
            spec = _resolve_articulation_joint_spec(goal)
            if spec is None:
                # NOTE: unlike a relational goal, this skips (== treats as
                # satisfied) rather than failing closed.  Kept for now so the
                # diagnostic run reports the truth about the current build; the
                # warning is what tells us whether the asymmetry ever fires.
                _warn_unresolved(binding, f"{goal['target']}:{goal['operation']}:{goal.get('layer', '-')}")
                continue
            low, high, joint_pattern = spec
            joint_pos = _articulation_joint_pos(env, binding, goal["target"], joint_pattern, task_idx)
            if joint_pos is None:
                _warn_unresolved(binding, f"{goal['target']}:joint:{joint_pattern}")
                continue
            ok[env_ids] &= (joint_pos >= low) & (joint_pos <= high)
    return ok


def relational_goals_satisfied(
    env: ManagerBasedRLEnv, tasks: list[TaskBinding] | None = None, speed_threshold: float | None = 0.4
) -> torch.Tensor:
    """Per-env bool: EVERY relationship goal of the env's task holds.

    Driven by the task's ``goals`` list (``$LIBERO_CONFIG_DIR/<suite>.json``). Each
    goal brings ``ref_obj``, ``target``, ``xy_threshold``, ``height_threshold``,
    ``height_diff``, and optional terms that make the test mean *placed* rather than
    merely *nearby*: ``target_offset`` (asset-local support surface/cavity floor, see
    :func:`_goal_target_pos`), ``orientation`` (world-frame side offset, e.g. "right of
    the plate"), ``max_speed`` (per-goal at-rest bound), ``max_tilt_deg``/``min_tilt_deg``/
    ``upright_axis`` (see :func:`_tilt_satisfied`), ``xy_half_extents`` (container
    footprint as a box in the target's frame, see :func:`_xy_satisfied`), and
    ``require_released`` (see :func:`_released_satisfied`). All optional; a goal
    omitting them falls back to plain distance.

    Evaluates every relational goal (not just the first), so multi-object goals like
    "put both X and Y in the basket" are scored on all of them.

    Vacuously ``True`` for envs whose task has no relational goal, matching
    :func:`articulation_goals_satisfied` (the caller ANDs the two and requires at
    least one goal of either kind). An unresolvable operand yields ``False``, not a skip.
    """
    tasks = _resolve_tasks(env, tasks)
    rt = _runtime(env, tasks)
    ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for task_idx, binding in enumerate(tasks):
        goals = [g for g in getattr(binding, "goals", ()) if _is_relational(g)]
        if not goals:
            continue
        env_ids = rt.env_ids_for(task_idx)
        if env_ids.numel() == 0:
            continue
        for goal_idx, goal in enumerate(goals):
            ref_pose = _resolve_entity_pose(env, binding, goal["ref_obj"], task_idx)
            target_pose = _goal_target_pose(env, binding, goal, task_idx)
            if ref_pose is None or target_pose is None:
                ok[env_ids] = False
                _warn_unresolved(binding, goal["ref_obj"] if ref_pose is None else goal["target"])
            ref_pos, ref_quat = ref_pose
            target_pos, target_quat = target_pose
            consts = rt.goal_consts(goal)
            xy_margin = _xy_margin(goal, ref_pos, target_pos, target_quat, consts["xy_half_extents"])
            # height_diff is the *expected* rise of ref over target, not a slack
            height_err = (ref_pos[:, 2] - target_pos[:, 2] - float(goal.get("height_diff", 0.0))).abs()
            height_margin = height_err - float(goal.get("height_threshold", DEFAULT_GOAL_HEIGHT_THRESHOLD))
            terms = {"xy": xy_margin < 0.0, "height": height_margin < 0.0}
            # Per-goal "at rest" bound; falls back to the caller's global value.
            max_speed = goal.get("max_speed", speed_threshold)
            if max_speed is not None:
                speed = _resolve_entity_speed(env, binding, goal["ref_obj"], task_idx)
                if speed is not None:
                    terms["speed"] = speed < float(max_speed)
            terms["tilt"] = _tilt_satisfied(goal, ref_quat, consts["upright_axis"])
            terms["released"] = _released_satisfied(env, goal, ref_pos, env_ids)
            reached = terms["xy"]
            for name, mask in terms.items():
                if name != "xy":
                    reached = reached & mask
            ok[env_ids] &= reached
    return ok


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


#: Require the success predicate to have been FALSE once before it may count.
REQUIRE_SUCCESS_TRANSITION = settings.truthy("LIBERO_SUCCESS_REQUIRE_TRANSITION", "1")

_TRANSITION_ATTR = "_libero_success_seen_false"


def _require_transition(env: ManagerBasedRLEnv, raw: torch.Tensor) -> torch.Tensor:
    """Drop "successes" the episode was already in at reset.

    The success predicate is a *state* test, and some LIBERO initial layouts already
    satisfy it (source object starts near the target, or a "close the ..." goal
    starts closed), so those envs would report success on step 1 with a random
    policy, forever. Latching on "predicate has been false at least once this
    episode" requires a real transition into the goal state (an episode that starts
    satisfied, is disturbed, and recovers still counts).
    """
    if not REQUIRE_SUCCESS_TRANSITION:
        return raw
    seen_false = getattr(env, _TRANSITION_ATTR, None)
    if seen_false is None or seen_false.shape != raw.shape or seen_false.device != raw.device:
        seen_false = torch.zeros_like(raw)
        setattr(env, _TRANSITION_ATTR, seen_false)
    # episode_length_buf is incremented before the managers run, so a freshly
    # reset env is at 1 here; clear the latch for exactly those envs.
    episode_length = getattr(env, "episode_length_buf", None)
    if episode_length is not None:
        seen_false &= episode_length > 1
    seen_false |= ~raw
    return raw & seen_false


def libero_task_success(
    env: ManagerBasedRLEnv, tasks: list[TaskBinding] | None = None, speed_threshold: float | None = 0.4
) -> torch.Tensor:
    """Per-env task success: EVERY goal predicate the task's config declares.

    Combines every relationship goal (:func:`relational_goals_satisfied`) and every
    articulation operation goal (:func:`articulation_goals_satisfied`) from
    ``$LIBERO_CONFIG_DIR/<suite>.json``. A task declaring only one kind is judged on
    that kind alone; one declaring neither can never succeed. A predicate already
    holding at reset does not count -- see :func:`_require_transition`
    (disable via ``LIBERO_SUCCESS_REQUIRE_TRANSITION=0``).
    """

    rt = _runtime(env, tasks)
    rel_ok = relational_goals_satisfied(env, tasks, speed_threshold)
    art_ok = articulation_goals_satisfied(env, tasks)
    # guard: a task with no goal predicate at all can never succeed
    result = rel_ok & art_ok & (rt.has_rel_goal | rt.has_op_goal)
    result = _require_transition(env, result)
    # Publish the pre-reset mask: every reward mode evaluates this each step
    # (the goal/metaworld task_success bonus or the world_state_tracking_reward aggregated
    # payouts), so the task-SR curriculum term can read it when no success
    # termination is active (training — success does not end the episode).
    stash_task_success(env, result)
    return result


def command_finished(env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.999) -> torch.Tensor:
    """True where the demo command's ``progress`` metric is at/past ``threshold``.

    Used as the training-time ``command_depleted`` truncation (episode ends when the
    demo is exhausted).
    """
    term = resolve_command_term(env, command_name)
    progress = term.metrics.get("progress")
    if progress is None:
        raise RuntimeError(f"command '{command_name}' does not expose a progress metric")
    return progress >= threshold


def joint_vel_exceeds_scaled_limit(
    env: ManagerBasedRLEnv,
    scale: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any joint velocity exceeds ``scale`` x its soft limit."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    limits = asset.data.soft_joint_vel_limits[:, joint_ids] * float(scale)
    return torch.any(torch.abs(asset.data.joint_vel[:, joint_ids]) > limits, dim=1)


def libero_success_after_command(
    env: ManagerBasedRLEnv,
    tasks: list[TaskBinding] | None = None,
    speed_threshold: float | None = 0.4,
    command_name: str = "source_action",
    threshold: float = 0.0,
) -> torch.Tensor:
    """Success proxy AND the demo command at/past ``threshold`` progress.

    Used by the aggregated demo-tracking reward payouts. With ``threshold <= 0``
    (or no demo command) this reduces to the plain success proxy.
    """
    success = libero_task_success(env, tasks, speed_threshold)
    if threshold <= 0.0:
        return success
    if getattr(env, "command_manager", None) is None:
        return success
    try:
        term = resolve_command_term(env, command_name)
    except Exception:
        return success
    progress = getattr(term, "metrics", {}).get("progress")
    if progress is None:
        return success
    return success & (progress >= threshold)


# ---------------------------------------------------------------------------
# Rewards (per-env gather)
# ---------------------------------------------------------------------------


def libero_task_success_reward(
    env: ManagerBasedRLEnv, tasks: list[TaskBinding] | None = None, speed_threshold: float | None = 0.4
) -> torch.Tensor:
    """Sparse task-success bonus: 1.0 where the geometric success proxy holds."""
    return libero_task_success(env, tasks, speed_threshold).float()


def libero_reach_primary(
    env: ManagerBasedRLEnv,
    tasks: list[TaskBinding] | None = None,
    std: float = 0.1,
    ee_frame_name: str = "franka_ee_frame",
) -> torch.Tensor:
    """Demo-free reach shaping: ``1 - tanh(||ee - primary|| / std)`` per env."""
    rt = _runtime(env, tasks)
    primary_pos, _ = _gather_object_state(env, rt.primary_envs)
    ee_frame = env.scene[ee_frame_name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[:, 0, :3]
    dist = torch.linalg.norm(ee_pos_w - primary_pos, dim=-1)
    has_primary = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for envs in rt.primary_envs.values():
        has_primary[envs] = True
    return (1.0 - torch.tanh(dist / std)) * has_primary.float()


def libero_primary_lifted(
    env: ManagerBasedRLEnv,
    tasks: list[TaskBinding] | None = None,
    minimal_height: float = 0.95,
) -> torch.Tensor:
    """Demo-free lift bonus: 1.0 where the primary object is above ``minimal_height`` [m]."""
    rt = _runtime(env, tasks)
    primary_pos, _ = _gather_object_state(env, rt.primary_envs)
    lifted = primary_pos[:, 2] > minimal_height
    has_primary = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for envs in rt.primary_envs.values():
        has_primary[envs] = True
    return (lifted & has_primary).float()


def libero_primary_to_target(
    env: ManagerBasedRLEnv,
    tasks: list[TaskBinding] | None = None,
    std: float = 0.1,
) -> torch.Tensor:
    """Demo-free place shaping: ``1 - tanh(||primary - target|| / std)`` per env."""
    rt = _runtime(env, tasks)
    primary_pos, _ = _gather_object_state(env, rt.primary_envs)
    target_pos, _ = _gather_object_state(env, rt.target_envs)
    dist = torch.linalg.norm(primary_pos - target_pos, dim=-1)
    has_both = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for envs in rt.primary_envs.values():
        has_both[envs] = True
    has_target = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for envs in rt.target_envs.values():
        has_target[envs] = True
    has_both &= has_target
    return (1.0 - torch.tanh(dist / std)) * has_both.float()


# ---------------------------------------------------------------------------
# Reset event
# ---------------------------------------------------------------------------


def reset_libero_prototypes(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    tasks: list[TaskBinding] | None = None,
    pose_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Write each task's authored object poses into its own envs, with jitter.

    For every task, restricts the reset envs to that task's envs
    (``env % n_tasks == task_idx``), locates each object's shared prototype view via
    :meth:`filter_reset_ids`, and writes this task's ``(x, y, z, w)`` pose plus a
    small uniform position jitter.
    """
    tasks = _resolve_tasks(env, tasks)
    selector = env.scene.selector
    n_tasks = len(tasks)
    device = env.device
    ranges = torch.tensor([(pose_range or {}).get(k, (0.0, 0.0)) for k in ("x", "y", "z")], device=device)
    for task_idx, task in enumerate(tasks):
        task_envs = env_ids[(env_ids % n_tasks) == task_idx]
        if task_envs.numel() == 0:
            continue
        for binding in task.object_bindings:
            global_ids, view_ids = selector.filter_reset_ids(binding.proto_name, task_envs)
            if global_ids.numel() == 0:
                continue
            asset = env.scene[binding.proto_name]
            count = global_ids.shape[0]
            jitter = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (count, 3), device=device)
            pose = torch.zeros((count, 7), device=device)
            pose[:, :3] = torch.tensor(binding.pos, device=device) + env.scene.env_origins[global_ids] + jitter
            pose[:, 3:7] = torch.tensor(binding.rot, device=device)
            asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=view_ids)
            asset.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((count, 6), device=device), env_ids=view_ids
            )
            if binding.is_articulation:
                joint_pos = wp.to_torch(asset.data.default_joint_pos)[view_ids].clone()
                joint_vel = wp.to_torch(asset.data.default_joint_vel)[view_ids].clone()
                asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=view_ids)
                asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=view_ids)
