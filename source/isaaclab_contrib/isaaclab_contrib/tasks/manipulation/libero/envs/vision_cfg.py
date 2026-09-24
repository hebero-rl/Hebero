# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision variant of the LIBERO DGPO env: Theia patch tokens replace the object-pose buffer.

The state-based env hands the actor ``buffer_pose`` -- 234 dims of ground-truth object and
target poses read straight out of the simulator. That is the one input a real robot cannot
have. This variant deletes it and mounts two cameras (agentview + eye-in-hand) whose RGB is
encoded by a **frozen** Theia-tiny ViT, published as patch tokens ``(B, P, 192)`` per camera
and concatenated to ``(B, P, 384)`` in the ``perception`` group.

Pooling is deliberately NOT done here. The token sequence is consumed by an attention-pool
head inside :class:`~isaaclab_rl.rsl_rl.models.TokenPoolModel`, so which patches matter is
learned with the policy loss rather than fixed to a uniform mean.

The critic keeps the full ``privileged_proprio`` group (demo-tracking residuals incl.
``object_target_pose_diff``): asymmetric actor-critic, so the value function is still a
function of the state while the policy is a function of what a camera can see.

``task_multi_hot`` stays in the actor's policy group. It is the task *specification* --
what the robot has been asked to do -- not privileged perception, and several LIBERO
suites share a scene, so an image cannot disambiguate them.

Observation widths:
policy 43 = last_action 7 + task_multi_hot 36, proprio 23, perception (P, 384),
privileged_proprio 252 -- so actor 450 and critic 702. The token count ``P`` is
64 at 128x128, compared with 36 at 100x100. That changes no layer
width, because the attention pool sums over ``P``, so a checkpoint is portable across
camera resolutions.

Environment variables
---------------------

* ``CAMERA_HEIGHT`` / ``CAMERA_WIDTH`` (default 128) -- render resolution per camera.
  128 with Theia's patch16 gives an even 8x8 = 64 patch tokens per camera.
* ``THEIA_MODEL_NAME`` (default ``theia-tiny-patch16-224-cdiv``) -- any Theia checkpoint
  from the Isaac Lab model zoo; the token width follows the checkpoint (tiny = 192).
* ``THEIA_DEVICE`` -- device to hold the frozen encoder on. Defaults to the env device.

Gym ids: ``Isaac-Hebero-All-Vision-v0`` (train), plus ``*-Vision-Play-v0``
variants. Training requires ``--enable_cameras``.
"""

from __future__ import annotations

import re

import torch

import isaaclab.envs.mdp as env_mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from .dgpo_env_cfg import make_libero_dgpo_env_cfg, make_libero_dgpo_play_cfg
from .dgpo_mdp_cfg import DgpoPolicyCfg, DgpoPrivilegedProprioCfg, DgpoProprioCfg
from ... import settings

# Theia-tiny at patch16.  Kept at the 224-trained checkpoint and fed a smaller image with
# ``interpolate_pos_encoding=True`` (what ``mdp.image_features`` does), which is how the
# Theia authors intend off-resolution inference.
THEIA_MODEL_NAME = settings.text("THEIA_MODEL_NAME") or "theia-tiny-patch16-224-cdiv"

# At 128x128, patch16 divides the image evenly (8x8 = 64 tokens per
# camera against 100's ragged 6x6 = 36), so no patch straddles the image border and the
# actor sees ~1.8x the spatial resolution.  The cost is real -- rendering and the ViT
# forward both scale with pixels, and the token count sets the attention-pool cost -- so
# it is a knob, not a constant.  224 (Theia's native) is 196 tokens per camera and roughly
# 3x this one's perception cost; useful for a final run, heavy for a sweep.
CAMERA_HEIGHT = settings.integer("CAMERA_HEIGHT", 128)
CAMERA_WIDTH = settings.integer("CAMERA_WIDTH", 128)

# ``model_device`` is omitted rather than passed as None: ``image_features.__init__``
# reads it with ``cfg.params.get("model_device", env.device)``, so an explicit None would
# override the env-device default with None instead of falling back to it.
_THEIA_PARAMS: dict = {"data_type": "rgb", "model_name": THEIA_MODEL_NAME}
if settings.text("THEIA_DEVICE"):
    _THEIA_PARAMS["model_device"] = settings.text("THEIA_DEVICE")


def _camera_cfg(
    prim_path: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    focal_length: float,
    clipping_range: tuple[float, float],
) -> CameraCfg:
    """Build an RGB-only camera matching the LIBERO/robosuite intrinsics.

    ``CameraCfg`` is used directly rather than ``TiledCameraCfg``: the tiled class is
    deprecated in this Isaac Lab -- ``Camera`` already carries the vectorized rendering
    path through the renderer abstraction.
    """
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,  # every render; render_interval is pinned to the control step
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=15.0,
            clipping_range=clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
    )


# ---------------------------------------------------------------------------
# Agentview pose, per workspace
# ---------------------------------------------------------------------------

#: LIBERO's own agentview pose per scene, ``usd_dir -> (pos, quat)`` with the quaternion
#: **scalar-first** exactly as MuJoCo writes it, copied verbatim from
#: ``libero/envs/problems/libero_*_manipulation.py``.  There is no single agentview: the
#: four workspaces this benchmark spans put the camera at different heights *and*
#: different pitches (the table scenes use one quaternion family, floor/living-room
#: another).  ``table`` shares ``kitchen_table``'s USD directory and its camera.
#:
#: These are LIBERO **world**-frame poses.  This repo rigidly translates every fixture
#: and object by ``workspace_shift = ROBOT_BASE_KITCHEN - task_robot_base`` so all tasks
#: share one robot base, so the camera has to move with them -- see :func:`agentview_pose`.
LIBERO_AGENTVIEW_POSES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
    "kitchen_table": (
        (0.6586131746834771, 0.0, 1.6103500240372423),
        (0.6380177736282349, 0.3048497438430786, 0.30485090613365173, 0.6380177736282349),
    ),
    "study_table": (
        (0.4586131746834771, 0.0, 1.6103500240372423),
        (0.6380177736282349, 0.3048497438430786, 0.30485090613365173, 0.6380177736282349),
    ),
    "living_room_table": (
        (0.6065773716836134, 0.0, 0.96),
        (0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979),
    ),
    "floor": (
        (0.8965773716836134, 5.216182733499864e-07, 0.65),
        (0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979),
    ),
}


def wxyz_to_xyzw(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Reorder a MuJoCo/LIBERO quaternion to the ``(x, y, z, w)`` this stack uses.

    LIBERO stores camera quaternions the MuJoCo way, scalar first; ``CameraCfg.OffsetCfg.rot``
    and ``Camera.set_world_poses`` both want scalar last.  Passed through unreordered the
    agentview quaternion loses its entire 31.9 deg downward pitch -- it renders a
    horizontal view of the backdrop, which is what the cameras were doing.
    """
    w, x, y, z = q
    return (x, y, z, w)


def _workspace_of(binding) -> str:
    """USD directory of a task's workspace fixture, from its prototype name.

    ``build_combined_tasks`` names fixture prototypes with ``_unique_name(usd_dir)``,
    which is the directory itself or ``<dir>_<n>`` when several fixture poses share it.
    """
    return re.sub(r"_\d+$", "", binding.fixture_proto)


def agentview_pose(binding) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Env-frame agentview pose for one task: LIBERO's scene pose + its workspace shift."""
    workspace = _workspace_of(binding)
    if workspace not in LIBERO_AGENTVIEW_POSES:
        raise KeyError(
            f"no LIBERO agentview pose for workspace {workspace!r} (task {binding.name!r}); "
            f"add it to LIBERO_AGENTVIEW_POSES -- guessing one silently mis-frames the camera"
        )
    pos, quat = LIBERO_AGENTVIEW_POSES[workspace]
    shift = tuple(float(v) for v in binding.workspace_shift)
    return (pos[0] + shift[0], pos[1] + shift[1], pos[2] + shift[2]), wxyz_to_xyzw(quat)


def set_agentview_poses(env, env_ids, sensor_name: str = "agentview_cam") -> None:
    """Place each env's agentview where its own task's workspace puts it.

    One ``CameraCfg`` carries one offset, but ``env i`` runs ``task i % n_tasks`` and the
    four workspaces disagree about where the camera belongs by up to 0.65 m in height and
    7 degrees in pitch.  A single offset therefore cannot be right for more than one
    workspace, so the pose is written per env once the scene exists.

    ``env_ids`` takes no default even though startup mode passes ``None``: the event
    manager sorts a term's parameters into with- and without-defaults and treats the
    first two positionally, so a default here shifts ``sensor_name`` into the slot it
    checks against ``params`` and the term fails to resolve.
    """
    camera = env.scene.sensors[sensor_name]
    bindings = env.cfg.dgpo_task_bindings
    n_tasks = len(bindings)
    if env_ids is None:
        ids = torch.arange(env.num_envs, device=env.device)
    else:
        ids = torch.as_tensor(env_ids, device=env.device)

    poses = [agentview_pose(bindings[int(i) % n_tasks]) for i in ids.tolist()]
    positions = torch.tensor([p for p, _ in poses], dtype=torch.float32, device=env.device)
    orientations = torch.tensor([q for _, q in poses], dtype=torch.float32, device=env.device)
    # the cfg offset is env-relative; set_world_poses is not
    positions = positions + env.scene.env_origins[ids]
    camera.set_world_poses(positions, orientations, env_ids=ids, convention="opengl")


def libero_agentview_camera_cfg() -> CameraCfg:
    """Agentview camera; the pose here is a placeholder, overwritten per env.

    Every env gets the ``kitchen_table`` pose at spawn and :func:`set_agentview_poses`
    then moves each one to its own task's workspace.  The placeholder is the pose 24 of
    the 40 tasks want, so a run whose startup event failed to fire still renders
    something recognisable for most tasks rather than a view of nothing -- but it is
    wrong for the other 16, which is what the event exists to fix.

    ``clipping_range`` is widened from LIBERO's per-scene value: one range now has to
    cover every workspace, and the far plane has to reach the floor tasks' 1.9 m.
    """
    pos, rot = LIBERO_AGENTVIEW_POSES["kitchen_table"]
    return _camera_cfg(
        prim_path="{ENV_REGEX_NS}/agentview_camera",
        pos=pos,
        rot=wxyz_to_xyzw(rot),
        focal_length=18.0,
        clipping_range=(0.05, 3.0),
    )


def libero_eye_in_hand_camera_cfg() -> CameraCfg:
    """Eye-in-hand camera rigidly attached to ``panda_hand``."""
    return _camera_cfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/eye_in_hand_camera",
        pos=(0.05, 0.0, 0.0),
        # robosuite's mount quaternion, scalar-first as MuJoCo writes it
        rot=wxyz_to_xyzw((0.0, 0.707108, 0.707108, 0.0)),
        focal_length=9.77,
        clipping_range=(0.001, 1.0),
    )


@configclass
class DgpoVisionPolicyCfg(DgpoPolicyCfg):
    """Policy group without ``buffer_pose``: last_action + task_multi_hot → 43.

    ``buffer_pose`` is the ground-truth object/target pose read -- exactly what the
    ``perception`` group is here to replace, so keeping it would make the vision arm
    strictly more informed than the state-based one rather than differently informed.
    ``task_multi_hot`` stays; see the module docstring.
    """

    buffer_pose = None


@configclass
class DgpoPerceptionCfg(ObsGroup):
    """Frozen Theia-tiny patch tokens, ``(B, P, 192)`` per camera → ``(B, P, 384)``.

    Concatenation is along the feature dim, so the two cameras contribute the same patch
    positions with stacked channels rather than a longer sequence -- the attention pool
    then weighs a *location*, jointly across both views.
    """

    agentview_features = ObsTerm(
        func=env_mdp.image_features,
        params={"sensor_cfg": SceneEntityCfg("agentview_cam"), **_THEIA_PARAMS},
    )
    eye_in_hand_features = ObsTerm(
        func=env_mdp.image_features,
        params={"sensor_cfg": SceneEntityCfg("eye_in_hand_cam"), **_THEIA_PARAMS},
    )

    def __post_init__(self):
        """Disable corruption; concatenate the two cameras along the token feature dim."""
        self.enable_corruption = False
        self.concatenate_terms = True
        self.concatenate_dim = -1


@configclass
class DgpoVisionObservationsCfg:
    """Observation groups for the vision variant (actor sees ``perception``, not object poses)."""

    policy: DgpoVisionPolicyCfg = DgpoVisionPolicyCfg()
    proprio: DgpoProprioCfg = DgpoProprioCfg()
    perception: DgpoPerceptionCfg = DgpoPerceptionCfg()
    privileged_proprio: DgpoPrivilegedProprioCfg = DgpoPrivilegedProprioCfg()


def _install_vision(env_cfg) -> None:
    """Mount both cameras and swap in the vision observation groups."""
    env_cfg.scene.agentview_cam = libero_agentview_camera_cfg()
    env_cfg.scene.eye_in_hand_cam = libero_eye_in_hand_camera_cfg()
    # The agentview belongs to the workspace, not the env: place it per task once the
    # scene exists.  Startup rather than reset -- the fixtures never move.
    env_cfg.events.agentview_pose = EventTerm(func=set_agentview_poses, mode="startup", params={})
    # Re-render after a reset so the first observation of an episode shows the new scene
    # rather than the last frame of the previous one.  (``rerender_on_reset`` is the
    # deprecated spelling of this in current Isaac Lab.)
    env_cfg.num_rerenders_on_reset = 1
    env_cfg.observations = DgpoVisionObservationsCfg()
    # The state-based contract in ``dgpo_layout`` no longer describes this env's actor;
    # say so rather than leaving the stale 300/552 in the run metadata.
    meta = dict(getattr(env_cfg, "dgpo_layout", {}) or {})
    meta.update(
        {
            "name": f"{meta.get('name', 'libero_dgpo')}_vision",
            "actor_obs_dim": None,  # flat 1D width + pooled tokens; see TokenPoolModel
            "critic_obs_dim": None,
            "vision": {
                "encoder": THEIA_MODEL_NAME,
                "cameras": ("agentview_cam", "eye_in_hand_cam"),
                "resolution": (CAMERA_HEIGHT, CAMERA_WIDTH),
            },
        }
    )
    env_cfg.dgpo_layout = meta


def make_libero_dgpo_vision_env_cfg(*args, **kwargs) -> type:
    """Train cfg: :func:`~.dgpo_env_cfg.make_libero_dgpo_env_cfg` plus cameras and Theia obs."""
    base_cls = make_libero_dgpo_env_cfg(*args, **kwargs)
    parent_post_init = base_cls.__post_init__

    def _post_init(self):
        parent_post_init(self)
        _install_vision(self)

    return configclass(type("_LiberoDgpoOscVisionEnvCfg", (base_cls,), {"__post_init__": _post_init}))


def make_libero_dgpo_vision_play_cfg(*args, **kwargs) -> type:
    """Play/eval cfg: one env per task, cameras mounted, no observation corruption."""
    base_cls = make_libero_dgpo_play_cfg(*args, **kwargs)
    parent_post_init = base_cls.__post_init__

    def _post_init(self):
        parent_post_init(self)
        _install_vision(self)
        self.observations.policy.enable_corruption = False
        self.observations.proprio.enable_corruption = False
        self.observations.perception.enable_corruption = False
        self.observations.privileged_proprio.enable_corruption = False

    return configclass(type("_LiberoDgpoOscVisionPlayEnvCfg", (base_cls,), {"__post_init__": _post_init}))
