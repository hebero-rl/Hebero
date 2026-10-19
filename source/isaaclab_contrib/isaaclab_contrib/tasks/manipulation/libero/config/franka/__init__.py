# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for Hebero tasks in the DGPO training framework.

Every environment is built by
:func:`~...envs.dgpo_env_cfg.make_libero_dgpo_env_cfg` /
:func:`~...envs.dgpo_env_cfg.make_libero_dgpo_play_cfg`: harvest+cloner scene,
OSC actions (dim 7), and DGPO obs groups (actor=300 / critic=552). Suite order
for All matches :data:`~...dgpo_layout.DGPO_ABC_HARVEST_SUITES`
(long → object → spatial → goal). Importing this package registers:

* ``Isaac-Hebero-All-State-v0`` — train all four suites (40 tasks).
* ``Isaac-Hebero-Long-State-v0`` — train the long-horizon suite only
  (10 tasks; ``libero_long`` is ``libero_10`` in assignment space).
* ``Isaac-Hebero-All-State-Play-v0`` — play/eval all suites.
* ``Isaac-Hebero-{Long,Object,Spatial,Goal}-State-Play-v0`` — single-suite
  play (10 tasks).
* ``Isaac-Hebero-Spatial-Goal-State-Play-v0`` — spatial + goal play
  (20 tasks).
* ``Isaac-Hebero-Object-Long-State-Play-v0`` — long + object play
  (20 tasks; DGPO relative order).

Play variants use one env per task and disable observation corruption.
Subset train configs still exist on :mod:`.libero_dgpo_env_cfg` for
programmatic use (no separate gym ids).
"""

from __future__ import annotations

import gymnasium as gym

from . import agents

_DGPO_ENTRY = "isaaclab.envs:ManagerBasedRLEnv"
_AGENTS = f"{agents.__name__}.rsl_rl_ppo_cfg"
_CFG = f"{__name__}.libero_dgpo_env_cfg"


def _register_dgpo(gym_id: str, env_cfg_attr: str) -> None:
    """Register a DGPO OSC env (train or play) with all algorithm runner cfgs.

    ``rsl_rl_cfg_entry_point`` (default) selects IW-ABC; pass
    ``--agent`` to ``train.py`` to select another method:
    ``rsl_rl_mtppo_cfg_entry_point`` (MT-PPO, the base config),
    ``rsl_rl_dapg_cfg_entry_point`` (MT-DAPG),
    ``rsl_rl_ppo_rfcl_cfg_entry_point`` is MT-PPO + RFCL's reset-state
    curriculum (on-policy, no BC loss) -- isolates that mechanism against ABC.
    ``rsl_rl_iw_abc_cfg_entry_point`` explicitly selects IW-ABC.
    ``rsl_rl_dgpo_cfg_entry_point`` and ``rsl_rl_abc_cfg_entry_point`` remain
    compatibility aliases for IW-ABC -- the config
    used to be called DGPO+ABC, with a separate no-BC tier named DGPO that was
    identical to MT-PPO.
    """
    gym.register(
        id=gym_id,
        entry_point=_DGPO_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_CFG}:{env_cfg_attr}",
            "rsl_rl_cfg_entry_point": f"{_AGENTS}:LiberoAllIwAbcRunnerCfg",
            "rsl_rl_iw_abc_cfg_entry_point": f"{_AGENTS}:LiberoAllIwAbcRunnerCfg",
            "rsl_rl_dgpo_cfg_entry_point": f"{_AGENTS}:LiberoAllIwAbcRunnerCfg",
            # back-compat: this key selected the same config when it was DGPO+ABC
            "rsl_rl_abc_cfg_entry_point": f"{_AGENTS}:LiberoAllIwAbcRunnerCfg",
            "rsl_rl_dapg_cfg_entry_point": f"{_AGENTS}:LiberoAllMtDapgRunnerCfg",
            "rsl_rl_mtppo_cfg_entry_point": f"{_AGENTS}:LiberoAllMtPPORunnerCfg",
            "rsl_rl_ppo_rfcl_cfg_entry_point": f"{_AGENTS}:LiberoAllPpoRfclRunnerCfg",
        },
    )


# ---------------------------------------------------------------------------
# Train: all suites (40 tasks)
# ---------------------------------------------------------------------------

_register_dgpo("Isaac-Hebero-All-State-v0", "LiberoAllDgpoOscEnvCfg")

# ---------------------------------------------------------------------------
# Train: single suite (10 tasks)
# ---------------------------------------------------------------------------
# The cfg has existed on libero_dgpo_env_cfg for programmatic use; it needs a gym
# id to be reachable from train.py --task.
_register_dgpo("Isaac-Hebero-Long-State-v0", "LiberoLongDgpoOscEnvCfg")

# ---------------------------------------------------------------------------
# Play / eval: all suites + suite subsets
# ---------------------------------------------------------------------------

_register_dgpo("Isaac-Hebero-All-State-Play-v0", "LiberoAllDgpoOscEnvCfg_PLAY")

_register_dgpo("Isaac-Hebero-Long-State-Play-v0", "LiberoLongDgpoOscEnvCfg_PLAY")
_register_dgpo("Isaac-Hebero-Object-State-Play-v0", "LiberoObjectDgpoOscEnvCfg_PLAY")
_register_dgpo("Isaac-Hebero-Spatial-State-Play-v0", "LiberoSpatialDgpoOscEnvCfg_PLAY")
_register_dgpo("Isaac-Hebero-Goal-State-Play-v0", "LiberoGoalDgpoOscEnvCfg_PLAY")

_register_dgpo("Isaac-Hebero-Spatial-Goal-State-Play-v0", "LiberoSpatialGoalDgpoOscEnvCfg_PLAY")
_register_dgpo("Isaac-Hebero-Object-Long-State-Play-v0", "LiberoObjectLongDgpoOscEnvCfg_PLAY")


# ---------------------------------------------------------------------------
# Vision (frozen Theia + attention pool)
# ---------------------------------------------------------------------------
# Actor sees camera tokens instead of ground-truth object poses; the critic keeps the
# privileged group. Requires ``--enable_cameras`` on train.py / play.py.
_VISION_CFG = f"{__name__}.libero_dgpo_vision_env_cfg"


def _register_vision(gym_id: str, env_cfg_attr: str) -> None:
    """Register a vision DGPO OSC env with the PPO-based runner configurations."""
    gym.register(
        id=gym_id,
        entry_point=_DGPO_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_VISION_CFG}:{env_cfg_attr}",
            "rsl_rl_cfg_entry_point": f"{_AGENTS}:LiberoAllVisIwAbcRunnerCfg",
            "rsl_rl_iw_abc_cfg_entry_point": f"{_AGENTS}:LiberoAllVisIwAbcRunnerCfg",
            "rsl_rl_dgpo_cfg_entry_point": f"{_AGENTS}:LiberoAllVisIwAbcRunnerCfg",
            "rsl_rl_mtppo_cfg_entry_point": f"{_AGENTS}:LiberoAllMtPPOVisionRunnerCfg",
            "rsl_rl_dapg_cfg_entry_point": f"{_AGENTS}:LiberoAllMtDapgVisionRunnerCfg",
        },
    )


_register_vision("Isaac-Hebero-All-Vision-v0", "LiberoAllDgpoOscVisionEnvCfg")
_register_vision("Isaac-Hebero-Long-Vision-v0", "LiberoLongDgpoOscVisionEnvCfg")

_register_vision("Isaac-Hebero-All-Vision-Play-v0", "LiberoAllDgpoOscVisionEnvCfg_PLAY")
_register_vision("Isaac-Hebero-Long-Vision-Play-v0", "LiberoLongDgpoOscVisionEnvCfg_PLAY")
