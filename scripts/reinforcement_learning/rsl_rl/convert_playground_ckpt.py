# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a legacy RSL-RL checkpoint to rsl-rl >= 4/5 format.

Legacy checkpoints (rsl-rl 2.x ``OnPolicyRunner``, optionally patched by
``patch_rsl_rl_ppo_with_importance_weights``) store one combined ActorCritic::

    model_state_dict:
        log_std | std                       -> distribution.{log_std_param|std_param}
        actor.N.{weight,bias}               -> actor_state_dict:  mlp.N.*
        actor_obs_normalizer.{_mean,...}    -> actor_state_dict:  obs_normalizer.*
        critic.N.{weight,bias}              -> critic_state_dict: mlp.N.*
        critic_obs_normalizer.{_mean,...}   -> critic_state_dict: obs_normalizer.*

rsl-rl >= 4/5 expects separate ``actor_state_dict`` / ``critic_state_dict``
(``MLPModel`` with an embedded ``obs_normalizer`` and a ``distribution``).

The optimizer state is NOT converted (param layouts differ) — the converted file
is for evaluation / fine-tune initialisation. Load it with a ``load_cfg`` that
skips the optimizer (``eval_libero.py`` does this automatically and also
auto-converts legacy files in memory, so running this CLI is only needed to use
a legacy checkpoint with the stock ``play.py`` / ``--resume``-less ``train.py``).

Usage::

    python convert_playground_ckpt.py --input <old_model.pt> [--output <new_model.pt>]
"""

from __future__ import annotations

import argparse
import os

import torch

_NORM_KEYS = ("_mean", "_var", "_std", "count")


def is_legacy_playground_checkpoint(loaded: dict) -> bool:
    """True when the dict is a legacy combined-ActorCritic checkpoint."""
    return "actor_state_dict" not in loaded and "model_state_dict" in loaded


def convert_playground_checkpoint(loaded: dict) -> dict:
    """Convert a legacy playground checkpoint dict to rsl-rl >= 4/5 layout."""
    msd = loaded["model_state_dict"]
    actor_sd: dict[str, torch.Tensor] = {}
    critic_sd: dict[str, torch.Tensor] = {}
    for key, value in msd.items():
        if key == "log_std":
            actor_sd["distribution.log_std_param"] = value
        elif key == "std":
            actor_sd["distribution.std_param"] = value
        elif key.startswith("actor_obs_normalizer."):
            actor_sd["obs_normalizer." + key.split(".", 1)[1]] = value
        elif key.startswith("critic_obs_normalizer."):
            critic_sd["obs_normalizer." + key.split(".", 1)[1]] = value
        elif key.startswith("actor."):
            actor_sd["mlp." + key.split(".", 1)[1]] = value
        elif key.startswith("critic."):
            critic_sd["mlp." + key.split(".", 1)[1]] = value
        else:
            print(f"[convert] skipping unrecognised model key: {key}")

    # Legacy runners without empirical normalization have no normalizer entries;
    # synthesise identity stats so strict loading into a normalizing model works.
    def _ensure_normalizer(sd: dict[str, torch.Tensor], in_dim: int | None):
        if any(k.startswith("obs_normalizer.") for k in sd) or in_dim is None:
            return
        sd["obs_normalizer._mean"] = torch.zeros(1, in_dim)
        sd["obs_normalizer._var"] = torch.ones(1, in_dim)
        sd["obs_normalizer._std"] = torch.ones(1, in_dim)
        sd["obs_normalizer.count"] = torch.tensor(1e-8)

    actor_in = actor_sd["mlp.0.weight"].shape[1] if "mlp.0.weight" in actor_sd else None
    critic_in = critic_sd["mlp.0.weight"].shape[1] if "mlp.0.weight" in critic_sd else None
    _ensure_normalizer(actor_sd, actor_in)
    _ensure_normalizer(critic_sd, critic_in)

    return {
        "actor_state_dict": actor_sd,
        "critic_state_dict": critic_sd,
        "iter": loaded.get("iter", 0),
        "infos": loaded.get("infos"),
        # provenance for debugging
        "converted_from": "playground_model_state_dict",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Legacy playground checkpoint (.pt).")
    parser.add_argument("--output", default=None, help="Output path (default: <input>_rslrl5.pt).")
    args = parser.parse_args()

    loaded = torch.load(args.input, map_location="cpu", weights_only=False)
    if not is_legacy_playground_checkpoint(loaded):
        raise SystemExit(f"'{args.input}' already has actor_state_dict / is not a legacy checkpoint.")
    converted = convert_playground_checkpoint(loaded)
    output = args.output or os.path.splitext(args.input)[0] + "_rslrl5.pt"
    torch.save(converted, output)
    actor_keys = len(converted["actor_state_dict"])
    critic_keys = len(converted["critic_state_dict"])
    print(f"[convert] wrote {output} (actor keys: {actor_keys}, critic keys: {critic_keys}, iter: {converted['iter']})")


if __name__ == "__main__":
    main()
