# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL neural models customized for Isaac Lab."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel as _CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict


class CNNModel(_CNNModel):
    """CNN model that supports pure image-only observations.

    The rsl_rl CNN model does not support image-only observations as it calls
    :meth:`get_latent` without checking whether the observation groups are empty.
    """

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        latent_cnn = torch.cat([self.cnns[group](obs[group]) for group in self.obs_groups_2d], dim=-1)
        if not self.obs_groups:
            return latent_cnn
        latent_1d = MLPModel.get_latent(self, obs, masks, hidden_state)
        return torch.cat([latent_1d, latent_cnn], dim=-1)


class AttentionPool(nn.Module):
    """Single-query attention pooling over a token sequence.

    Takes ``(B, P, D)`` and returns ``(B, D)``. A learnable query scores each of the
    ``P`` tokens; tokens are softmax-weighted and summed. Mean-pooling a ViT's patch
    tokens throws away which patch mattered -- the gripper and the target object are a
    handful of patches out of dozens -- so the weights are learned end-to-end with the
    policy loss instead.
    """

    def __init__(self, dim: int) -> None:
        """Create a pool over ``dim``-wide tokens."""
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale = dim**-0.5

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool ``(B, P, D)`` tokens into ``(B, D)``."""
        scores = (tokens * self.query).sum(dim=-1) * self.scale  # (B, P)
        weights = scores.softmax(dim=-1).unsqueeze(-1)  # (B, P, 1)
        return (weights * tokens).sum(dim=1)  # (B, D)


class TokenPoolModel(MLPModel):
    """MLP model that consumes 3D token-sequence observation groups via attention pooling.

    Stands to a frozen ViT (Theia) as :class:`CNNModel` stands to raw pixels: 3D groups
    ``(B, P, D)`` are pooled by a per-group :class:`AttentionPool` and the pooled vectors
    are concatenated with the normalized 1D groups before the MLP head. The encoder
    itself stays in the environment (``mdp.image_features``) and stays frozen -- only the
    pooling query and the MLP train.

    Unlike ``CNNModel``'s ``share_cnn_encoders``, actor and critic get their own pools.
    They are already separate networks under rsl-rl >= 4.0 and the pool is a single
    ``D``-wide query vector, so sharing would save ~400 parameters at the cost of coupling
    two models that no other layer couples.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        pools: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        """Initialize the token-pooling model.

        Args:
            obs: Observation dictionary.
            obs_groups: Mapping from observation sets to lists of observation groups.
            obs_set: Observation set to use for this model ("actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the 1D observations before the MLP.
                Pooled tokens are not normalized -- they leave the ViT after ImageNet
                preprocessing and are already roughly zero-mean unit-variance.
            distribution_cfg: Configuration for the output distribution.
            pools: Pooling modules to reuse, e.g. to share them with another model. If
                None, new pools are created.
        """
        # Populates self.obs_groups_3d / self.obs_token_dims for the pool construction
        # below; the parent __init__ calls it again for the 1D groups.
        self._get_obs_dim(obs, obs_groups, obs_set)

        if pools is not None:
            if set(pools.keys()) != set(self.obs_groups_3d):
                raise ValueError("The 3D observations must be identical for all models sharing token pools.")
        else:
            pools = {group: AttentionPool(dim) for group, dim in zip(self.obs_groups_3d, self.obs_token_dims)}

        self.pool_latent_dim = sum(self.obs_token_dims)

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        self.pools = pools if isinstance(pools, nn.ModuleDict) else nn.ModuleDict(pools)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent from normalized 1D groups and attention-pooled 3D groups."""
        latent_pool = torch.cat([self.pools[group](obs[group]) for group in self.obs_groups_3d], dim=-1)
        if not self.obs_groups:
            return latent_pool
        latent_1d = MLPModel.get_latent(self, obs, masks, hidden_state)
        return torch.cat([latent_1d, latent_pool], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update the 1D observation normalizer (no-op when the model is vision-only)."""
        if self.obs_groups:
            super().update_normalization(obs)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchTokenPoolModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxTokenPoolModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Split the active groups into 1D and 3D and record the 3D token dimensions."""
        obs_dim_1d = 0
        obs_groups_1d: list[str] = []
        obs_groups_3d: list[str] = []
        obs_num_tokens: list[int] = []
        obs_token_dims: list[int] = []

        for obs_group in obs_groups[obs_set]:
            shape = obs[obs_group].shape
            if len(shape) == 3:  # B, P, D
                obs_groups_3d.append(obs_group)
                obs_num_tokens.append(shape[-2])
                obs_token_dims.append(shape[-1])
            elif len(shape) == 2:  # B, D
                obs_groups_1d.append(obs_group)
                obs_dim_1d += shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {tuple(shape)}")

        if not obs_groups_3d:
            raise ValueError("No 3D observations are provided. If this is intentional, use the MLP model instead.")

        self.obs_groups_3d = obs_groups_3d
        self.obs_num_tokens = obs_num_tokens
        self.obs_token_dims = obs_token_dims
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.pool_latent_dim


class _TorchTokenPoolModel(nn.Module):
    """Exportable token-pooling model for JIT."""

    def __init__(self, model: TokenPoolModel) -> None:
        """Create a TorchScript-friendly copy of a TokenPoolModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # ModuleDict -> ModuleList so the export iterates in obs_groups_3d order.
        self.pools = nn.ModuleList([copy.deepcopy(model.pools[g]) for g in model.obs_groups_3d])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_1d: torch.Tensor, obs_3d: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from separated 1D and token inputs."""
        latent_1d = self.obs_normalizer(obs_1d)
        latent_pool_list = []
        for i, pool in enumerate(self.pools):  # obs_3d is assumed to follow obs_groups_3d order
            latent_pool_list.append(pool(obs_3d[i]))
        latent = torch.cat([latent_1d, torch.cat(latent_pool_list, dim=-1)], dim=-1)
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for token-pool exports)."""
        pass


class _OnnxTokenPoolModel(nn.Module):
    """Exportable token-pooling model for ONNX."""

    def __init__(self, model: TokenPoolModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a TokenPoolModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.pools = nn.ModuleList([copy.deepcopy(model.pools[g]) for g in model.obs_groups_3d])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_groups_3d = model.obs_groups_3d
        self.obs_token_dims = model.obs_token_dims
        self.obs_num_tokens = model.obs_num_tokens
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs_1d: torch.Tensor, *obs_3d: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent_1d = self.obs_normalizer(obs_1d)
        latent_pool_list = []
        for i, pool in enumerate(self.pools):
            latent_pool_list.append(pool(obs_3d[i]))
        latent = torch.cat([latent_1d, torch.cat(latent_pool_list, dim=-1)], dim=-1)
        return self.deterministic_output(self.mlp(latent))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_3d = [
            torch.zeros(1, num_tokens, dim)
            for num_tokens, dim in zip(self.obs_num_tokens, self.obs_token_dims)
        ]
        return (dummy_1d, *dummy_3d)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.obs_groups_3d]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
