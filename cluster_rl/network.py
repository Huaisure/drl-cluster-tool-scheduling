from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from problem import ClusterProblem, ModuleType, WaferKey

GLOBAL_FEATURE_DIM = 6
WAFER_FEATURE_DIM = 8
MODULE_FEATURE_DIM = 7
ROBOT_FEATURE_DIM = 4

GLOBAL_TYPE = 0
ROBOT_TYPE = 1
WAFER_TYPE = 2
MODULE_TYPE = 3
PAD_TYPE = 4
TYPE_COUNT = 5

CANDIDATE_RELATION = 0
WAFER_LOCATION_RELATION = 1
ROBOT_LOCATION_RELATION = 2
ROBOT_HOLDS_RELATION = 3
CANDIDATE_TIME_RELATION = 4
RELATION_COUNT = 5


@dataclass(frozen=True)
class EncodedObservation:
    """One unpadded entity-state representation."""

    global_features: np.ndarray
    robot_features: np.ndarray
    wafer_features: np.ndarray
    module_features: np.ndarray
    candidate_modules: np.ndarray
    candidate_process_times: np.ndarray
    wafer_locations: np.ndarray
    robot_location: np.ndarray
    robot_holds: np.ndarray
    action_mask: np.ndarray


@dataclass(frozen=True)
class EntityBatch:
    """Padded tensors consumed by :class:`ClusterActorCritic`."""

    global_features: Tensor
    robot_features: Tensor
    wafer_features: Tensor
    module_features: Tensor
    candidate_modules: Tensor
    candidate_process_times: Tensor
    wafer_locations: Tensor
    robot_location: Tensor
    robot_holds: Tensor
    wafer_valid: Tensor
    module_valid: Tensor
    action_mask: Tensor

    def to(self, device: torch.device | str) -> EntityBatch:
        return EntityBatch(**{field: value.to(device) for field, value in self.__dict__.items()})

    def to_model_actions(self, env_actions: Tensor) -> Tensor:
        """Map each environment action into the padded logits layout."""

        if env_actions.shape != (self.wafer_features.shape[0],):
            raise ValueError("env_actions must contain one action per batch item")

        env_actions = env_actions.to(self.action_mask.device)
        wafer_counts = self.wafer_valid.sum(dim=1)
        module_counts = self.module_valid.sum(dim=1)
        is_place = env_actions >= wafer_counts
        place_index = env_actions - wafer_counts
        if torch.any(env_actions < 0) or torch.any(is_place & (place_index >= module_counts)):
            raise ValueError("env_actions contains an out-of-range action")

        return torch.where(
            is_place,
            self.wafer_features.shape[1] + place_index,
            env_actions,
        )

    def to_env_actions(self, model_actions: Tensor) -> Tensor:
        """Map actions sampled from padded logits back to each environment."""

        if model_actions.shape != (self.wafer_features.shape[0],):
            raise ValueError("model_actions must contain one action per batch item")

        model_actions = model_actions.to(self.action_mask.device)
        max_wafers = self.wafer_features.shape[1]
        wafer_counts = self.wafer_valid.sum(dim=1)
        module_counts = self.module_valid.sum(dim=1)
        is_place = model_actions >= max_wafers
        place_index = model_actions - max_wafers
        invalid_pick = ~is_place & (model_actions >= wafer_counts)
        invalid_place = is_place & (place_index >= module_counts)
        if torch.any(model_actions < 0) or torch.any(invalid_pick) or torch.any(invalid_place):
            raise ValueError("model_actions contains a padded or invalid action")

        return torch.where(
            is_place,
            wafer_counts + place_index,
            model_actions,
        )


@dataclass(frozen=True)
class PolicyValueOutput:
    logits: Tensor
    value: Tensor


@dataclass(frozen=True)
class TransformerConfig:
    model_dim: int = 128
    num_heads: int = 8
    num_layers: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.num_heads <= 0 or self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.num_layers <= 0 or self.feedforward_dim <= 0:
            raise ValueError("num_layers and feedforward_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class ClusterObservationEncoder:
    """Build entity features and relations for one problem instance.

    The ordering must match the environment's ``wafer_keys`` and
    ``module_ids`` so actor logits retain the environment action ordering.
    """

    def __init__(
        self,
        problem: ClusterProblem,
        wafer_keys: Sequence[WaferKey],
        module_ids: Sequence[str],
        time_scale: float = 1.0,
    ) -> None:
        if not np.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        self.problem = problem
        self.wafer_keys = tuple(wafer_keys)
        self.module_ids = tuple(module_ids)
        self.time_scale = float(time_scale)
        self._module_index = {module_id: index for index, module_id in enumerate(self.module_ids)}
        self._lp_id = next(module_id for module_id in self.module_ids if problem.Modules[module_id].type is ModuleType.LP)

        snapshot_keys = set(problem.initial_state.to_snapshot().wafers_by_key)
        if set(self.wafer_keys) != snapshot_keys:
            raise ValueError("wafer_keys must match the problem wafers")
        if set(self.module_ids) != set(problem.Modules):
            raise ValueError("module_ids must match the problem modules")

    @classmethod
    def from_env(
        cls,
        env: Any,
        time_scale: float = 1.0,
    ) -> ClusterObservationEncoder:
        return cls(
            env.problem,
            env.wafer_keys,
            env.module_ids,
            time_scale,
        )

    def encode(
        self,
        observation: Mapping[str, Any],
    ) -> EncodedObservation:
        wafer_count = len(self.wafer_keys)
        module_count = len(self.module_ids)
        wafer_module = np.asarray(observation["wafer_module"], dtype=np.int64)
        wafer_step = np.asarray(observation["wafer_step"], dtype=np.int64)
        process_remaining = np.asarray(observation["process_remaining"], dtype=np.float32)
        action_mask = np.asarray(observation["action_mask"], dtype=np.bool_)
        robot_module = int(observation["robot_module"])

        if wafer_module.shape != (wafer_count,) or wafer_step.shape != (wafer_count,) or process_remaining.shape != (wafer_count,):
            raise ValueError("observation wafer arrays have an invalid shape")
        if action_mask.shape != (wafer_count + module_count,):
            raise ValueError("observation action_mask has an invalid shape")
        if np.any((wafer_module < 0) | (wafer_module > module_count)):
            raise ValueError("wafer_module contains an invalid module index")
        if not 0 <= robot_module <= module_count:
            raise ValueError("robot_module contains an invalid module index")

        wafer_features = np.zeros((wafer_count, WAFER_FEATURE_DIM), dtype=np.float32)
        candidate_modules = np.zeros((wafer_count, module_count), dtype=np.bool_)
        candidate_process_times = np.zeros(
            (wafer_count, module_count),
            dtype=np.float32,
        )
        wafer_locations = np.zeros_like(candidate_modules)
        robot_holds = wafer_module == module_count
        total_process_work = 0.0
        total_visits = 0

        for wafer_index, (route_id, _) in enumerate(self.wafer_keys):
            route = self.problem.routes[route_id]
            completed_step = len(route.visits) + 1
            route_process_times = [
                visit.process_time or 0.0 for visit in route.visits
            ]
            total_process_work += sum(route_process_times)
            total_visits += len(route.visits)
            step = int(wafer_step[wafer_index])
            if not 0 <= step <= completed_step:
                raise ValueError(f"wafer_step[{wafer_index}] is outside its route")

            remaining = float(process_remaining[wafer_index])
            if not np.isfinite(remaining) or remaining < 0:
                raise ValueError("process_remaining must contain finite non-negative values")

            remaining_steps = max(0, completed_step - step)
            next_process_time = (
                route_process_times[step]
                if step < len(route_process_times)
                else 0.0
            )
            wafer_features[wafer_index] = (
                step / completed_step,
                remaining / self.time_scale,
                float(remaining == 0.0),
                float(robot_holds[wafer_index]),
                float(step == completed_step),
                remaining_steps / completed_step,
                next_process_time / self.time_scale,
                sum(route_process_times[step:]) / self.time_scale,
            )
            if wafer_module[wafer_index] < module_count:
                wafer_locations[wafer_index, wafer_module[wafer_index]] = True

            next_step = step + 1
            if next_step <= len(route.visits):
                targets = route.visits[next_step - 1].module_ids
            elif next_step == completed_step:
                targets = (self._lp_id,)
            else:
                targets = ()
            for module_id in targets:
                module_index = self._module_index[module_id]
                candidate_modules[wafer_index, module_index] = True
                candidate_process_times[
                    wafer_index, module_index
                ] = next_process_time / self.time_scale

        occupancy = wafer_locations.sum(axis=0)
        module_features = np.zeros((module_count, MODULE_FEATURE_DIM), dtype=np.float32)
        for module_index, module_id in enumerate(self.module_ids):
            module = self.problem.Modules[module_id]
            type_index = {
                ModuleType.LP: 0,
                ModuleType.PM: 1,
                ModuleType.LL: 2,
            }[module.type]
            capacity = module.capacity
            available = capacity - occupancy[module_index]
            module_features[module_index, type_index] = 1.0
            module_features[module_index, 3:] = (
                np.log1p(capacity),
                occupancy[module_index] / capacity,
                available / capacity,
                float(available == 0),
            )

        robot_location = np.zeros(module_count, dtype=np.bool_)
        if robot_module < module_count:
            robot_location[robot_module] = True
        robot = next(iter(self.problem.ClusterTool.values()))

        return EncodedObservation(
            global_features=np.asarray(
                [
                    np.log1p(wafer_count),
                    np.log1p(module_count),
                    np.log1p(len(self.problem.routes)),
                    np.log1p(self.time_scale),
                    total_process_work / self.time_scale,
                    total_visits / wafer_count,
                ],
                dtype=np.float32,
            ),
            robot_features=np.asarray(
                [
                    robot_holds.any(),
                    robot.pick_time / self.time_scale,
                    robot.place_time / self.time_scale,
                    float(robot.travel_times) / self.time_scale,
                ],
                dtype=np.float32,
            ),
            wafer_features=wafer_features,
            module_features=module_features,
            candidate_modules=candidate_modules,
            candidate_process_times=candidate_process_times,
            wafer_locations=wafer_locations,
            robot_location=robot_location,
            robot_holds=robot_holds,
            action_mask=action_mask,
        )


def collate_observations(
    encoders: Sequence[ClusterObservationEncoder],
    observations: Sequence[Mapping[str, Any]],
    *,
    device: torch.device | str | None = None,
) -> EntityBatch:
    """Encode and pad observations from potentially different instances."""

    if not encoders or len(encoders) != len(observations):
        raise ValueError("encoders and observations must have the same non-zero length")

    encoded = [encoder.encode(observation) for encoder, observation in zip(encoders, observations)]
    batch_size = len(encoded)
    max_wafers = max(item.wafer_features.shape[0] for item in encoded)
    max_modules = max(item.module_features.shape[0] for item in encoded)

    global_features = torch.zeros(
        batch_size,
        GLOBAL_FEATURE_DIM,
        dtype=torch.float32,
        device=device,
    )
    robot_features = torch.zeros(batch_size, ROBOT_FEATURE_DIM, dtype=torch.float32, device=device)
    wafer_features = torch.zeros(
        batch_size,
        max_wafers,
        WAFER_FEATURE_DIM,
        dtype=torch.float32,
        device=device,
    )
    module_features = torch.zeros(
        batch_size,
        max_modules,
        MODULE_FEATURE_DIM,
        dtype=torch.float32,
        device=device,
    )
    candidate_modules = torch.zeros(
        batch_size,
        max_wafers,
        max_modules,
        dtype=torch.bool,
        device=device,
    )
    candidate_process_times = torch.zeros(
        batch_size,
        max_wafers,
        max_modules,
        dtype=torch.float32,
        device=device,
    )
    wafer_locations = torch.zeros_like(candidate_modules)
    robot_location = torch.zeros(batch_size, max_modules, dtype=torch.bool, device=device)
    robot_holds = torch.zeros(batch_size, max_wafers, dtype=torch.bool, device=device)
    wafer_valid = torch.zeros_like(robot_holds)
    module_valid = torch.zeros_like(robot_location)
    action_mask = torch.zeros(
        batch_size,
        max_wafers + max_modules,
        dtype=torch.bool,
        device=device,
    )

    for batch_index, item in enumerate(encoded):
        wafer_count = item.wafer_features.shape[0]
        module_count = item.module_features.shape[0]

        global_features[batch_index] = torch.as_tensor(
            item.global_features,
            device=device,
        )
        robot_features[batch_index] = torch.as_tensor(item.robot_features, device=device)
        wafer_features[batch_index, :wafer_count] = torch.as_tensor(item.wafer_features, device=device)
        module_features[batch_index, :module_count] = torch.as_tensor(item.module_features, device=device)
        candidate_modules[batch_index, :wafer_count, :module_count] = torch.as_tensor(item.candidate_modules, device=device)
        candidate_process_times[
            batch_index, :wafer_count, :module_count
        ] = torch.as_tensor(
            item.candidate_process_times,
            device=device,
        )
        wafer_locations[batch_index, :wafer_count, :module_count] = torch.as_tensor(item.wafer_locations, device=device)
        robot_location[batch_index, :module_count] = torch.as_tensor(item.robot_location, device=device)
        robot_holds[batch_index, :wafer_count] = torch.as_tensor(item.robot_holds, device=device)
        wafer_valid[batch_index, :wafer_count] = True
        module_valid[batch_index, :module_count] = True

        action_mask[batch_index, :wafer_count] = torch.as_tensor(item.action_mask[:wafer_count], device=device)
        action_mask[batch_index, max_wafers : max_wafers + module_count] = torch.as_tensor(item.action_mask[wafer_count:], device=device)

    return EntityBatch(
        global_features=global_features,
        robot_features=robot_features,
        wafer_features=wafer_features,
        module_features=module_features,
        candidate_modules=candidate_modules,
        candidate_process_times=candidate_process_times,
        wafer_locations=wafer_locations,
        robot_location=robot_location,
        robot_holds=robot_holds,
        wafer_valid=wafer_valid,
        module_valid=module_valid,
        action_mask=action_mask,
    )


class _FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


class ClusterActorCritic(nn.Module):
    """Relation-aware entity-token Transformer actor-critic.

    Token order is ``[GLOBAL], [ROBOT], [WAFER] * N, [MODULE] * M``.
    Padded wafer and module slots receive the PAD type embedding and are
    excluded as attention keys. No positional embedding is used, preserving
    permutation equivariance within each entity type.
    """

    def __init__(
        self,
        config: TransformerConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TransformerConfig()
        model_dim = self.config.model_dim

        self.global_encoder = _FeatureEncoder(GLOBAL_FEATURE_DIM, model_dim)
        self.robot_encoder = _FeatureEncoder(ROBOT_FEATURE_DIM, model_dim)
        self.wafer_encoder = _FeatureEncoder(WAFER_FEATURE_DIM, model_dim)
        self.module_encoder = _FeatureEncoder(MODULE_FEATURE_DIM, model_dim)
        self.type_embedding = nn.Embedding(TYPE_COUNT, model_dim)
        self.global_token = nn.Parameter(torch.empty(model_dim))
        self.pad_token = nn.Parameter(torch.empty(model_dim))
        self.relation_bias = nn.Parameter(torch.zeros(RELATION_COUNT, self.config.num_heads))

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=self.config.num_heads,
                    dim_feedforward=self.config.feedforward_dim,
                    dropout=self.config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.pick_head = nn.Linear(model_dim, 1)
        self.place_head = nn.Linear(model_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.global_token, std=0.02)
        nn.init.normal_(self.pad_token, std=0.02)
        nn.init.normal_(self.type_embedding.weight, std=0.02)
        nn.init.normal_(self.relation_bias, std=0.02)

    def forward(self, batch: EntityBatch) -> PolicyValueOutput:
        self._validate_batch(batch)
        tokens, token_valid = self._tokens(batch)
        attention_bias = self._attention_bias(batch, token_valid, tokens.dtype)

        for layer in self.layers:
            tokens = layer(tokens, src_mask=attention_bias)
        tokens = self.final_norm(tokens)

        wafer_count = batch.wafer_features.shape[1]
        module_count = batch.module_features.shape[1]
        wafer_states = tokens[:, 2 : 2 + wafer_count]
        module_states = tokens[:, 2 + wafer_count : 2 + wafer_count + module_count]
        logits = torch.cat(
            (
                self.pick_head(wafer_states).squeeze(-1),
                self.place_head(module_states).squeeze(-1),
            ),
            dim=-1,
        )
        logits = logits.masked_fill(~batch.action_mask, -torch.inf)
        value = self.value_head(tokens[:, 0]).squeeze(-1)
        return PolicyValueOutput(logits=logits, value=value)

    def _tokens(self, batch: EntityBatch) -> tuple[Tensor, Tensor]:
        batch_size, wafer_count = batch.wafer_valid.shape
        module_count = batch.module_valid.shape[1]
        device = batch.wafer_features.device

        global_token = (
            self.global_encoder(batch.global_features)
            + self.global_token
        ).unsqueeze(1)
        robot_token = self.robot_encoder(batch.robot_features).unsqueeze(1)
        wafer_tokens = self.wafer_encoder(batch.wafer_features)
        module_tokens = self.module_encoder(batch.module_features)

        types = torch.full(
            (batch_size, 2 + wafer_count + module_count),
            PAD_TYPE,
            dtype=torch.long,
            device=device,
        )
        types[:, 0] = GLOBAL_TYPE
        types[:, 1] = ROBOT_TYPE
        types[:, 2 : 2 + wafer_count] = torch.where(
            batch.wafer_valid,
            WAFER_TYPE,
            PAD_TYPE,
        )
        types[:, 2 + wafer_count :] = torch.where(
            batch.module_valid,
            MODULE_TYPE,
            PAD_TYPE,
        )
        token_valid = torch.cat(
            (
                torch.ones(batch_size, 2, dtype=torch.bool, device=device),
                batch.wafer_valid,
                batch.module_valid,
            ),
            dim=1,
        )

        tokens = torch.cat((global_token, robot_token, wafer_tokens, module_tokens), dim=1)
        padding = self.pad_token.view(1, 1, -1)
        tokens = torch.where(token_valid.unsqueeze(-1), tokens, padding)
        return tokens + self.type_embedding(types), token_valid

    def _attention_bias(
        self,
        batch: EntityBatch,
        token_valid: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        batch_size, wafer_count = batch.wafer_valid.shape
        module_count = batch.module_valid.shape[1]
        token_count = token_valid.shape[1]
        wafer_slice = slice(2, 2 + wafer_count)
        module_slice = slice(2 + wafer_count, token_count)

        relations = torch.zeros(
            batch_size,
            RELATION_COUNT,
            token_count,
            token_count,
            dtype=dtype,
            device=token_valid.device,
        )
        candidate = batch.candidate_modules.to(dtype)
        candidate_process_times = batch.candidate_process_times.to(dtype)
        wafer_locations = batch.wafer_locations.to(dtype)
        robot_location = batch.robot_location.to(dtype)
        robot_holds = batch.robot_holds.to(dtype)

        relations[:, CANDIDATE_RELATION, wafer_slice, module_slice] = candidate
        relations[:, CANDIDATE_RELATION, module_slice, wafer_slice] = candidate.transpose(1, 2)
        relations[:, WAFER_LOCATION_RELATION, wafer_slice, module_slice] = wafer_locations
        relations[:, WAFER_LOCATION_RELATION, module_slice, wafer_slice] = wafer_locations.transpose(1, 2)
        relations[:, ROBOT_LOCATION_RELATION, 1, module_slice] = robot_location
        relations[:, ROBOT_LOCATION_RELATION, module_slice, 1] = robot_location
        relations[:, ROBOT_HOLDS_RELATION, 1, wafer_slice] = robot_holds
        relations[:, ROBOT_HOLDS_RELATION, wafer_slice, 1] = robot_holds
        relations[
            :, CANDIDATE_TIME_RELATION, wafer_slice, module_slice
        ] = candidate_process_times
        relations[
            :, CANDIDATE_TIME_RELATION, module_slice, wafer_slice
        ] = candidate_process_times.transpose(1, 2)

        attention_bias = torch.einsum(
            "brij,rh->bhij",
            relations,
            self.relation_bias.to(dtype),
        )
        attention_bias = attention_bias.masked_fill(
            ~token_valid[:, None, None, :],
            -torch.inf,
        )
        return attention_bias.flatten(0, 1)

    @staticmethod
    def _validate_batch(batch: EntityBatch) -> None:
        batch_size, wafer_count, wafer_dim = batch.wafer_features.shape
        if wafer_dim != WAFER_FEATURE_DIM:
            raise ValueError("wafer_features has an invalid feature dimension")
        if batch.global_features.shape != (
            batch_size,
            GLOBAL_FEATURE_DIM,
        ):
            raise ValueError("global_features has an invalid shape")
        if batch.robot_features.shape != (
            batch_size,
            ROBOT_FEATURE_DIM,
        ):
            raise ValueError("robot_features has an invalid shape")
        if batch.module_features.ndim != 3:
            raise ValueError("module_features must be a rank-3 tensor")
        module_count = batch.module_features.shape[1]
        if batch.module_features.shape != (
            batch_size,
            module_count,
            MODULE_FEATURE_DIM,
        ):
            raise ValueError("module_features has an invalid feature dimension")

        expected_shapes = {
            "candidate_modules": (batch_size, wafer_count, module_count),
            "candidate_process_times": (
                batch_size,
                wafer_count,
                module_count,
            ),
            "wafer_locations": (batch_size, wafer_count, module_count),
            "robot_location": (batch_size, module_count),
            "robot_holds": (batch_size, wafer_count),
            "wafer_valid": (batch_size, wafer_count),
            "module_valid": (batch_size, module_count),
            "action_mask": (
                batch_size,
                wafer_count + module_count,
            ),
        }
        for name, shape in expected_shapes.items():
            if getattr(batch, name).shape != shape:
                raise ValueError(f"{name} has an invalid shape")
