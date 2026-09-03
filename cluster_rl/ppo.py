"""Environment-independent PPO math shared by the legacy and IR trainers."""

from __future__ import annotations

import torch
from torch import Tensor


def advantages(rewards: Tensor, dones: Tensor, values: Tensor, last_values: Tensor,
               gamma: float, gae_lambda: float) -> Tensor:
    result = torch.zeros_like(rewards)
    advantage = torch.zeros_like(last_values)
    for step in reversed(range(rewards.shape[0])):
        next_values = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        next_non_terminal = ~dones[step]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
        result[step] = advantage
    return result


def normalize_choice_advantages(advantages: Tensor, choice_mask: Tensor) -> Tensor:
    normalized = torch.zeros_like(advantages)
    if choice_mask.any():
        chosen = advantages[choice_mask]
        normalized[choice_mask] = (chosen - chosen.mean()) / (chosen.std(unbiased=False) + 1e-8)
    return normalized


def clipped_losses(log_ratio: Tensor, values: Tensor, old_values: Tensor, returns: Tensor,
                   advantages: Tensor, entropies: Tensor, has_choice: Tensor,
                   clip_coefficient: float) -> tuple[Tensor, Tensor, Tensor]:
    """Clipped actor/value losses; forced choices train the critic, not actor."""
    if has_choice.any():
        ratio = log_ratio[has_choice].exp()
        chosen = advantages[has_choice]
        policy = torch.maximum(-chosen * ratio,
                               -chosen * ratio.clamp(1 - clip_coefficient, 1 + clip_coefficient)).mean()
        entropy = entropies[has_choice].mean()
    else:
        policy = values.sum() * 0
        entropy = values.sum() * 0
    clipped = old_values + (values - old_values).clamp(-clip_coefficient, clip_coefficient)
    value = 0.5 * torch.maximum((values - returns).square(), (clipped - returns).square()).mean()
    return policy, value, entropy
