"""Actor-critic MLP policy.

Kept small on purpose (two hidden layers of 64 units) so that the
activations can be inspected neuron-by-neuron in the mech interp stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from drone_mi.env import ACT_DIM, OBS_DIM


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, ACT_DIM)
        self.actor_log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))
        self.critic = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mean = self.actor_mean(h)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def act(self, obs: torch.Tensor):
        mean, std, value = self(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        mean, std, value = self(obs)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value
