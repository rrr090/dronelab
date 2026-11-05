"""Actor-critic MLP policy.

Kept small on purpose (two hidden layers of 64 units) so that the
activations can be inspected neuron-by-neuron in the mech interp stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dronelab.hover.env import ACT_DIM, OBS_DIM


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 64, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 obs_norm: bool = False, init_log_std: float = -0.5):
        super().__init__()
        self.config = dict(hidden=hidden, obs_dim=obs_dim, act_dim=act_dim,
                           obs_norm=obs_norm, init_log_std=init_log_std)
        self.obs_norm = obs_norm
        if obs_norm:
            # Running observation statistics, saved with the weights.
            self.register_buffer("obs_mean", torch.zeros(obs_dim))
            self.register_buffer("obs_var", torch.ones(obs_dim))
            self.register_buffer("obs_count", torch.tensor(1e-4))
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        self.critic = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    @torch.no_grad()
    def update_obs_stats(self, batch: torch.Tensor):
        """Merge a batch of raw observations into the running mean/var."""
        n = batch.shape[0]
        mean, var = batch.mean(0), batch.var(0, unbiased=False)
        tot = self.obs_count + n
        delta = mean - self.obs_mean
        self.obs_mean += delta * n / tot
        self.obs_var.copy_((self.obs_var * self.obs_count + var * n
                            + delta ** 2 * self.obs_count * n / tot) / tot)
        self.obs_count.copy_(tot)

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.obs_norm:
            return obs
        return ((obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)).clamp(-10, 10)

    def forward(self, obs: torch.Tensor):
        h = self.body(self.normalize(obs))
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
