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
                 obs_norm: bool = False, init_log_std: float = -0.5, priv_dim: int = 0):
        """``priv_dim > 0``: the critic gets its own network, which also reads that many
        privileged inputs. The actor never sees them, so the policy can fly without them."""
        super().__init__()
        self.config = dict(hidden=hidden, obs_dim=obs_dim, act_dim=act_dim,
                           obs_norm=obs_norm, init_log_std=init_log_std, priv_dim=priv_dim)
        self.obs_norm = obs_norm
        self.priv_dim = 0
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
        if priv_dim:
            self._add_critic_network(priv_dim)
            for layer in (self.critic_body[0], self.critic_body[2]):
                nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
                nn.init.zeros_(layer.bias)

    def _add_critic_network(self, priv_dim: int):
        hidden, obs_dim = self.config["hidden"], self.config["obs_dim"]
        self.priv_dim = self.config["priv_dim"] = priv_dim
        self.register_buffer("priv_mean", torch.zeros(priv_dim))
        self.register_buffer("priv_var", torch.ones(priv_dim))
        self.register_buffer("priv_count", torch.tensor(1e-4))
        self.critic_body = nn.Sequential(
            nn.Linear(obs_dim + priv_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    @torch.no_grad()
    def split_critic(self, priv_dim: int):
        """Turn a shared-body network into one with a separate critic, in place.

        The actor is untouched. The critic starts as a copy of the shared body with
        zero weights on the privileged inputs, so actions and values are both exactly
        what they were until training changes them.
        """
        if self.priv_dim:
            raise ValueError("this network already has a separate critic")
        self._add_critic_network(priv_dim)
        first, second = self.critic_body[0], self.critic_body[2]
        first.weight.zero_()
        first.weight[:, :self.config["obs_dim"]] = self.body[0].weight
        first.bias.copy_(self.body[0].bias)
        second.load_state_dict(self.body[2].state_dict())

    def grad_groups(self) -> list[list[nn.Parameter]]:
        """Parameters clipped together: a separate critic must not shrink the actor's step."""
        if not self.priv_dim:
            return [list(self.parameters())]
        critic = list(self.critic_body.parameters()) + list(self.critic.parameters())
        ids = {id(q) for q in critic}
        return [[q for q in self.parameters() if id(q) not in ids], critic]

    @staticmethod
    def _merge_stats(batch: torch.Tensor, running_mean, running_var, count):
        n = batch.shape[0]
        mean, var = batch.mean(0), batch.var(0, unbiased=False)
        tot = count + n
        delta = mean - running_mean
        running_mean += delta * n / tot
        running_var.copy_((running_var * count + var * n + delta ** 2 * count * n / tot) / tot)
        count.copy_(tot)

    @torch.no_grad()
    def update_obs_stats(self, batch: torch.Tensor, priv: torch.Tensor | None = None):
        """Merge a batch of raw observations (and privileged inputs) into the running mean/var."""
        self._merge_stats(batch, self.obs_mean, self.obs_var, self.obs_count)
        if priv is not None:
            self._merge_stats(priv, self.priv_mean, self.priv_var, self.priv_count)

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.obs_norm:
            return obs
        return ((obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)).clamp(-10, 10)

    def forward(self, obs: torch.Tensor, priv: torch.Tensor | None = None):
        """With a separate critic, ``value`` is None unless ``priv`` is given (flying needs no value)."""
        x = self.normalize(obs)
        h = self.body(x)
        mean = self.actor_mean(h)
        std = self.actor_log_std.exp().expand_as(mean)
        if not self.priv_dim:
            return mean, std, self.critic(h).squeeze(-1)
        if priv is None:
            return mean, std, None
        p = ((priv - self.priv_mean) / torch.sqrt(self.priv_var + 1e-8)).clamp(-10, 10)
        return mean, std, self.critic(self.critic_body(torch.cat([x, p], -1))).squeeze(-1)

    def act(self, obs: torch.Tensor, priv: torch.Tensor | None = None):
        mean, std, value = self(obs, priv)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor, priv: torch.Tensor | None = None):
        mean, std, value = self(obs, priv)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value
