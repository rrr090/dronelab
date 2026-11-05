"""Train a hover policy on PlanarQuadrotorEnv with PPO.

Usage:
    python train.py --updates 300 --out checkpoints/hover_mlp.pt
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from drone_mi.env import PlanarQuadrotorEnv
from drone_mi.policy import ActorCritic


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rewards)
    last_adv = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        adv[t] = last_adv
    returns = adv + values
    return adv, returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--steps", type=int, default=2048, help="env steps per update")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints/hover_mlp.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    env = PlanarQuadrotorEnv(seed=args.seed)
    model = ActorCritic().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    obs = env.reset()
    ep_return, ep_returns = 0.0, []

    for update in range(1, args.updates + 1):
        buf_obs = np.zeros((args.steps, 6), dtype=np.float32)
        buf_act = np.zeros((args.steps, 2), dtype=np.float32)
        buf_logp = np.zeros(args.steps, dtype=np.float32)
        buf_rew = np.zeros(args.steps, dtype=np.float32)
        buf_val = np.zeros(args.steps, dtype=np.float32)
        buf_done = np.zeros(args.steps, dtype=np.float32)

        for t in range(args.steps):
            buf_obs[t] = obs
            with torch.no_grad():
                a, logp, v = model.act(torch.as_tensor(obs, device=device).unsqueeze(0))
            a_np = a.squeeze(0).cpu().numpy()
            buf_act[t] = a_np
            buf_logp[t] = logp.item()
            buf_val[t] = v.item()

            obs, r, term, trunc, _ = env.step(a_np)
            buf_rew[t] = r
            ep_return += r
            done = term or trunc
            buf_done[t] = float(done)
            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                obs = env.reset()

        with torch.no_grad():
            last_v = model.critic(model.body(torch.as_tensor(obs, device=device).unsqueeze(0))).item()
        adv, ret = compute_gae(buf_rew, buf_val, buf_done, last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T = torch.as_tensor
        b_obs = T(buf_obs, device=device)
        b_act = T(buf_act, device=device)
        b_logp = T(buf_logp, device=device)
        b_adv = T(adv, device=device)
        b_ret = T(ret, device=device)

        idx = np.arange(args.steps)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, args.steps, args.minibatch):
                mb = torch.as_tensor(idx[s:s + args.minibatch], device=device)
                logp, ent, value = model.evaluate(b_obs[mb], b_act[mb])
                ratio = (logp - b_logp[mb]).exp()
                pg1 = -b_adv[mb] * ratio
                pg2 = -b_adv[mb] * ratio.clamp(0.8, 1.2)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * (value - b_ret[mb]).pow(2).mean()
                loss = pg_loss + 0.5 * v_loss - 0.001 * ent.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        if update % 10 == 0 or update == 1:
            recent = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            print(f"update {update:4d} | mean return (last 20 eps) {recent:8.2f} "
                  f"| episodes {len(ep_returns)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
