"""Vectorised PPO shared by the cargo and mission training scripts."""
from __future__ import annotations

import argparse
import os
import time
from typing import Callable

import numpy as np
import torch

from dronelab.policy import ActorCritic


def add_ppo_args(p: argparse.ArgumentParser, **defaults):
    p.add_argument("--updates", type=int, default=defaults.get("updates", 400))
    p.add_argument("--envs", type=int, default=defaults.get("envs", 64))
    p.add_argument("--steps", type=int, default=256, help="steps per env per update")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=defaults.get("lr", 3e-4))
    p.add_argument("--ent-coef", type=float, default=defaults.get("ent_coef", 0.002))
    p.add_argument("--critic-warmup", type=int, default=defaults.get("critic_warmup", 0),
                   help="first N updates train only the value function")
    p.add_argument("--hidden", type=int, default=defaults.get("hidden", 128))
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--curriculum", type=float, default=0.6,
                   help="fraction of training over which difficulty ramps 0 -> 1")
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=defaults.get("eval_every", 25))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument("--out", default=defaults["out"])
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0,
                   help="CPU threads for PyTorch, useful when running seeds side by side (0 = default)")
    return p


def save(model, path, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict(), **(extra or {})}, path)


def train_ppo(args, env, obs_dim: int, act_dim: int,
              eval_fn: Callable[[ActorCritic, torch.device], str] | None = None,
              episode_stat: Callable[[dict, int], dict] | None = None,
              model_init: Callable[[ActorCritic], None] | None = None,
              save_extra: dict | None = None):
    """Train with PPO. ``env`` is a batched env with a ``difficulty`` attribute.

    ``eval_fn(model, device)`` returns a one-line report, run every --eval-every.
    ``episode_stat(info, i)`` extracts extra per-episode stats when env i finishes.
    ``model_init(model)`` can warm-start a fresh model (ignored with --resume).
    ``save_extra`` is stored in every checkpoint (e.g. how to interpret the actions).
    """
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    print(f"device: {device} | envs: {args.envs} | samples/update: {args.envs * args.steps}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model = ActorCritic(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"resumed from {args.resume}")
    else:
        model = ActorCritic(hidden=args.hidden, obs_dim=obs_dim, act_dim=act_dim,
                            obs_norm=True, init_log_std=-1.0).to(device)
        if model_init:
            model_init(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    env.difficulty = 0.0
    obs = env.reset()
    B, S = args.envs, args.steps
    ep_ret, ep_len = np.zeros(B), np.zeros(B)
    finished, extra_stats = [], []
    t_start = time.time()

    for update in range(1, args.updates + 1):
        frac = (update - 1) / max(1, args.updates - 1)
        env.difficulty = min(1.0, frac / args.curriculum) if args.curriculum > 0 else 1.0
        for g in opt.param_groups:
            g["lr"] = args.lr * (1.0 - 0.9 * frac)

        b_obs = np.zeros((S, B, obs_dim), np.float32)
        b_act = np.zeros((S, B, act_dim), np.float32)
        b_logp, b_val = np.zeros((S, B), np.float32), np.zeros((S, B), np.float32)
        b_rew, b_done = np.zeros((S, B), np.float32), np.zeros((S, B), np.float32)

        model.eval()
        for t in range(S):
            b_obs[t] = obs
            with torch.no_grad():
                a, logp, v = model.act(torch.as_tensor(obs, device=device))
            a = a.cpu().numpy()
            obs, r, term, trunc, info = env.step(a)
            done = term | trunc
            b_act[t], b_logp[t], b_val[t] = a, logp.cpu().numpy(), v.cpu().numpy()
            b_rew[t], b_done[t] = r * args.reward_scale, done
            ep_ret += r; ep_len += 1
            for i in np.nonzero(done)[0]:
                finished.append((ep_ret[i], ep_len[i], bool(term[i])))
                if episode_stat:
                    extra_stats.append(episode_stat(info, i))
                ep_ret[i] = ep_len[i] = 0
        model.train()

        with torch.no_grad():
            last_v = model(torch.as_tensor(obs, device=device))[2].cpu().numpy()
        adv = np.zeros((S, B), np.float32)
        last = np.zeros(B, np.float32)
        for t in reversed(range(S)):
            nv = last_v if t == S - 1 else b_val[t + 1]
            nonterm = 1.0 - b_done[t]
            delta = b_rew[t] + args.gamma * nv * nonterm - b_val[t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        ret = adv + b_val

        T = lambda x: torch.as_tensor(x.reshape(S * B, *x.shape[2:]), device=device)
        o_t, a_t, lp_t, adv_t, ret_t = T(b_obs), T(b_act), T(b_logp), T(adv), T(ret)
        model.update_obs_stats(o_t)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = S * B
        for _ in range(args.epochs):
            perm = torch.as_tensor(rng.permutation(n), device=device)
            for s in range(0, n, args.minibatch):
                mb = perm[s:s + args.minibatch]
                logp, ent, value = model.evaluate(o_t[mb], a_t[mb])
                ratio = (logp - lp_t[mb]).exp()
                pg = torch.max(-adv_t[mb] * ratio, -adv_t[mb] * ratio.clamp(0.8, 1.2)).mean()
                v_loss = 0.5 * (value - ret_t[mb]).pow(2).mean()
                if update <= args.critic_warmup:
                    loss = 0.5 * v_loss
                else:
                    loss = pg + 0.5 * v_loss - args.ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        if update % 5 == 0 or update == 1:
            recent = finished[-100:]
            mr = np.mean([f[0] for f in recent]) if recent else float("nan")
            ml = np.mean([f[1] for f in recent]) if recent else float("nan")
            cr = np.mean([f[2] for f in recent]) if recent else float("nan")
            line = (f"update {update:4d} | difficulty {env.difficulty:.2f} | return {mr:7.1f} "
                    f"| ep len {ml:5.0f} | ended early {cr:5.1%}")
            if extra_stats:
                rs = extra_stats[-100:]
                for k in rs[0]:
                    line += f" | {k} {np.mean([x[k] for x in rs]):.2f}"
            line += f" | std {model.actor_log_std.exp().mean().item():.2f} | {time.time() - t_start:5.0f}s"
            print(line, flush=True)
        if eval_fn and (update % args.eval_every == 0 or update == args.updates):
            model.eval()
            print("   eval  " + eval_fn(model, device), flush=True)
            model.train()
            save(model, args.out, {"update": update, **(save_extra or {})})
    save(model, args.out, {"update": args.updates, **(save_extra or {})})
    print(f"saved {args.out}")
    return model
