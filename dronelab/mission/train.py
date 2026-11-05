"""Train the drone team to deliver cargo through generated maps (PPO).

Each episode draws a random map (theme: forest, city, gates, storm, mixed).
The curriculum starts on open, calm maps and ramps up obstacle density,
wind zones, gusts, cargo mass, rope length and motor wear.

Reward: follow the route reference, make progress, keep clear of obstacles,
fly smoothly, and above all avoid cargo damage. Delivery pays a bonus that
scales with the cargo health left.

Usage:
    dronelab mission train                       # default 600 updates
    dronelab mission train --updates 1000
    dronelab mission train --resume checkpoints/mission_mlp.pt --updates 300
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from dronelab.cargo.env import ACT_DIM
from dronelab.mission.env import MISSION_OBS_DIM, MissionEnv
from dronelab.mission.testing import mission_maps, run_missions, summarize_missions
from dronelab.ppo import add_ppo_args, train_ppo


def make_eval():
    maps = mission_maps(difficulties=(0.6,), per=1, seed=4_000_000)   # 5 maps, one per theme

    def quick_eval(model, device):
        @torch.no_grad()
        def batched(O):
            return model(torch.as_tensor(O, device=device))[0].cpu().numpy()

        class P:
            pass
        pol = P(); pol.batched = batched
        s = summarize_missions(run_missions(pol, maps))
        fails = ", ".join(f"{k} x{v}" for k, v in s["failures"].items()) or "none"
        return (f"delivered {s['delivered_rate']:.0%} | score {s['mean_score']:.1f} "
                f"| health {s['health_if_delivered']:.1f} | failures: {fails}")
    return quick_eval


@torch.no_grad()
def warm_start_from(path, init_std=0.12):
    """Copy a trained cargo-hover network into the bigger mission network.

    The first 63 mission inputs mean the same thing as the cargo inputs (cargo
    error is now measured from the moving route reference), so their weights
    are copied; weights for the 30 new route/obstacle inputs start at zero.
    The value head is left freshly initialised (mission rewards differ from
    hover rewards), and exploration noise is reset low: noisy thrust jerks
    the ropes and destroys the cargo, which would drown the learning signal.
    """
    def init(model):
        src = torch.load(path, map_location="cpu")
        sd = src["state_dict"]
        n_old = src["config"]["obs_dim"]
        if src["config"]["hidden"] != model.config["hidden"]:
            raise ValueError("warm start needs the same --hidden size as the cargo network")
        tgt = model.state_dict()
        for k, v in sd.items():
            if k == "body.0.weight":
                tgt[k].zero_(); tgt[k][:, :n_old] = v
            elif k in ("obs_mean", "obs_var"):
                tgt[k][:n_old] = v
            elif k == "obs_count":
                tgt[k].copy_(v)
            elif k.startswith("critic."):
                continue
            elif k == "actor_log_std":
                tgt[k].fill_(float(np.log(init_std)))
            else:
                tgt[k].copy_(v)
        model.load_state_dict(tgt)
        print(f"warm-started from {path}")
    return init


def episode_stat(info, i):
    return {"delivered": float(info["delivered"][i]), "health": float(info["health"][i])}


def main(argv=None):
    p = add_ppo_args(argparse.ArgumentParser(prog="dronelab mission train", description=__doc__,
                                             formatter_class=argparse.RawDescriptionHelpFormatter), out="checkpoints/mission_mlp.pt",
                     updates=600, eval_every=50, hidden=128, lr=1.5e-4, ent_coef=0.0005,
                     critic_warmup=10)
    p.add_argument("--init-from", default="checkpoints/cargo_mlp.pt",
                   help="warm-start from a trained cargo network ('' to train from scratch)")
    p.add_argument("--init-std", type=float, default=0.12, help="exploration noise after warm start")
    args = p.parse_args(argv)
    env = MissionEnv(args.envs, seed=args.seed, difficulty=0.0, autoreset=True)
    init = warm_start_from(args.init_from, args.init_std) if args.init_from and not args.resume else None
    train_ppo(args, env, MISSION_OBS_DIM, ACT_DIM, eval_fn=make_eval(),
              episode_stat=episode_stat, model_init=init)


if __name__ == "__main__":
    main()
