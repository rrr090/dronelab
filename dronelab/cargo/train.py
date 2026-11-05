"""Train three drones to carry one cargo on ropes, under gusts and faults (PPO).

Runs many environments in parallel and ramps up a curriculum: early updates
are calm, then wind, gusts, cargo mass, rope length, motor wear, sensor noise
and cargo kicks are randomised more and more until full difficulty.

Usage:
    dronelab cargo train                         # default ~400 updates
    dronelab cargo train --updates 800 --envs 128
    dronelab cargo train --resume checkpoints/cargo_mlp.pt --updates 200
"""
from __future__ import annotations

import argparse

import torch

from dronelab.cargo.env import ACT_DIM, OBS_DIM, CargoEnv
from dronelab.cargo.testing import CARGO_SCENARIOS, run_cargo_episode, summarize_cargo
from dronelab.ppo import add_ppo_args, train_ppo


def quick_eval(model, device, episodes=4):
    @torch.no_grad()
    def pol(o):
        return model(torch.as_tensor(o, device=device).unsqueeze(0))[0].squeeze(0).cpu().numpy()

    parts = []
    for name in ["nominal", "gusts", "storm"]:
        eps = [run_cargo_episode(pol, CARGO_SCENARIOS[name], seed=50_000 + i) for i in range(episodes)]
        s = summarize_cargo(eps)
        parts.append(f"{name}: {s['success_rate']:.0%} ok / {s['crash_rate']:.0%} crash")
    return "  ".join(parts)


def main(argv=None):
    p = add_ppo_args(argparse.ArgumentParser(prog="dronelab cargo train", description=__doc__,
                                             formatter_class=argparse.RawDescriptionHelpFormatter), out="checkpoints/cargo_mlp.pt")
    args = p.parse_args(argv)
    env = CargoEnv(args.envs, seed=args.seed, difficulty=0.0, autoreset=True)
    train_ppo(args, env, OBS_DIM, ACT_DIM, eval_fn=quick_eval)


if __name__ == "__main__":
    main()
