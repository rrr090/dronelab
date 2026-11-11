"""Train the drone team to deliver cargo through generated maps (PPO).

Each episode draws a random map (theme: forest, city, gates, storm, mixed).
The curriculum starts on open, calm maps and ramps up obstacle density,
wind zones, gusts, cargo mass, rope length and motor wear.

Reward: follow the route reference, make progress, keep clear of obstacles,
fly smoothly, and above all avoid cargo damage. Delivery pays a bonus that
scales with the cargo health left.

Residual mode (default): the formation baseline keeps flying the route, and the
network only outputs a bounded correction on top of it (``--residual`` sets the
size). The correction starts near zero, so training starts from the baseline's
performance instead of from scratch. ``--residual 0`` trains a policy that
outputs the full action, warm-started from the cargo network.

Every evaluation flies validation maps, which are separate from the training
and test maps. The best-scoring network so far is kept next to ``--out`` as
``<name>_best.pt``, since the last update is not always the best one.

Usage:
    dronelab mission train                       # residual, default 400 updates
    dronelab mission train --updates 600
    dronelab mission train --residual 0 --init-from checkpoints/cargo_mlp.pt
    dronelab mission train --resume checkpoints/mission_mlp.pt --updates 300
    dronelab mission train --history --jerk-weight 1.0   # past frames as inputs, early jerk penalty
    dronelab mission train --resume checkpoints/chosen.pt --curriculum 0 --critic-warmup 0 --lr 5e-5
    dronelab mission train --resume checkpoints/chosen.pt --spike-weight 1.0 --curriculum 0 --critic-warmup 0 --lr 5e-5
    dronelab mission train --resume checkpoints/chosen.pt --privileged --curriculum 0 --critic-warmup 10 --lr 5e-5
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from dronelab.cargo.env import ACT_DIM
from dronelab.mission.env import HISTORY_DIM, MISSION_OBS_DIM, PRIV_DIM, MissionEnv, compose_residual
from dronelab.mission.testing import mission_maps, run_missions, summarize_missions
from dronelab.ppo import add_ppo_args, save, train_ppo

VALIDATION_SEED = 4_000_000      # between the training (< 1e6) and test (5e6) map seeds


def make_eval(residual=0.0, best_path=None, maps=None, history=False, extra=None):
    maps = maps or mission_maps(difficulties=(0.5, 1.0), per=3, seed=VALIDATION_SEED)   # 30 maps
    best = -np.inf

    def quick_eval(model, device):
        nonlocal best
        @torch.no_grad()
        def batched(O):
            a = model(torch.as_tensor(O, device=device))[0].cpu().numpy()
            return compose_residual(O, a, residual) if residual > 0 else a

        class P:
            pass
        pol = P(); pol.batched = batched; pol.history = history
        s = summarize_missions(run_missions(pol, maps))
        fails = ", ".join(f"{k} x{v}" for k, v in s["failures"].items()) or "none"
        improved = best_path is not None and s["mean_score"] > best
        if improved:
            best = s["mean_score"]
            save(model, best_path, {"residual": residual, "validation_score": best, **(extra or {})})
        return (f"delivered {s['delivered_rate']:.0%} | score {s['mean_score']:.1f} "
                f"| health {s['health_if_delivered']:.1f} | failures: {fails}"
                + (" | new best" if improved else ""))
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


def zero_correction(init_std):
    """Residual start: the correction is exactly zero, so the first rollouts fly
    the baseline, with small exploration noise around it."""
    @torch.no_grad()
    def init(model):
        model.actor_mean.weight.zero_()
        model.actor_mean.bias.zero_()
        model.actor_log_std.fill_(float(np.log(init_std)))
    return init


def episode_stat(info, i):
    return {"delivered": float(info["delivered"][i]), "health": float(info["health"][i])}


def main(argv=None):
    p = add_ppo_args(argparse.ArgumentParser(prog="dronelab mission train", description=__doc__,
                                             formatter_class=argparse.RawDescriptionHelpFormatter), out="checkpoints/mission_mlp.pt",
                     updates=400, eval_every=25, hidden=128, lr=1.5e-4, ent_coef=0.0005,
                     critic_warmup=10)
    p.add_argument("--residual", type=float, default=0.1,
                   help="size of the learned correction on top of the baseline (0 = full actions)")
    p.add_argument("--init-from", default="",
                   help="warm-start a full-action policy from a trained cargo network")
    p.add_argument("--init-std", type=float, default=None,
                   help="initial exploration noise (default 0.2 residual, 0.12 warm start)")
    p.add_argument("--history", action="store_true",
                   help="add peak rope tension and past frames to the inputs")
    p.add_argument("--jerk-weight", type=float, default=None,
                   help="extra penalty on rope tension approaching the damage level "
                        "(default 0, or the value saved in the --resume checkpoint)")
    p.add_argument("--spike-weight", type=float, default=None,
                   help="quadratic penalty on rope tension above the damage level "
                        "(default 0, or the value saved in the --resume checkpoint)")
    p.add_argument("--privileged", action="store_true",
                   help="give the critic its own network with wind, gust and cargo state; "
                        "the policy's inputs do not change")
    args = p.parse_args(argv)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        args.residual = float(ckpt.get("residual", 0.0))
        args.history = ckpt["config"]["obs_dim"] > MISSION_OBS_DIM
        args.privileged = args.privileged or ckpt["config"].get("priv_dim", 0) > 0
        if args.jerk_weight is None:
            args.jerk_weight = float(ckpt.get("jerk_weight", 0.0))
        if args.spike_weight is None:
            args.spike_weight = float(ckpt.get("spike_weight", 0.0))
    args.jerk_weight = args.jerk_weight or 0.0
    args.spike_weight = args.spike_weight or 0.0
    env = MissionEnv(args.envs, seed=args.seed, difficulty=0.0, autoreset=True, residual=args.residual,
                     history=args.history, jerk_weight=args.jerk_weight, spike_weight=args.spike_weight)
    if args.resume:
        init = None
    elif args.residual > 0:
        init = zero_correction(args.init_std or 0.2)
    elif args.init_from:
        init = warm_start_from(args.init_from, args.init_std or 0.12)
    else:
        init = None
    print(f"mode: {'residual x%.2f' % args.residual if args.residual > 0 else 'full actions'}")
    best_path = "{}_best{}".format(*os.path.splitext(args.out))
    extra = {"jerk_weight": args.jerk_weight, "spike_weight": args.spike_weight}
    obs_dim = MISSION_OBS_DIM + (HISTORY_DIM if args.history else 0)
    train_ppo(args, env, obs_dim, ACT_DIM,
              eval_fn=make_eval(args.residual, best_path, history=args.history, extra=extra),
              episode_stat=episode_stat, model_init=init,
              save_extra={"residual": args.residual, **extra},
              priv_dim=PRIV_DIM if args.privileged else 0)


if __name__ == "__main__":
    main()
