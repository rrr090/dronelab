"""Run the drone testing environment against a policy.

Usage:
    dronelab hover evaluate --policy pd                        # reference PD controller
    dronelab hover evaluate --policy ppo --checkpoint checkpoints/hover_mlp.pt
    dronelab hover evaluate --policy ppo --scenarios nominal wind latency --plot outputs/eval.png
    dronelab hover evaluate --list                             # show available scenarios
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from dronelab.hover.env import DT, TARGET_Z
from dronelab.hover.testing import (
    SCENARIOS, PDController, format_table, hover_policy, load_ppo_policy, run_suite,
)


def plot_results(results: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    cols = min(4, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        for ep in results[name]:
            tr = ep.trajectory
            t = np.arange(len(tr)) * DT
            err = np.hypot(tr[:, 0], tr[:, 1] - TARGET_Z)
            ax.plot(t, err, color="tab:red" if ep.crashed else "tab:blue", alpha=0.5, lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("position error [m]")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle("Position error per episode (red = crashed)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    print(f"saved plot {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="dronelab hover evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["ppo", "pd", "hover"], default="ppo")
    p.add_argument("--checkpoint", default="checkpoints/hover_mlp.pt")
    p.add_argument("--stochastic", action="store_true", help="sample PPO actions")
    p.add_argument("--scenarios", nargs="*", default=None, help="subset of scenario names")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1000, help="first eval seed (train uses 0)")
    p.add_argument("--plot", default=None, help="save an error-vs-time plot here")
    p.add_argument("--json", default=None, help="save summary metrics as JSON here")
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        for s in SCENARIOS.values():
            print(f"{s.name:<14}{s.description}")
        return

    if args.policy == "ppo":
        policy = load_ppo_policy(args.checkpoint, deterministic=not args.stochastic)
    elif args.policy == "pd":
        policy = PDController()
    else:
        policy = hover_policy

    names = args.scenarios or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        p.error(f"unknown scenarios: {unknown}. Use --list.")

    summaries, results = run_suite(
        policy, [SCENARIOS[n] for n in names], episodes=args.episodes, seed=args.seed
    )
    print(f"policy: {args.policy} | episodes per scenario: {args.episodes}\n")
    print(format_table(summaries))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"saved metrics {args.json}")
    if args.plot:
        plot_results(results, args.plot)


if __name__ == "__main__":
    main()
