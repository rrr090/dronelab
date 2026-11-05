"""Run the cargo test suite against a policy.

Usage:
    dronelab cargo evaluate --policy rl                 # trained network
    dronelab cargo evaluate --policy formation          # hand-written baseline
    dronelab cargo evaluate --policy rl --scenarios storm cargo_kick --episodes 20
    dronelab cargo evaluate --list
"""
from __future__ import annotations

import argparse
import json
import os

from dronelab.cargo.testing import (
    CARGO_SCENARIOS, FormationController, format_cargo_table, load_cargo_policy, run_cargo_suite,
)


def main(argv=None):
    p = argparse.ArgumentParser(prog="dronelab cargo evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["rl", "formation"], default="rl")
    p.add_argument("--checkpoint", default="checkpoints/cargo_mlp.pt")
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--json", default=None)
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        for s in CARGO_SCENARIOS.values():
            print(f"{s.name:<14}{s.description}")
        return
    names = args.scenarios or list(CARGO_SCENARIOS)
    unknown = [n for n in names if n not in CARGO_SCENARIOS]
    if unknown:
        p.error(f"unknown scenarios: {unknown}. Use --list.")

    policy = load_cargo_policy(args.checkpoint) if args.policy == "rl" else FormationController()
    summaries, _ = run_cargo_suite(policy, [CARGO_SCENARIOS[n] for n in names],
                                   episodes=args.episodes, seed=args.seed)
    print(f"policy: {args.policy} | episodes per scenario: {args.episodes}\n")
    print(format_cargo_table(summaries))
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"saved metrics {args.json}")


if __name__ == "__main__":
    main()
