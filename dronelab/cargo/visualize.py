"""Build an interactive 3D viewer of the drone team carrying cargo.

Usage:
    dronelab cargo visualize --policy rl formation       # trained vs baseline, side by side
    dronelab cargo visualize --policy rl --scenarios storm cargo_kick motor_fault
    dronelab cargo visualize --policy formation --out outputs/cargo.html --no-open
"""
from __future__ import annotations

import argparse
import webbrowser

from dronelab.cargo.testing import CARGO_SCENARIOS, FormationController, load_cargo_policy
from dronelab.cargo.viewer import build_viewer_data, write_viewer


def main(argv=None):
    p = argparse.ArgumentParser(prog="dronelab cargo visualize", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", nargs="+", choices=["rl", "formation"], default=["rl", "formation"])
    p.add_argument("--checkpoint", default="checkpoints/cargo_mlp.pt")
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--episodes", type=int, default=2, help="teams per policy")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default="outputs/cargo_viewer.html")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)

    names = args.scenarios or list(CARGO_SCENARIOS)
    unknown = [n for n in names if n not in CARGO_SCENARIOS]
    if unknown:
        p.error(f"unknown scenarios: {unknown}. See: dronelab cargo evaluate --list")

    makers = {"rl": lambda: load_cargo_policy(args.checkpoint), "formation": FormationController}
    policies = {k.upper(): makers[k]() for k in dict.fromkeys(args.policy)}
    print(f"running {len(names)} scenarios x {len(policies)} policies x {args.episodes} teams ...")
    data = build_viewer_data(policies, [CARGO_SCENARIOS[n] for n in names],
                                   episodes=args.episodes, seed=args.seed)
    path = write_viewer(data, args.out).resolve()
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    if not args.no_open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
