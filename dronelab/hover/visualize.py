"""Build an interactive 3D viewer of drones flying the test scenarios.

Usage:
    dronelab hover visualize --policy pd                                # reference controller
    dronelab hover visualize --policy ppo pd --episodes 3               # compare side by side
    dronelab hover visualize --policy ppo --scenarios wind gusts impulse
    dronelab hover visualize --policy ppo --out outputs/run.html --no-open

Opens the generated HTML in your browser (needs internet for three.js).
"""
from __future__ import annotations

import argparse
import webbrowser

from dronelab.hover.testing import SCENARIOS, PDController, hover_policy, load_ppo_policy
from dronelab.hover.viewer import build_viewer_data, write_viewer


def main(argv=None):
    p = argparse.ArgumentParser(prog="dronelab hover visualize", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", nargs="+", choices=["ppo", "pd", "hover"], default=["ppo"])
    p.add_argument("--checkpoint", default="checkpoints/hover_mlp.pt")
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--episodes", type=int, default=3, help="episodes (lanes) per policy")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default="outputs/drone_viewer.html")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)

    names = args.scenarios or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        p.error(f"unknown scenarios: {unknown}. See: dronelab hover evaluate --list")

    makers = {
        "ppo": lambda: load_ppo_policy(args.checkpoint),
        "pd": PDController,
        "hover": lambda: hover_policy,
    }
    policies = {k.upper(): makers[k]() for k in dict.fromkeys(args.policy)}

    data = build_viewer_data(policies, [SCENARIOS[n] for n in names],
                             episodes=args.episodes, seed=args.seed)
    path = write_viewer(data, args.out).resolve()
    size_mb = path.stat().st_size / 1e6
    print(f"saved {path} ({size_mb:.1f} MB)")
    if not args.no_open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
