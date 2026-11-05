"""Score policies on cargo delivery missions through unseen generated maps.

Usage:
    dronelab mission evaluate                          # trained policy vs baseline
    dronelab mission evaluate --policy rl --per 5      # 5 maps per theme x difficulty
    dronelab mission evaluate --themes storm city --difficulties 1.0
"""
from __future__ import annotations

import argparse
import json
import os

from dronelab.mission.maps import THEMES
from dronelab.mission.testing import (
    EVAL_SEED, MissionFormationController, format_mission_table, load_mission_policy,
    mission_maps, run_missions, summarize_missions,
)


def main(argv=None):
    p = argparse.ArgumentParser(prog="dronelab mission evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", nargs="+", choices=["rl", "formation"], default=["rl", "formation"])
    p.add_argument("--checkpoint", default="checkpoints/mission_mlp.pt")
    p.add_argument("--themes", nargs="*", default=THEMES, choices=THEMES)
    p.add_argument("--difficulties", nargs="*", type=float, default=[0.5, 1.0])
    p.add_argument("--per", type=int, default=3, help="maps per theme x difficulty")
    p.add_argument("--cruise", type=float, default=1.0, help="route cruise speed [m/s]")
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    maps = mission_maps(args.themes, args.difficulties, args.per, args.seed)
    print(f"{len(maps)} unseen maps ({', '.join(args.themes)} x difficulty {args.difficulties})\n")
    out = {}
    for name in args.policy:
        pol = load_mission_policy(args.checkpoint) if name == "rl" else MissionFormationController()
        res = run_missions(pol, maps, cruise=args.cruise)
        print(f"policy: {name}")
        print(format_mission_table(res) + "\n")
        out[name] = summarize_missions(res)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
