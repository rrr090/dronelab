"""Score policies on cargo delivery missions through unseen generated maps.

Usage:
    dronelab mission evaluate                          # trained policy vs baseline
    dronelab mission evaluate --checkpoint a.pt b.pt c.pt   # one checkpoint per training seed
    dronelab mission evaluate --policy rl --per 5      # 5 maps per theme x difficulty
    dronelab mission evaluate --themes storm city --difficulties 1.0
    dronelab mission evaluate --policy rl --checkpoint a.pt b.pt --json out.json   # per-map scores for paired comparisons
"""
from __future__ import annotations

import argparse
import json
import os

from dronelab.mission.maps import THEMES
from dronelab.mission.testing import (
    EVAL_SEED, MissionFormationController, compare_to_baseline, format_comparison,
    format_mission_table, load_mission_policy, mission_maps, run_missions, summarize_missions,
)


def main(argv=None):
    p = argparse.ArgumentParser(prog="dronelab mission evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", nargs="+", choices=["rl", "formation"], default=["rl", "formation"])
    p.add_argument("--checkpoint", nargs="+", default=["checkpoints/mission_mlp.pt"],
                   help="one or more checkpoints, e.g. one per training seed")
    p.add_argument("--themes", nargs="*", default=THEMES, choices=THEMES)
    p.add_argument("--difficulties", nargs="*", type=float, default=[0.5, 1.0])
    p.add_argument("--per", type=int, default=10, help="maps per theme x difficulty")
    p.add_argument("--cruise", type=float, default=1.0, help="route cruise speed [m/s]")
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--env-seed", type=int, default=0, help="seed for gusts and sensor noise on the same maps")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    maps = mission_maps(args.themes, args.difficulties, args.per, args.seed)
    print(f"{len(maps)} unseen maps ({', '.join(args.themes)} x difficulty {args.difficulties})\n")
    policies = {}
    if "rl" in args.policy:
        for path in args.checkpoint:
            name = "rl" if len(args.checkpoint) == 1 else os.path.splitext(os.path.basename(path))[0]
            policies[name] = load_mission_policy(path)
    if "formation" in args.policy:
        policies["formation"] = MissionFormationController()

    out, results = {}, {}
    for name, pol in policies.items():
        results[name] = run_missions(pol, maps, cruise=args.cruise, seed=args.env_seed)
        print(f"policy: {name}")
        print(format_mission_table(results[name]) + "\n")
        out[name] = {**summarize_missions(results[name]),
                     "per_map": [{"map_seed": int(r.map.seed), "theme": r.map.theme,
                                  "difficulty": float(r.map.difficulty), "score": float(r.score),
                                  "delivered": bool(r.delivered), "fail_reason": r.fail_reason,
                                  "health": float(r.health), "damage": r.damage, "time_s": float(r.time_s)}
                                 for r in results[name]]}
    baseline = results.pop("formation", None)
    if baseline is not None and results:
        out["comparison"] = compare_to_baseline(results, baseline)
        print(format_comparison(out["comparison"]) + "\n")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
