"""3D viewer of the drone team delivering cargo through generated maps.

Usage:
    dronelab mission visualize                               # trained vs baseline, 5 maps
    dronelab mission visualize --themes city storm --difficulty 1.0 --per 2
    dronelab mission visualize --policy rl --seed 123
"""
from __future__ import annotations

import argparse
import webbrowser

from dronelab.mission.maps import THEMES, generate_map
from dronelab.mission.testing import EVAL_SEED, MissionFormationController, load_mission_policy
from dronelab.mission.viewer import build_viewer_data, write_viewer


def main(argv=None):
    p = argparse.ArgumentParser(prog="dronelab mission visualize", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", nargs="+", choices=["rl", "formation"], default=["rl", "formation"])
    p.add_argument("--checkpoint", default="checkpoints/mission_mlp.pt")
    p.add_argument("--themes", nargs="*", default=THEMES, choices=THEMES)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--per", type=int, default=1, help="maps per theme")
    p.add_argument("--seed", type=int, default=EVAL_SEED + 777)
    p.add_argument("--cruise", type=float, default=1.0)
    p.add_argument("--out", default="outputs/mission_viewer.html")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)

    maps = [generate_map(args.seed + 100 * i + k, th, args.difficulty)
            for i, th in enumerate(args.themes) for k in range(args.per)]
    makers = {"rl": lambda: load_mission_policy(args.checkpoint), "formation": MissionFormationController}
    policies = {k.upper(): makers[k]() for k in dict.fromkeys(args.policy)}
    print(f"flying {len(maps)} maps x {len(policies)} policies ...")
    data = build_viewer_data(policies, maps, cruise=args.cruise)
    path = write_viewer(data, args.out).resolve()
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    for name, s in data["meta"]["summary"].items():
        print(f"  {name:<10} score {s['mean_score']:.1f} | delivered {s['delivered_rate']:.0%}")
    if not args.no_open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
