"""Build the static project site: a results page and a replay viewer.

Every policy flies every evaluation map once, and the page is generated from
those flights. The replays are a selection of the same flights the numbers
describe, not a separate run (a smaller batch would draw different gusts).

Usage:
    dronelab mission site                                    # 100 maps, the chosen policy
    dronelab mission site --checkpoint a.pt --per 1 --out outputs/site_preview
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from dronelab.mission.maps import THEMES
from dronelab.mission.testing import (
    MissionFormationController, MissionResult, compare_to_baseline,
    load_mission_policy, mission_maps, run_missions, summarize_missions,
)
from dronelab.mission.viewer import viewer_data, write_viewer
from dronelab.viz import write_page

TEMPLATE = Path(__file__).with_name("site.html")
IMAGES = Path(__file__).parents[2] / "docs" / "images"
PAGE_IMAGES = ("hover-lanes.png", "cargo-storm.png", "mission-forest.png")
SEEDS = ["checkpoints/chosen.pt"]
REPORT_SEED = 10_500_000         # maps never used for training or for choosing a model


def median_seed(runs: dict[str, Sequence[MissionResult]]) -> str:
    """The seed with the middle mean score, so the replays show a typical policy."""
    ranked = sorted(runs, key=lambda name: np.mean([r.score for r in runs[name]]))
    return ranked[len(ranked) // 2]


def pick_showcase(trained: Sequence[MissionResult], baseline: Sequence[MissionResult]) -> list[int]:
    """Map indices to replay: the trained policy's best map of each theme, then its worst map overall.

    Best means the largest score gain over the baseline, most impressive first.
    The worst map always comes last, so the replays don't only show wins.
    """
    diffs = np.array([t.score - b.score for t, b in zip(trained, baseline)])
    themes = [r.map.theme for r in baseline]
    worst = int(diffs.argmin())
    best = [max((i for i, th in enumerate(themes) if th == theme), key=lambda i: diffs[i])
            for theme in dict.fromkeys(themes)]
    return sorted((i for i in best if i != worst), key=lambda i: -diffs[i]) + [worst]


def theme_scores(runs: dict[str, Sequence[MissionResult]], baseline: Sequence[MissionResult]) -> list[dict]:
    """Scores per map theme, with the seeds pooled."""
    rows = []
    for theme in dict.fromkeys(r.map.theme for r in baseline):
        base = summarize_missions([r for r in baseline if r.map.theme == theme])
        trained = summarize_missions([r for rs in runs.values() for r in rs if r.map.theme == theme])
        rows.append({"theme": theme, "maps": base["missions"], "trained": trained["mean_score"],
                     "baseline": base["mean_score"], "trained_delivered": trained["delivered_rate"],
                     "baseline_delivered": base["delivered_rate"]})
    return rows


def _outcome(r: MissionResult) -> dict:
    return {"delivered": r.delivered, "score": round(r.score, 1), "reason": r.fail_reason}


def build_site_data(runs: dict[str, Sequence[MissionResult]], baseline: Sequence[MissionResult],
                    replay_seed: str, picks: Sequence[int]) -> dict:
    trained = runs[replay_seed]
    return {
        "comparison": compare_to_baseline(runs, baseline),
        "summary": {"trained": summarize_missions([r for rs in runs.values() for r in rs]),
                    "baseline": summarize_missions(baseline)},
        "themes": theme_scores(runs, baseline),
        "replay_seed": replay_seed,
        "showcase": [{"theme": baseline[i].map.theme, "seed": baseline[i].map.seed,
                      "difficulty": baseline[i].map.difficulty, "worst": k == len(picks) - 1,
                      "trained": _outcome(trained[i]), "baseline": _outcome(baseline[i])}
                     for k, i in enumerate(picks)],
        "generated": _dt.date.today().isoformat(),
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="dronelab mission site", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", default=SEEDS, help="one checkpoint per training seed")
    p.add_argument("--themes", nargs="*", default=THEMES, choices=THEMES)
    p.add_argument("--difficulties", nargs="*", type=float, default=[0.5, 1.0])
    p.add_argument("--per", type=int, default=10, help="maps per theme x difficulty")
    p.add_argument("--cruise", type=float, default=1.0, help="route cruise speed [m/s]")
    p.add_argument("--seed", type=int, default=REPORT_SEED)
    p.add_argument("--stride", type=int, default=2, help="keep every Nth frame of the replays")
    p.add_argument("--out", default="site")
    args = p.parse_args(argv)

    maps = mission_maps(args.themes, args.difficulties, args.per, args.seed)
    print(f"flying {len(maps)} maps x {len(args.checkpoint) + 1} policies ...")
    runs = {os.path.splitext(os.path.basename(path))[0]:
            run_missions(load_mission_policy(path), maps, cruise=args.cruise, record=True)
            for path in args.checkpoint}
    baseline = run_missions(MissionFormationController(), maps, cruise=args.cruise, record=True)

    replay_seed = median_seed(runs)
    picks = pick_showcase(runs[replay_seed], baseline)
    data = build_site_data(runs, baseline, replay_seed, picks)

    out = Path(args.out)
    write_viewer(viewer_data({"trained": runs[replay_seed], "baseline": baseline}, picks, args.stride),
                 out / "viewer.html")
    write_page(TEMPLATE, data, out / "index.html")
    (out / "images").mkdir(exist_ok=True)
    for name in PAGE_IMAGES:
        shutil.copy(IMAGES / name, out / "images" / name)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"saved {out.resolve()} ({size:.1f} MB), replays from {replay_seed}")
    print(f"verdict: {data['comparison']['verdict']}")


if __name__ == "__main__":
    main()
