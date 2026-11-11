"""Stress test: fly policies through the same unseen maps with hardware effects switched on.

The default simulator has instant motors, no control delay, perfect sensors and the
nominal payload and rope (see cargo.env.Realism). This command adds them one at a time
at the "harsh" level, then all together at the "realistic" and "harsh" levels, and
compares each policy with the formation baseline map by map.

Presets (values per effect):
    realistic  typical small-drone hardware: 30 ms motor lag, 20 ms delay, cm-level
               sensor noise, 10%/min battery sag, payload within the training range
    harsh      beyond anything seen in training: 60 ms motor lag, 40 ms delay, double the
               noise, worn motors, 1.5x wind, payload +-0.3 kg, rope stiffness x/2

Usage:
    dronelab mission stress --checkpoint checkpoints/chosen.pt
    dronelab mission stress --checkpoint a.pt b.pt --per 10 --seed 16000000 --json outputs/stress.json
    dronelab mission stress --checkpoint a.pt --conditions default realistic harsh
    dronelab mission stress --checkpoint a.pt --conditions latency@realistic wind_scale@realistic --jobs 8

A condition is 'default', a preset, an effect alone at its harsh value ('latency'), or an
effect alone at another preset's value ('latency@realistic'), or a preset with one
effect switched off ('realistic-wind_scale', for leave-one-out attribution). --jobs flies the
(condition, policy) pairs in parallel processes; results are identical to --jobs 1.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields, replace

import numpy as np

from dronelab.cargo.env import Realism
from dronelab.mission.maps import THEMES
from dronelab.mission.testing import (
    MissionFormationController, bootstrap_ci, load_mission_policy, mission_maps, run_missions,
    summarize_missions,
)

STRESS_SEED = 16_000_000         # maps not used by training, evaluation or earlier comparisons

PRESETS = {
    "realistic": Realism(motor_tau=0.03, latency=1, sensor_noise=0.02, sensor_bias=0.01, battery_sag=0.10,
                         motor_wear=0.10, wind_scale=1.25, payload_spread=0.2, rope_len_spread=0.1,
                         rope_k_spread=1.5),
    "harsh": Realism(motor_tau=0.06, latency=2, sensor_noise=0.04, sensor_bias=0.02, battery_sag=0.20,
                     motor_wear=0.20, wind_scale=1.5, payload_spread=0.3, rope_len_spread=0.15,
                     rope_k_spread=2.0),
}
EFFECTS = [f.name for f in fields(Realism)]


def condition(name: str) -> Realism:
    """'default', a preset name, one effect at its harsh value (e.g. 'latency'),
    one effect at a named preset's value (e.g. 'latency@realistic'), or a preset with one
    effect back at its default (e.g. 'realistic-wind_scale')."""
    if name == "default":
        return Realism()
    if name in PRESETS:
        return PRESETS[name]
    preset, _, off = name.partition("-")
    if preset in PRESETS and off in EFFECTS:
        return replace(PRESETS[preset], **{off: getattr(Realism(), off)})
    effect, _, preset = name.partition("@")
    if effect in EFFECTS and (preset or "harsh") in PRESETS:
        return Realism(**{effect: getattr(PRESETS[preset or "harsh"], effect)})
    raise ValueError(f"unknown condition {name!r}")


def _fly(job):
    """One (condition, policy) pair: per-map scores and the summary."""
    cname, pname, path, maps, env_seed = job
    import torch
    torch.set_num_threads(1)
    pol = MissionFormationController() if path is None else load_mission_policy(path)
    rs = run_missions(pol, maps, seed=env_seed, realism=condition(cname))
    return cname, pname, summarize_missions(rs), [r.score for r in rs]


def main(argv=None):
    p = argparse.ArgumentParser(prog="dronelab mission stress", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--conditions", nargs="+", default=["default", *EFFECTS, "realistic", "harsh"])
    p.add_argument("--themes", nargs="*", default=THEMES, choices=THEMES)
    p.add_argument("--difficulties", nargs="*", type=float, default=[0.5, 1.0])
    p.add_argument("--per", type=int, default=10, help="maps per theme x difficulty")
    p.add_argument("--seed", type=int, default=STRESS_SEED)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--json", default=None)
    p.add_argument("--jobs", type=int, default=1, help="parallel processes")
    args = p.parse_args(argv)
    for c in args.conditions:
        condition(c)                                   # fail early on a typo

    maps = mission_maps(args.themes, args.difficulties, args.per, args.seed)
    paths = {"formation": None}
    for path in args.checkpoint:
        paths[os.path.splitext(os.path.basename(path))[0]] = path
    names = list(paths)
    jobs = [(c, n, paths[n], maps, args.env_seed) for c in args.conditions for n in names]
    pool = ProcessPoolExecutor(args.jobs) if args.jobs > 1 else None
    done = pool.map(_fly, jobs) if pool else map(_fly, jobs)
    flown = {}
    print(f"{len(maps)} unseen maps (seed {args.seed}), policies: {', '.join(names)}\n")

    head = f"{'condition':<17}" + "".join(f"{n[:14]:>16}" for n in names) + "   learned - baseline [95% CI]"
    print(head + "\n" + "-" * len(head))
    out = {"maps": len(maps), "seed": args.seed, "presets": {k: asdict(v) for k, v in PRESETS.items()},
           "conditions": {}}
    for cname in args.conditions:
        real = condition(cname)
        while not all((cname, n) in flown for n in names):
            c, n, summary, sc = next(done)
            flown[c, n] = (summary, sc)
        scores = {n: np.array(flown[cname, n][1]) for n in names}
        row = {n: {**flown[cname, n][0], "scores": flown[cname, n][1]} for n in names}
        diffs = {}
        for n in names[1:]:
            m, lo, hi = bootstrap_ci(scores[n] - scores["formation"])
            diffs[n] = {"diff": m, "ci_low": lo, "ci_high": hi}
        out["conditions"][cname] = {"realism": asdict(real), "policies": row, "vs_baseline": diffs}
        cells = "".join(f"{row[n]['mean_score']:>9.1f} ({row[n]['delivered_rate']:>4.0%})" for n in names)
        d = diffs[names[-1]]
        print(f"{cname:<17}{cells}   {d['diff']:+5.1f} [{d['ci_low']:+.1f}, {d['ci_high']:+.1f}]", flush=True)
    print("\ncells: mean score (delivered share); the difference column is for the last checkpoint")
    if pool:
        pool.shutdown()
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
