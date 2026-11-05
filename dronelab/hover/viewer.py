"""Export test-suite episodes to a self-contained interactive 3D viewer (HTML).

The page embeds the trajectories as JSON and renders them with three.js
(loaded from a CDN). Each episode gets its own lane, so several seeds and
policies can be compared side by side under the same scenario.

The sim is planar (x, z, pitch); in 3D, x is sideways, z is up, and lanes
are spread along the depth axis.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from dronelab.hover.env import ARM, DT, TARGET_Z
from dronelab.viz import write_page
from dronelab.hover.testing import (
    SCENARIOS, SUCCESS_RADIUS, EpisodeResult, Policy, Scenario, run_episode, summarize,
)

TEMPLATE = Path(__file__).with_name("viewer.html")


def _r(a, nd: int = 4) -> list:
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def export_episode(ep: EpisodeResult, scenario: Scenario, policy_name: str) -> dict:
    tr = ep.trajectory
    # Thrust as a fraction of max, including any motor degradation.
    thrust = (ep.actions + 1.0) / 2.0 * np.asarray(scenario.motor_gain)
    err = np.hypot(tr[:, 0], tr[:, 1] - TARGET_Z)
    return {
        "policy": policy_name, "seed": ep.seed, "crashed": ep.crashed,
        "success": ep.success, "steps": ep.steps, "final_err": round(ep.final_pos_err, 4),
        "x": _r(tr[:, 0]), "z": _r(tr[:, 1]), "th": _r(tr[:, 2]), "err": _r(err),
        "t0": _r(thrust[:, 0], 3), "t1": _r(thrust[:, 1], 3),
        "fx": _r(ep.forces[:, 0], 3), "fz": _r(ep.forces[:, 1], 3),
    }


def build_viewer_data(policies: Mapping[str, Policy], scenarios: Sequence[Scenario] | None = None,
                      episodes: int = 3, seed: int = 1000) -> dict:
    scenarios = list(SCENARIOS.values()) if scenarios is None else list(scenarios)
    out = []
    for sc in scenarios:
        runs, summary = [], {}
        for name, policy in policies.items():
            eps = [run_episode(policy, sc, seed=seed + i) for i in range(episodes)]
            summary[name] = summarize(eps)
            runs += [export_episode(ep, sc, name) for ep in eps]
        out.append({
            "name": sc.name, "description": sc.description,
            "has_wind": bool(any(sc.wind) or sc.gust_std > 0),
            "impulse_step": sc.impulse_step, "summary": summary, "runs": runs,
        })
    return {
        "meta": {
            "dt": DT, "target_z": TARGET_Z, "arm": ARM, "success_radius": SUCCESS_RADIUS,
            "bounds": {"x": 2.0, "zmin": 0.0, "zmax": 2.5},
            "policies": list(policies), "episodes": episodes,
            "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "scenarios": out,
    }


def write_viewer(data: dict, path: str | Path) -> Path:
    return write_page(TEMPLATE, data, path)
