"""Export cargo-task episodes to the interactive 3D viewer (viewer.html)."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from dronelab.cargo.env import (
    ARM, CARGO_MASS, DT, MOTOR_MAX, MOTOR_XY, ROPE_K, TARGET,
)
from dronelab.viz import P, quat_from_matrix, to_three, write_page
from dronelab.cargo.testing import (
    CARGO_SCENARIOS, SUCCESS_RADIUS, CargoEpisode, CargoScenario, Policy,
    run_cargo_episode, summarize_cargo,
)

TEMPLATE = Path(__file__).with_name("viewer.html")


def _r(a, nd=3):
    return np.round(np.asarray(a, dtype=np.float64).reshape(-1), nd).tolist()


def export_cargo_episode(ep: CargoEpisode, policy_name: str) -> dict:
    Rt = P @ ep.drone_rot @ P.T
    err = np.linalg.norm(ep.cargo_pos - TARGET, axis=-1)
    return {
        "policy": policy_name, "seed": ep.seed, "crashed": ep.crashed, "reason": ep.crash_reason,
        "success": ep.success, "steps": ep.steps, "final_err": round(ep.final_cargo_err, 4),
        "n": len(ep.cargo_pos),
        "dp": _r(to_three(ep.drone_pos)), "dq": _r(quat_from_matrix(Rt), 4),
        "cp": _r(to_three(ep.cargo_pos)), "ten": _r(ep.tension, 2),
        "thr": _r(ep.thrust / MOTOR_MAX, 2), "cf": _r(to_three(ep.cargo_force), 2),
        "err": _r(err),
    }


def build_viewer_data(policies: Mapping[str, Policy],
                            scenarios: Sequence[CargoScenario] | None = None,
                            episodes: int = 2, seed: int = 1000) -> dict:
    scenarios = list(CARGO_SCENARIOS.values()) if scenarios is None else list(scenarios)
    out = []
    for sc in scenarios:
        runs, summary = [], {}
        for name, pol in policies.items():
            eps = [run_cargo_episode(pol, sc, seed=seed + i) for i in range(episodes)]
            summary[name] = summarize_cargo(eps)
            runs += [export_cargo_episode(e, name) for e in eps]
        c = sc.conditions
        out.append({
            "name": sc.name, "description": sc.description,
            "has_wind": bool(any(c.wind) or c.gust_std > 0), "kick_step": c.kick_step,
            "cargo_mass": c.cargo_mass, "rope_len": c.rope_len,
            "summary": summary, "runs": runs,
        })
    motors = [to_three(np.array([x, y, 0.0])).tolist() for x, y in MOTOR_XY]
    return {
        "meta": {
            "dt": DT, "target": to_three(TARGET).tolist(), "arm": ARM, "motors": motors,
            "nominal_mass": CARGO_MASS, "success_radius": SUCCESS_RADIUS,
            "nominal_tension": CARGO_MASS * 9.81 / 3 * 1.2, "rope_k": ROPE_K,
            "policies": list(policies), "episodes": episodes,
            "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "scenarios": out,
    }


def write_viewer(data: dict, path: str | Path) -> Path:
    return write_page(TEMPLATE, data, path)
