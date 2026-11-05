"""Export mission runs to the interactive 3D viewer (viewer.html).

Map geometry is exported in sim coordinates (z-up) and converted in the
browser; trajectories are converted here (see dronelab.viz.to_three).
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from dronelab.cargo.env import ARM, CARGO_MASS, DT, MOTOR_MAX, MOTOR_XY
from dronelab.viz import P, quat_from_matrix, to_three, write_page
from dronelab.mission.maps import MissionMap
from dronelab.mission.testing import MissionResult, Policy, run_missions, summarize_missions

TEMPLATE = Path(__file__).with_name("viewer.html")


def _r(a, nd=3):
    return np.round(np.asarray(a, dtype=np.float64).reshape(-1), nd).tolist()


def export_run(res: MissionResult, policy_name: str, stride: int = 2) -> dict:
    rc = res.rec
    k = slice(None, None, stride)
    Rt = P @ rc["dr"][k] @ P.T
    hits = (np.nonzero(rc["hit"])[0] // stride).tolist()
    return {
        "policy": policy_name, "delivered": res.delivered, "reason": res.fail_reason,
        "health": round(res.health, 1), "score": round(res.score, 1), "time": round(res.time_s, 1),
        "damage": {kk: round(v, 1) for kk, v in res.damage.items()},
        "n": len(rc["cp"][k]),
        "dp": _r(to_three(rc["dp"][k])), "dq": _r(quat_from_matrix(Rt), 4),
        "cp": _r(to_three(rc["cp"][k])), "ref": _r(to_three(rc["ref"][k]), 2),
        "ten": _r(rc["ten"][k], 2), "hp": _r(rc["hp"][k], 1),
        "thr": _r(rc["thr"][k] / MOTOR_MAX, 2), "cf": _r(to_three(rc["cf"][k]), 2),
        "hits": hits,
    }


def build_viewer_data(policies: Mapping[str, Policy], maps: Sequence[MissionMap],
                              stride: int = 2, cruise: float = 1.0) -> dict:
    per_policy = {name: run_missions(pol, maps, cruise=cruise, record=True) for name, pol in policies.items()}
    out = []
    for i, m in enumerate(maps):
        runs = [export_run(per_policy[name][i], name, stride) for name in policies]
        out.append({"map": m.to_dict(), "runs": runs})
    summary = {name: summarize_missions(rs) for name, rs in per_policy.items()}
    motors = [to_three(np.array([x, y, 0.0])).tolist() for x, y in MOTOR_XY]
    return {
        "meta": {
            "dt": DT * stride, "arm": ARM, "motors": motors, "nominal_mass": CARGO_MASS,
            "nominal_tension": CARGO_MASS * 9.81 / 3 * 1.2, "policies": list(policies),
            "summary": summary, "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "missions": out,
    }


def write_viewer(data: dict, path: str | Path) -> Path:
    return write_page(TEMPLATE, data, path)
