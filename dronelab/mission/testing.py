"""Evaluate cargo delivery missions on generated maps.

Missions run batched (one env per map), so a whole suite flies in parallel.
A policy is a callable ``obs (93,) -> action (12,)``; if it has a ``batched``
attribute (``(B, 93) -> (B, 12)``), that is used instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from dronelab.cargo.env import DT, OBS_DIM
from dronelab.cargo.testing import FormationController
from dronelab.mission.maps import THEMES, MissionMap, generate_map
from dronelab.mission.env import (
    DAMAGE_KINDS, FAIL_REASONS, MISSION_OBS_DIM, MissionEnv, baseline_actions, compose_residual,
)

Policy = Callable[[np.ndarray], np.ndarray]
EVAL_SEED = 5_000_000            # training maps use seeds below 1e6


@dataclass
class MissionResult:
    map: MissionMap = field(repr=False)
    delivered: bool
    fail_reason: str               # "" if delivered, "timeout" if it ran out of time
    health: float
    score: float                   # health if delivered, else 0
    time_s: float
    damage: dict
    mean_track_err: float
    min_clearance: float
    steps: int = 0
    rec: dict | None = field(default=None, repr=False)


def mission_maps(themes=None, difficulties=(0.5, 1.0), per=2, seed=EVAL_SEED):
    """Unseen evaluation maps: every theme x difficulty, ``per`` maps each."""
    out = []
    for ti, th in enumerate(themes or THEMES):
        for di, d in enumerate(difficulties):
            for k in range(per):
                out.append(generate_map(seed + 1000 * ti + 100 * di + k, th, d))
    return out


def _batched(policy):
    if hasattr(policy, "batched"):
        return policy.batched
    return lambda O: np.stack([policy(o) for o in O])


def run_missions(policy: Policy, maps: Sequence[MissionMap], cruise: float = 1.0,
                 record: bool = False, seed: int = 0, realism=None) -> list[MissionResult]:
    B = len(maps)
    env = MissionEnv(B, seed=seed, maps=list(maps), cruise=cruise, history=getattr(policy, "history", False),
                     realism=realism)
    obs = env.reset()
    act = _batched(policy)
    alive = np.ones(B, dtype=bool)
    result = [None] * B
    err_sum, min_clear, steps = np.zeros(B), np.full(B, np.inf), np.zeros(B, dtype=np.int64)
    keys = ["dp", "dr", "cp", "ten", "thr", "cf", "hp", "ref", "hit", "dmg"]
    rec = {k: [] for k in keys} if record else None

    def snap():
        rec["dp"].append(env.p.copy()); rec["dr"].append(env.R.copy()); rec["cp"].append(env.cp.copy())
        rec["ten"].append(env.tension.copy()); rec["hp"].append(env.health.copy())
        rec["ref"].append(env.target.copy())

    if record:
        snap()
    while alive.any():
        obs, r, term, trunc, info = env.step(act(obs))
        dist = env._cache[0]
        err_sum += env.cargo_error() * alive
        min_clear = np.where(alive, np.minimum(min_clear, dist.min(-1)), min_clear)
        steps += alive
        if record:
            snap()
            rec["thr"].append(env.thrust.copy()); rec["cf"].append(env.cargo_force.copy())
            rec["hit"].append(env.impact_event.copy()); rec["dmg"].append(env.step_damage.copy())
        for i in np.nonzero(alive & (term | trunc))[0]:
            alive[i] = False
            delivered = bool(info["delivered"][i])
            reason = "" if delivered else (FAIL_REASONS[info["fail"][i]] if term[i] else "timeout")
            hp = float(env.health[i])
            result[i] = MissionResult(
                map=maps[i], delivered=delivered, fail_reason=reason, health=hp,
                score=hp if delivered else 0.0, time_s=float(steps[i] * DT),
                damage={k: float(v) for k, v in zip(DAMAGE_KINDS, env.damage[i])},
                mean_track_err=float(err_sum[i] / steps[i]), min_clearance=float(min_clear[i]),
                steps=int(steps[i]),
            )
    if record:
        stacked = {k: np.array(v) for k, v in rec.items()}
        for i, res in enumerate(result):
            n = int(steps[i])
            res.rec = {k: (v[: n + 1, i] if k in ("dp", "dr", "cp", "ten", "hp", "ref") else v[:n, i])
                       for k, v in stacked.items()}
    return result


def summarize_missions(results: Sequence[MissionResult]) -> dict:
    fails = {}
    for r in results:
        if not r.delivered:
            fails[r.fail_reason] = fails.get(r.fail_reason, 0) + 1
    dl = [r for r in results if r.delivered]
    return {
        "missions": len(results),
        "delivered_rate": float(np.mean([r.delivered for r in results])),
        "mean_score": float(np.mean([r.score for r in results])),
        "health_if_delivered": float(np.mean([r.health for r in dl])) if dl else float("nan"),
        "mean_time_s": float(np.mean([r.time_s for r in dl])) if dl else float("nan"),
        "mean_track_err": float(np.mean([r.mean_track_err for r in results])),
        "damage": {k: float(np.mean([r.damage[k] for r in results])) for k in DAMAGE_KINDS},
        "failures": fails,
    }


def format_mission_table(results: Sequence[MissionResult]) -> str:
    groups = {}
    for r in results:
        groups.setdefault(r.map.theme, []).append(r)
    groups["ALL"] = list(results)
    head = (f"{'theme':<8}{'delivered':>10}{'score':>8}{'health':>8}{'time s':>8}{'track':>7}"
            f"{'jerk':>7}{'g':>6}{'impact':>8}{'rub':>6}  failures")
    lines = [head, "-" * (len(head) + 12)]
    for name, rs in groups.items():
        s = summarize_missions(rs)
        d = s["damage"]
        why = ", ".join(f"{k} x{v}" for k, v in s["failures"].items())
        if name == "ALL":
            lines.append("-" * (len(head) + 12))
        lines.append(
            f"{name:<8}{s['delivered_rate']:>10.0%}{s['mean_score']:>8.1f}{s['health_if_delivered']:>8.1f}"
            f"{s['mean_time_s']:>8.1f}{s['mean_track_err']:>7.2f}{d['rope jerk']:>7.1f}{d['g-force']:>6.1f}"
            f"{d['impact']:>8.1f}{d['rope rub']:>6.1f}  {why}")
    return "\n".join(lines)


# ---------------------------------------------------------------- comparison

def bootstrap_ci(values, n_boot: int = 10_000, level: float = 0.95, seed: int = 0):
    """Mean of ``values`` and its bootstrap confidence interval: (mean, low, high)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(values.mean()), float(low), float(high)


def compare_to_baseline(runs: dict[str, Sequence[MissionResult]],
                        baseline: Sequence[MissionResult]) -> dict:
    """Compare trained policies (one per training seed) with the baseline, map by map.

    All results must come from the same maps, so each map is a paired comparison.
    Seeds are averaged per map before bootstrapping over maps. The interval
    therefore covers which maps were drawn, not training luck, so the verdict
    also requires every seed to land on the same side of the baseline.
    """
    if any(len(rs) != len(baseline) for rs in runs.values()):
        raise ValueError("every policy must be scored on the same maps")

    def paired(metric):
        base = np.array([metric(r) for r in baseline], dtype=float)
        per_seed = np.array([[metric(r) for r in rs] for rs in runs.values()], dtype=float)
        mean, low, high = bootstrap_ci(per_seed.mean(axis=0) - base)
        return {"baseline": float(base.mean()), "trained": float(per_seed.mean()),
                "diff": mean, "ci_low": low, "ci_high": high,
                "seed_diffs": dict(zip(runs, (per_seed - base).mean(axis=1).tolist()))}

    score = paired(lambda r: r.score)
    diffs = list(score["seed_diffs"].values())
    if score["ci_low"] > 0 and min(diffs) > 0:
        verdict = "trained policy beats the baseline"
    elif score["ci_high"] < 0 and max(diffs) < 0:
        verdict = "baseline beats the trained policy"
    else:
        verdict = "no clear winner"
    return {"maps": len(baseline), "seeds": len(runs), "score": score,
            "delivered": paired(lambda r: float(r.delivered)), "verdict": verdict}


def format_comparison(c: dict) -> str:
    s, d = c["score"], c["delivered"]
    lines = [f"trained ({c['seeds']} seed{'s' if c['seeds'] != 1 else ''}) vs baseline, "
             f"paired on {c['maps']} maps, 95% CI:",
             f"  score      {s['trained']:6.1f} vs {s['baseline']:6.1f}   "
             f"diff {s['diff']:+6.1f}  [{s['ci_low']:+.1f}, {s['ci_high']:+.1f}]",
             f"  delivered  {d['trained']:6.1%} vs {d['baseline']:6.1%}   "
             f"diff {d['diff']:+6.1%}  [{d['ci_low']:+.1%}, {d['ci_high']:+.1%}]"]
    if c["seeds"] > 1:
        lines.append("  score diff by seed: "
                     + ", ".join(f"{k} {v:+.1f}" for k, v in s["seed_diffs"].items()))
    lines.append(f"verdict: {c['verdict']}")
    return "\n".join(lines)


# ------------------------------------------------------------------ policies

class MissionFormationController(FormationController):
    """The formation baseline, following the moving route reference with velocity feedforward.
    It does not look at obstacles; it relies on the planned route's clearance."""

    def __call__(self, obs):
        return self._act(obs[:OBS_DIM], 2 * obs[OBS_DIM:OBS_DIM + 3])

    def batched(self, O):
        return baseline_actions(np.asarray(O))


def load_mission_policy(path: str, deterministic: bool = True, device: str = "cpu") -> Policy:
    import torch
    from dronelab.policy import ActorCritic

    ckpt = torch.load(path, map_location=device)
    model = ActorCritic(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    residual = float(ckpt.get("residual", 0.0))

    @torch.no_grad()
    def batched(O):
        O = np.asarray(O)
        mean, std, _ = model(torch.as_tensor(O, dtype=torch.float32))
        a = (mean if deterministic else torch.normal(mean, std)).numpy()
        return compose_residual(O, a, residual) if residual > 0 else a

    def policy(o):
        return batched(np.asarray(o)[None])[0]

    policy.batched, policy.model = batched, model
    policy.history = model.config["obs_dim"] > MISSION_OBS_DIM
    return policy
