"""Testing environment for the 3-drone cargo task.

A policy is any callable ``obs (63,) -> action (12,)``: the trained network
(``load_cargo_policy``), the hand-written ``FormationController``, or later an
ablated network.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from dronelab.cargo.env import (
    ACT_DIM, ACT_PER_DRONE, CARGO_MASS, CRASH_REASONS, DRONE_MASS, DT, F_HOVER_NOM, G,
    MAX_STEPS, MAX_TILT_CMD, MAX_YAW_RATE, N_DRONES, OBS_PER_DRONE, ROPE_LEN, TARGET,
    CargoEnv, Conditions, formation_offsets,
)

Policy = Callable[[np.ndarray], np.ndarray]
SUCCESS_RADIUS = 0.2
SETTLE_RADIUS = 0.15


@dataclass(frozen=True)
class CargoScenario:
    name: str
    description: str
    conditions: Conditions = Conditions()
    steps: int = MAX_STEPS
    action_delay: int = 0


def _fault():
    g = np.ones((3, 4)); g[0, 0] = 0.7
    return tuple(map(tuple, g))


CARGO_SCENARIOS: dict[str, CargoScenario] = {s.name: s for s in [
    CargoScenario("nominal", "calm air, small start offset"),
    CargoScenario("calm_start", "perfect start, no disturbance", Conditions(init_noise=0.0)),
    CargoScenario("wide_start", "cargo starts up to 0.5 m off target", Conditions(init_noise=0.5)),
    CargoScenario("gusts", "random gusts, 0.5 N std", Conditions(gust_std=0.5)),
    CargoScenario("storm", "strong gusts 1.0 N std + 0.5 N wind",
                  Conditions(gust_std=1.0, wind=(0.5, 0.0, 0.0))),
    CargoScenario("wind", "steady 0.8 N side wind", Conditions(wind=(0.8, 0.0, 0.0))),
    CargoScenario("heavy_cargo", "0.9 kg cargo (nominal 0.6)", Conditions(cargo_mass=0.9)),
    CargoScenario("light_cargo", "0.3 kg cargo", Conditions(cargo_mass=0.3)),
    CargoScenario("long_ropes", "1.15 m ropes (nominal 1.0)", Conditions(rope_len=1.15)),
    CargoScenario("motor_fault", "drone 1, one motor at 70%", Conditions(motor_gain=_fault())),
    CargoScenario("cargo_kick", "cargo knocked sideways at t=2 s, 1.5 m/s",
                  Conditions(kick_step=100, kick_vel=(1.5, 0.5, 0.0))),
    CargoScenario("sensor_noise", "observation noise std 0.03", Conditions(obs_noise=0.03)),
    CargoScenario("latency", "2-step (40 ms) command delay", action_delay=2),
    CargoScenario("long_hold", "4x episode length with gusts", Conditions(gust_std=0.5),
                  steps=4 * MAX_STEPS),
]}


@dataclass
class CargoEpisode:
    scenario: str
    seed: int
    crashed: bool
    crash_reason: str
    steps: int
    total_reward: float
    mean_cargo_err: float
    final_cargo_err: float
    max_swing_deg: float
    max_tilt_deg: float
    settle_time: float | None
    success: bool
    conditions: Conditions = field(repr=False)
    drone_pos: np.ndarray = field(repr=False)    # (T+1, 3, 3)
    drone_rot: np.ndarray = field(repr=False)    # (T+1, 3, 3, 3)
    cargo_pos: np.ndarray = field(repr=False)    # (T+1, 3)
    tension: np.ndarray = field(repr=False)      # (T+1, 3)
    thrust: np.ndarray = field(repr=False)       # (T, 3, 4) motor thrusts [N]
    cargo_force: np.ndarray = field(repr=False)  # (T, 3) wind/gust force on cargo
    drone_force: np.ndarray = field(repr=False)  # (T, 3, 3)
    kick_step: int | None = None


def run_cargo_episode(policy: Policy, scenario: CargoScenario, seed: int = 0) -> CargoEpisode:
    env = CargoEnv(1, seed=seed, conditions=scenario.conditions, max_steps=scenario.steps)
    obs = env.reset()[0]
    pending = deque([np.zeros(ACT_DIM)] * scenario.action_delay)
    rec = {k: [] for k in ["dp", "dr", "cp", "ten", "thr", "cf", "df"]}

    def snap():
        rec["dp"].append(env.p[0].copy()); rec["dr"].append(env.R[0].copy())
        rec["cp"].append(env.cp[0].copy()); rec["ten"].append(env.tension[0].copy())

    snap()
    total, crash = 0.0, 0
    for _ in range(scenario.steps):
        pending.append(np.asarray(policy(obs), dtype=np.float64).reshape(ACT_DIM))
        o, r, term, _, info = env.step(pending.popleft()[None])
        obs = o[0]
        total += float(r[0])
        snap()
        rec["thr"].append(env.thrust[0].copy()); rec["cf"].append(env.cargo_force[0].copy())
        rec["df"].append(env.drone_force[0].copy())
        if term[0]:
            crash = int(info["crash_code"][0])
            break

    dp, cp = np.array(rec["dp"]), np.array(rec["cp"])
    err = np.linalg.norm(cp - TARGET, axis=-1)
    hang = dp.mean(1) - cp                       # cargo -> drone centroid
    swing = np.degrees(np.arccos(np.clip(hang[:, 2] / np.linalg.norm(hang, axis=-1), -1, 1)))
    R = np.array(rec["dr"])
    tilt = np.degrees(np.arccos(np.clip(R[..., 2, 2], -1, 1)))
    outside = np.nonzero(err > SETTLE_RADIUS)[0]
    if crash:
        settle = None
    elif len(outside) == 0:
        settle = 0.0
    elif outside[-1] < len(err) - 1:
        settle = (outside[-1] + 1) * DT
    else:
        settle = None
    ks = scenario.conditions.kick_step
    return CargoEpisode(
        scenario=scenario.name, seed=seed, crashed=bool(crash), crash_reason=CRASH_REASONS[crash],
        steps=len(rec["thr"]), total_reward=total, mean_cargo_err=float(err.mean()),
        final_cargo_err=float(err[-1]), max_swing_deg=float(swing.max()),
        max_tilt_deg=float(tilt.max()), settle_time=settle,
        success=bool(not crash and err[-1] < SUCCESS_RADIUS), conditions=scenario.conditions,
        drone_pos=dp, drone_rot=R, cargo_pos=cp, tension=np.array(rec["ten"]),
        thrust=np.array(rec["thr"]).reshape(-1, 3, 4), cargo_force=np.array(rec["cf"]).reshape(-1, 3),
        drone_force=np.array(rec["df"]).reshape(-1, 3, 3), kick_step=ks,
    )


def summarize_cargo(eps: Sequence[CargoEpisode]) -> dict:
    settles = [e.settle_time for e in eps if e.settle_time is not None]
    reasons = {}
    for e in eps:
        if e.crashed:
            reasons[e.crash_reason] = reasons.get(e.crash_reason, 0) + 1
    return {
        "episodes": len(eps),
        "success_rate": float(np.mean([e.success for e in eps])),
        "crash_rate": float(np.mean([e.crashed for e in eps])),
        "mean_cargo_err": float(np.mean([e.mean_cargo_err for e in eps])),
        "final_cargo_err": float(np.mean([e.final_cargo_err for e in eps])),
        "max_swing_deg": float(np.max([e.max_swing_deg for e in eps])),
        "max_tilt_deg": float(np.max([e.max_tilt_deg for e in eps])),
        "settle_time": float(np.mean(settles)) if settles else float("nan"),
        "mean_reward": float(np.mean([e.total_reward for e in eps])),
        "crash_reasons": reasons,
    }


def run_cargo_suite(policy: Policy, scenarios: Sequence[CargoScenario] | None = None,
                    episodes: int = 10, seed: int = 1000):
    scenarios = list(CARGO_SCENARIOS.values()) if scenarios is None else list(scenarios)
    summaries, results = {}, {}
    for sc in scenarios:
        eps = [run_cargo_episode(policy, sc, seed=seed + i) for i in range(episodes)]
        results[sc.name], summaries[sc.name] = eps, summarize_cargo(eps)
    return summaries, results


def format_cargo_table(summaries: dict) -> str:
    head = (f"{'scenario':<14}{'success':>8}{'crash':>7}{'mean err':>10}{'final err':>11}"
            f"{'swing deg':>11}{'tilt deg':>10}{'settle s':>10}  crashes")
    lines = [head, "-" * (len(head) + 10)]
    for name, s in summaries.items():
        why = ", ".join(f"{k} x{v}" for k, v in s["crash_reasons"].items())
        lines.append(
            f"{name:<14}{s['success_rate']:>8.0%}{s['crash_rate']:>7.0%}{s['mean_cargo_err']:>10.3f}"
            f"{s['final_cargo_err']:>11.3f}{s['max_swing_deg']:>11.1f}{s['max_tilt_deg']:>10.1f}"
            f"{s['settle_time']:>10.2f}  {why}")
    return "\n".join(lines)


# ------------------------------------------------------------------ policies

class FormationController:
    """Hand-written baseline for the cargo task, reading the same observation.

    Each drone flies to its formation slot above the cargo target with a PD
    position loop, adds the measured rope tension as feedforward, and shifts
    the whole formation against the cargo's error and velocity to damp swing.
    It has no integral term and knows nothing about the disturbances.
    """

    def __init__(self, kp=5.0, kd=3.5, k_cargo=0.8, k_swing=0.6, kz_cargo=1.0):
        self.kp, self.kd = kp, kd
        self.k_cargo, self.k_swing, self.kz_cargo = k_cargo, k_swing, kz_cargo
        self.slots = formation_offsets(ROPE_LEN, CARGO_MASS)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._act(obs, np.zeros(3))

    def _act(self, obs: np.ndarray, v_ref: np.ndarray) -> np.ndarray:
        """v_ref: velocity of the (possibly moving) cargo target, used as feedforward."""
        e_c, cv = obs[0:3], 2 * obs[3:6] - v_ref
        d = obs[6:].reshape(N_DRONES, OBS_PER_DRONE)
        rel, v, R, ten = d[:, 0:3], 2 * d[:, 3:6], d[:, 6:15].reshape(3, 3, 3), 10 * d[:, 18]
        q = rel + e_c                                   # drone position relative to target
        shift = np.array([1.0, 1.0, 0.0]) * (-self.k_cargo * e_c - self.k_swing * cv)
        shift[2] = -self.kz_cargo * e_c[2]
        q_des = self.slots + shift
        acc = self.kp * (q_des - q) - self.kd * (v - v_ref)
        u = rel / np.linalg.norm(rel, axis=1, keepdims=True)
        F = DRONE_MASS * (acc + np.array([0, 0, G])) + np.maximum(ten, 0)[:, None] * u
        out = np.zeros((N_DRONES, ACT_PER_DRONE))
        for i in range(N_DRONES):
            psi = np.arctan2(R[i, 1, 0], R[i, 0, 0])
            c, s = np.cos(psi), np.sin(psi)
            fx, fy = c * F[i, 0] + s * F[i, 1], -s * F[i, 0] + c * F[i, 1]
            n = np.linalg.norm(F[i])
            roll = np.arcsin(np.clip(-fy / n, -1, 1))
            pitch = np.arctan2(fx, F[i, 2])
            out[i] = [F[i] @ R[i, :, 2] / F_HOVER_NOM - 1.0, roll / MAX_TILT_CMD,
                      pitch / MAX_TILT_CMD, -2.0 * psi / MAX_YAW_RATE]
        return np.clip(out.reshape(-1), -1, 1)


def load_cargo_policy(path: str, deterministic: bool = True, device: str = "cpu") -> Policy:
    import torch
    from dronelab.policy import ActorCritic

    ckpt = torch.load(path, map_location=device)
    model = ActorCritic(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    @torch.no_grad()
    def policy(obs: np.ndarray) -> np.ndarray:
        mean, std, _ = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        a = mean if deterministic else torch.normal(mean, std)
        return a.squeeze(0).numpy()

    policy.model = model
    return policy
