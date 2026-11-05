"""Drone testing environment: stress-test a hover policy under scenarios.

A *policy* is any callable ``obs (np.ndarray, shape (6,)) -> action (shape (2,))``.
That covers the trained PPO network (see ``load_ppo_policy``), the reference
``PDController`` below, and later on, ablated networks for step 4.

Each ``Scenario`` describes one test condition:

* physical disturbances applied inside the env: steady wind, wind gusts,
  impulse kicks, motor degradation;
* sensing/actuation problems applied between env and policy: observation
  noise, observation bias, actuation latency;
* the initial-condition spread (``init_noise``) and episode length.

``run_suite`` runs every scenario for N seeded episodes and returns per-episode
``EpisodeResult``s plus an aggregate summary per scenario.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from dronelab.hover.env import (
    ACT_DIM, ARM, DT, GRAVITY, INERTIA, MASS, MAX_STEPS, MAX_THRUST, TARGET_Z,
    PlanarQuadrotorEnv,
)

Policy = Callable[[np.ndarray], np.ndarray]

# An episode counts as a "hover success" if it survives and ends within this
# distance of the target.
SUCCESS_RADIUS = 0.15
SETTLE_RADIUS = 0.1


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str = ""
    init_noise: float = 0.3
    steps: int = MAX_STEPS
    wind: tuple[float, float] = (0.0, 0.0)          # steady (fx, fz) [N]
    gust_std: float = 0.0                           # random gust force std [N]
    gust_tau: float = 0.5                           # gust correlation time [s]
    impulse_step: int | None = None                 # step at which a kick happens
    impulse_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)  # dvx, dvz, domega
    motor_gain: tuple[float, float] = (1.0, 1.0)    # per-motor efficiency
    obs_noise_std: float = 0.0                      # gaussian noise on observations
    obs_bias: tuple[float, ...] = (0.0,) * 6        # constant sensor bias
    action_delay: int = 0                           # actuation latency [steps]


SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    Scenario("nominal", "training distribution"),
    Scenario("level_start", "zero initial noise, should be trivial", init_noise=0.0),
    Scenario("wide_start", "2x the training initial-state spread", init_noise=0.6),
    Scenario("wind", "steady 0.5 N sideways wind", wind=(0.5, 0.0)),
    Scenario("gusts", "random correlated gusts, 0.4 N std", gust_std=0.4),
    Scenario("impulse", "sideways + spin kick at t=2 s", impulse_step=100,
             impulse_velocity=(1.0, 0.0, 2.0)),
    Scenario("motor_fault", "left motor at 85% efficiency", motor_gain=(0.85, 1.0)),
    Scenario("heavy", "both motors at 85% (like +18% payload)", motor_gain=(0.85, 0.85)),
    Scenario("sensor_noise", "gaussian obs noise, std 0.05", obs_noise_std=0.05),
    Scenario("sensor_bias", "height sensor reads 0.1 m high",
             obs_bias=(0.0, 0.1, 0.0, 0.0, 0.0, 0.0)),
    Scenario("latency", "3-step (60 ms) actuation delay", action_delay=3),
    Scenario("long_hover", "4x episode length", steps=4 * MAX_STEPS),
]}


@dataclass
class EpisodeResult:
    scenario: str
    seed: int
    crashed: bool
    steps: int
    total_reward: float
    mean_pos_err: float
    final_pos_err: float
    max_tilt_deg: float
    settle_time: float | None      # seconds until it stays within SETTLE_RADIUS
    success: bool
    trajectory: np.ndarray = field(repr=False)   # (T+1, 6) true states
    actions: np.ndarray = field(repr=False)      # (T, 2) applied actions
    forces: np.ndarray = field(repr=False)       # (T, 2) external force (fx, fz)


def run_episode(policy: Policy, scenario: Scenario, seed: int = 0) -> EpisodeResult:
    rng = np.random.default_rng(seed)
    env = PlanarQuadrotorEnv(seed=seed, init_noise=scenario.init_noise)
    env.motor_gain = np.asarray(scenario.motor_gain, dtype=np.float64)
    wind = np.asarray(scenario.wind, dtype=np.float64)
    bias = np.asarray(scenario.obs_bias, dtype=np.float32)

    obs = env.reset()
    gust = np.zeros(2)
    # Ornstein-Uhlenbeck gust process with stationary std = gust_std.
    decay = np.exp(-DT / scenario.gust_tau)
    gust_kick = scenario.gust_std * np.sqrt(1.0 - decay ** 2)
    hover = np.full(ACT_DIM, 2 * (GRAVITY * MASS / 2) / MAX_THRUST - 1.0)
    pending = deque([hover.copy() for _ in range(scenario.action_delay)])

    states, actions, forces = [env.state.copy()], [], []
    total_reward, crashed = 0.0, False
    for t in range(scenario.steps):
        if scenario.impulse_step is not None and t == scenario.impulse_step:
            env.state[3:6] += np.asarray(scenario.impulse_velocity)
            obs = env.observe()

        seen = obs + bias
        if scenario.obs_noise_std > 0:
            seen = seen + rng.normal(0, scenario.obs_noise_std, size=seen.shape)
        a = np.asarray(policy(seen.astype(np.float32)), dtype=np.float64).reshape(ACT_DIM)
        pending.append(a)
        applied = pending.popleft()

        if scenario.gust_std > 0:
            gust = decay * gust + gust_kick * rng.normal(size=2)
        env.ext_force = wind + gust

        # The env truncates at its own MAX_STEPS; we control length here.
        env.t = 0
        obs, r, term, _, _ = env.step(applied)
        total_reward += r
        states.append(env.state.copy())
        actions.append(np.clip(applied, -1, 1))
        forces.append(env.ext_force.copy())
        if term:
            crashed = True
            break

    traj = np.array(states)
    pos_err = np.hypot(traj[:, 0], traj[:, 1] - TARGET_Z)
    outside = np.nonzero(pos_err > SETTLE_RADIUS)[0]
    if crashed:
        settle = None
    elif len(outside) == 0:
        settle = 0.0
    elif outside[-1] < len(pos_err) - 1:
        settle = (outside[-1] + 1) * DT
    else:
        settle = None
    final = float(pos_err[-1])
    return EpisodeResult(
        scenario=scenario.name, seed=seed, crashed=crashed, steps=len(actions),
        total_reward=total_reward, mean_pos_err=float(pos_err.mean()),
        final_pos_err=final, max_tilt_deg=float(np.degrees(np.abs(traj[:, 2]).max())),
        settle_time=settle, success=(not crashed) and final < SUCCESS_RADIUS,
        trajectory=traj, actions=np.array(actions).reshape(-1, ACT_DIM),
        forces=np.array(forces).reshape(-1, 2),
    )


def summarize(results: Sequence[EpisodeResult]) -> dict:
    settles = [r.settle_time for r in results if r.settle_time is not None]
    return {
        "episodes": len(results),
        "success_rate": float(np.mean([r.success for r in results])),
        "crash_rate": float(np.mean([r.crashed for r in results])),
        "mean_pos_err": float(np.mean([r.mean_pos_err for r in results])),
        "final_pos_err": float(np.mean([r.final_pos_err for r in results])),
        "max_tilt_deg": float(np.max([r.max_tilt_deg for r in results])),
        "settle_time": float(np.mean(settles)) if settles else float("nan"),
        "mean_reward": float(np.mean([r.total_reward for r in results])),
    }


def run_suite(policy: Policy, scenarios: Sequence[Scenario] | None = None,
              episodes: int = 20, seed: int = 0):
    """Run each scenario for ``episodes`` seeds. Returns (summaries, results)."""
    scenarios = list(SCENARIOS.values()) if scenarios is None else list(scenarios)
    summaries, results = {}, {}
    for sc in scenarios:
        eps = [run_episode(policy, sc, seed=seed + i) for i in range(episodes)]
        results[sc.name] = eps
        summaries[sc.name] = summarize(eps)
    return summaries, results


def format_table(summaries: dict) -> str:
    head = (f"{'scenario':<14}{'success':>9}{'crash':>8}{'mean err':>10}"
            f"{'final err':>11}{'tilt deg':>10}{'settle s':>10}{'reward':>9}")
    lines = [head, "-" * len(head)]
    for name, s in summaries.items():
        lines.append(
            f"{name:<14}{s['success_rate']:>9.0%}{s['crash_rate']:>8.0%}"
            f"{s['mean_pos_err']:>10.3f}{s['final_pos_err']:>11.3f}"
            f"{s['max_tilt_deg']:>10.1f}{s['settle_time']:>10.2f}{s['mean_reward']:>9.1f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- policies

class PDController:
    """Hand-tuned cascaded PD hover controller, used as a reference baseline.

    Outer loop: position error -> desired thrust and pitch.
    Inner loop: pitch error -> differential thrust.
    It reads the same (scaled) observation the network sees.
    """

    def __init__(self, kp_z=6.0, kd_z=4.0, kp_x=0.6, kd_x=0.9,
                 kp_th=120.0, kd_th=22.0, max_tilt=0.4):
        self.kp_z, self.kd_z = kp_z, kd_z
        self.kp_x, self.kd_x = kp_x, kd_x
        self.kp_th, self.kd_th = kp_th, kd_th
        self.max_tilt = max_tilt

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x, ez, th, vx, vz, om = obs[0], obs[1], obs[2], 2 * obs[3], 2 * obs[4], 2 * obs[5]
        # ax = -T sin(th)/m, so tilting positive pushes towards -x.
        th_des = np.clip(self.kp_x * x + self.kd_x * vx, -self.max_tilt, self.max_tilt)
        total = MASS * (GRAVITY - self.kp_z * ez - self.kd_z * vz) / max(np.cos(th), 0.5)
        torque = INERTIA * (self.kp_th * (th_des - th) - self.kd_th * om)
        diff = torque / ARM
        thrusts = np.array([total - diff, total + diff]) / 2.0
        return np.clip(2.0 * thrusts / MAX_THRUST - 1.0, -1.0, 1.0)


def hover_policy(obs: np.ndarray) -> np.ndarray:
    """Open-loop constant hover thrust: a 'do nothing clever' baseline."""
    return np.full(ACT_DIM, 2 * (GRAVITY * MASS / 2) / MAX_THRUST - 1.0)


def load_ppo_policy(path: str, deterministic: bool = True, device: str = "cpu") -> Policy:
    """Wrap a trained ActorCritic checkpoint as a numpy policy callable."""
    import torch
    from dronelab.policy import ActorCritic

    model = ActorCritic()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    @torch.no_grad()
    def policy(obs: np.ndarray) -> np.ndarray:
        mean, std, _ = model(torch.as_tensor(obs).unsqueeze(0))
        a = mean if deterministic else torch.normal(mean, std)
        return a.squeeze(0).numpy()

    policy.model = model  # exposed so ablation hooks can be attached later
    return policy
