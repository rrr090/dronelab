"""2D planar quadrotor hover environment (numpy only).

State:  x, z, theta, vx, vz, omega   (position, pitch, velocities, pitch rate)
Action: two motor thrusts in [-1, 1], mapped to [0, MAX_THRUST]
Goal:   hover at (0, TARGET_Z) while staying level.

This is deliberately simple: the point of step 1 is to get a trained MLP
policy that we can later open up with mechanistic interpretability tools.
"""
from __future__ import annotations

import numpy as np

GRAVITY = 9.81
MASS = 0.5
ARM = 0.15          # distance from centre of mass to each motor [m]
INERTIA = 0.02      # moment of inertia about the pitch axis [kg m^2]
MAX_THRUST = 6.0    # per-motor max thrust [N]
DT = 0.02           # integration step [s]
TARGET_Z = 1.0
MAX_STEPS = 500
OBS_DIM = 6
ACT_DIM = 2


class PlanarQuadrotorEnv:
    def __init__(self, seed: int | None = None, init_noise: float = 0.3):
        self.rng = np.random.default_rng(seed)
        self.init_noise = init_noise
        self.state = np.zeros(6, dtype=np.float64)
        self.t = 0
        # Disturbance hooks used by dronelab.hover.testing. Defaults leave the
        # training dynamics untouched.
        self.ext_force = np.zeros(2)          # external (fx, fz) [N], e.g. wind
        self.ext_torque = 0.0                 # external pitch torque [N m]
        self.motor_gain = np.ones(ACT_DIM)    # per-motor thrust efficiency

    def reset(self) -> np.ndarray:
        n = self.init_noise
        self.state = np.array(
            [
                self.rng.uniform(-n, n),                 # x
                TARGET_Z + self.rng.uniform(-n, n),      # z
                self.rng.uniform(-n, n) * 0.7,           # theta (scaled by init_noise)
                0.0, 0.0, 0.0,                           # vx, vz, omega
            ],
            dtype=np.float64,
        )
        self.t = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        x, z, th, vx, vz, om = self.state
        # Roughly unit-scale observation to keep MLP inputs well conditioned.
        return np.array(
            [x, z - TARGET_Z, th, vx / 2.0, vz / 2.0, om / 2.0], dtype=np.float32
        )

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        thrusts = (a + 1.0) / 2.0 * MAX_THRUST * self.motor_gain
        total = thrusts.sum()
        torque = (thrusts[1] - thrusts[0]) * ARM + self.ext_torque

        x, z, th, vx, vz, om = self.state
        ax = (-total * np.sin(th) + self.ext_force[0]) / MASS
        az = (total * np.cos(th) + self.ext_force[1]) / MASS - GRAVITY
        alpha = torque / INERTIA

        vx += ax * DT
        vz += az * DT
        om += alpha * DT
        x += vx * DT
        z += vz * DT
        th += om * DT
        self.state = np.array([x, z, th, vx, vz, om])
        self.t += 1

        pos_err = x * x + (z - TARGET_Z) ** 2
        vel_err = vx * vx + vz * vz
        reward = 1.0 - pos_err - 0.1 * vel_err - 0.5 * th * th - 0.01 * om * om
        fell = (
            abs(x) > 2.0 or z < 0.0 or z > 2.5 or abs(th) > np.pi / 3
        )
        terminated = bool(fell)
        truncated = self.t >= MAX_STEPS
        if terminated:
            reward -= 5.0
        return self.observe(), float(reward), terminated, truncated, {}
