import numpy as np

from dronelab.hover.env import PlanarQuadrotorEnv
from dronelab.hover.testing import (
    SCENARIOS, PDController, Scenario, hover_policy, run_episode, run_suite,
)


def test_default_hooks_do_not_change_dynamics():
    # The disturbance hooks must be inert by default so training is unchanged.
    a, b = PlanarQuadrotorEnv(seed=3), PlanarQuadrotorEnv(seed=3)
    a.reset(), b.reset()
    b.ext_force = np.zeros(2)
    for _ in range(20):
        oa = a.step(np.array([0.1, -0.2]))[0]
        ob = b.step(np.array([0.1, -0.2]))[0]
    assert np.allclose(oa, ob)


def test_pd_controller_hovers_nominal():
    summaries, _ = run_suite(PDController(), [SCENARIOS["nominal"]], episodes=5)
    assert summaries["nominal"]["success_rate"] == 1.0


def test_open_loop_hover_crashes_from_offset():
    summaries, _ = run_suite(hover_policy, [SCENARIOS["nominal"]], episodes=5)
    assert summaries["nominal"]["crash_rate"] == 1.0


def test_episodes_are_deterministic_per_seed():
    sc = SCENARIOS["gusts"]
    r1 = run_episode(PDController(), sc, seed=7)
    r2 = run_episode(PDController(), sc, seed=7)
    assert np.array_equal(r1.trajectory, r2.trajectory)


def test_wind_pushes_drone_downwind():
    sc = Scenario("w", init_noise=0.0, steps=50, wind=(1.0, 0.0))
    r = run_episode(hover_policy, sc)
    assert r.trajectory[-1, 0] > 0.05


def test_long_scenario_runs_past_env_max_steps():
    r = run_episode(PDController(), SCENARIOS["long_hover"], seed=0)
    assert r.steps == SCENARIOS["long_hover"].steps and not r.crashed


def test_action_delay_holds_hover_thrust_first():
    sc = Scenario("d", init_noise=0.0, steps=5, action_delay=3)
    r = run_episode(lambda obs: np.ones(2), sc)
    assert np.allclose(r.actions[:3], r.actions[0])
    assert np.allclose(r.actions[3:], 1.0)
