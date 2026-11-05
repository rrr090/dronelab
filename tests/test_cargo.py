import json

import numpy as np

from dronelab.cargo.env import (
    ACT_DIM, CARGO_MASS, G, OBS_DIM, CargoEnv, Conditions, formation_offsets,
)
from dronelab.cargo.testing import CARGO_SCENARIOS, CargoScenario, FormationController, run_cargo_episode
from dronelab.cargo.viewer import build_viewer_data, quat_from_matrix, write_viewer


def test_shapes_and_batch():
    env = CargoEnv(5, seed=0, difficulty=1.0, autoreset=True)
    obs = env.reset()
    assert obs.shape == (5, OBS_DIM)
    obs, r, term, trunc, info = env.step(np.zeros((5, ACT_DIM)))
    assert obs.shape == (5, OBS_DIM) and r.shape == term.shape == (5,)


def test_ropes_carry_cargo_weight_at_rest():
    env = CargoEnv(1, seed=0, conditions=Conditions(init_noise=0.0))
    env.reset()
    d = env.p[0] - env.cp[0]
    vertical = np.sum(env.tension[0] * d[:, 2] / np.linalg.norm(d, axis=1))
    assert abs(vertical - CARGO_MASS * G) < 1e-3


def test_slack_rope_has_no_tension():
    env = CargoEnv(1, seed=0, conditions=Conditions(init_noise=0.0))
    env.reset()
    env.cp[0, 2] += 0.3          # lift cargo: ropes go slack
    env._ropes()
    assert np.all(env.tension == 0)


def test_no_control_eventually_crashes():
    env = CargoEnv(1, seed=0, conditions=Conditions(init_noise=0.0))
    env.reset()
    for _ in range(500):
        _, _, term, _, _ = env.step(np.zeros((1, ACT_DIM)))
        if term[0]:
            break
    assert term[0]


def test_formation_controller_holds_cargo():
    ep = run_cargo_episode(FormationController(), CARGO_SCENARIOS["nominal"], seed=3)
    assert ep.success and not ep.crashed


def test_cargo_kick_and_episode_determinism():
    sc = CargoScenario("k", "", Conditions(kick_step=5, kick_vel=(1.0, 0, 0), gust_std=0.3), steps=40)
    a = run_cargo_episode(FormationController(), sc, seed=1)
    b = run_cargo_episode(FormationController(), sc, seed=1)
    assert np.array_equal(a.cargo_pos, b.cargo_pos)
    assert a.cargo_pos[10, 0] - a.cargo_pos[4, 0] > 0.03


def test_quaternion_conversion_roundtrip():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(20, 3, 3))
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt * np.sign(np.linalg.det(U @ Vt))[:, None, None]
    q = quat_from_matrix(R)
    x, y, z, w = q.T
    back = np.stack([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ]).transpose(2, 0, 1)
    assert np.allclose(back, R, atol=1e-6)


def test_cargo_viewer_export(tmp_path):
    sc = CargoScenario("short", "", Conditions(gust_std=0.3), steps=20)
    data = build_viewer_data({"FORMATION": FormationController()}, [sc], episodes=1)
    run = data["scenarios"][0]["runs"][0]
    assert len(run["dp"]) == run["n"] * 9 and len(run["dq"]) == run["n"] * 12
    html = write_viewer(data, tmp_path / "c.html").read_text(encoding="utf-8")
    payload = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(payload)["meta"]["policies"] == ["FORMATION"]
