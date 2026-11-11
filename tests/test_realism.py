import json

import numpy as np

from dronelab.cargo.env import ACT_DIM, F_HOVER_NOM, ROPE_K, CargoEnv, Conditions, Realism
from dronelab.mission import maps as M
from dronelab.mission.env import MissionEnv
from dronelab.mission.stress import EFFECTS, PRESETS, condition, main as stress_main
from dronelab.mission.testing import MissionFormationController, run_missions


def _easy_maps(n=2):
    return [M.generate_map(5_000_000 + k, "forest", 0.3) for k in range(n)]


def test_realism_off_changes_nothing():
    maps = _easy_maps()
    a = run_missions(MissionFormationController(), maps)
    b = run_missions(MissionFormationController(), maps, realism=Realism())
    assert [(r.score, r.steps) for r in a] == [(r.score, r.steps) for r in b]


def test_latency_delays_the_action_by_whole_steps():
    env = CargoEnv(1, seed=0, conditions=Conditions(init_noise=0.0), realism=Realism(latency=2))
    env.reset()
    push = np.zeros(ACT_DIM); push[0::4] = 1.0           # full thrust on every drone
    env.step(push)
    env.step(np.zeros(ACT_DIM))
    hover = env.thrust.sum()
    assert np.isclose(hover, 3 * F_HOVER_NOM, rtol=1e-3)   # still the zero command
    env.step(np.zeros(ACT_DIM))                          # the push from two steps ago lands now
    assert env.thrust.sum() > 1.5 * hover


def test_motor_lag_makes_thrust_rise_gradually():
    def first_substep_thrust(realism):
        env = CargoEnv(1, seed=0, conditions=Conditions(init_noise=0.0), realism=realism)
        env.reset()
        hover = env.motor_f.sum()
        push = np.zeros(ACT_DIM); push[0::4] = 1.0
        env.step(push)
        return env.thrust.sum() / hover
    instant, lagged = first_substep_thrust(Realism()), first_substep_thrust(Realism(motor_tau=0.05))
    assert lagged < instant
    assert lagged > 1.0


def test_battery_sag_and_wear_reduce_thrust():
    env = CargoEnv(1, seed=0, conditions=Conditions(init_noise=0.0), realism=Realism(battery_sag=0.6))
    env.reset()
    env.t[:] = 3000                                       # one minute into the flight
    env.step(np.zeros(ACT_DIM))
    assert np.allclose(env._gain, 0.4)
    worn = CargoEnv(1, seed=0, realism=Realism(motor_wear=0.3))
    worn.reset()
    assert (worn.wear <= 1).all() and (worn.wear >= 0.7).all() and worn.wear.min() < 1


def test_hardware_draws_stay_in_range_and_leave_gusts_alone():
    real = Realism(payload_spread=0.3, rope_len_spread=0.15, rope_k_spread=2.0)
    env = MissionEnv(64, seed=3, maps=[_easy_maps(1)[0]] * 64, realism=real)
    plain = MissionEnv(64, seed=3, maps=[_easy_maps(1)[0]] * 64)
    env.reset(); plain.reset()
    assert np.all(np.abs(env.cargo_mass - 0.6) <= 0.3) and env.cargo_mass.std() > 0.05
    assert np.all(env.rope_k >= ROPE_K / 2 - 1e-9) and np.all(env.rope_k <= ROPE_K * 2 + 1e-9)
    act = np.zeros((64, ACT_DIM))
    env.step(act); plain.step(act)
    assert np.allclose(env.gust_common, plain.gust_common)


def test_sensor_bias_is_constant_and_noise_is_fresh_each_step():
    m = _easy_maps(1)[0]
    clean = MissionEnv(1, seed=1, maps=[m]); biased = MissionEnv(1, seed=1, maps=[m], realism=Realism(sensor_bias=0.05))
    noisy = MissionEnv(1, seed=1, maps=[m], realism=Realism(sensor_noise=0.05))
    o0, o1, o2 = clean.reset(), biased.reset(), noisy.reset()
    assert np.allclose(o1 - o0, biased.sensor_bias, atol=1e-5)
    a = np.zeros((1, ACT_DIM))
    n1 = noisy.step(a)[0] - clean.step(a)[0]
    assert not np.allclose(n1, o2 - o0, atol=1e-3)


def test_wind_scale_scales_gusts():
    m = M.generate_map(5_000_001, "storm", 1.0)
    calm, windy = MissionEnv(1, seed=2, maps=[m]), MissionEnv(1, seed=2, maps=[m], realism=Realism(wind_scale=2.0))
    calm.reset(); windy.reset()
    a = np.zeros((1, ACT_DIM))
    for _ in range(5):
        calm.step(a); windy.step(a)
    assert np.allclose(windy.gust_common, 2 * calm.gust_common)


def test_stress_conditions_and_command(tmp_path, capsys):
    assert condition("default") == Realism()
    assert condition("latency") == Realism(latency=PRESETS["harsh"].latency)
    assert set(EFFECTS) <= set(PRESETS["harsh"].__dataclass_fields__)
    from dronelab.policy import ActorCritic
    import torch
    from dronelab.mission.env import MISSION_OBS_DIM
    ck = tmp_path / "p.pt"
    model = ActorCritic(obs_dim=MISSION_OBS_DIM, act_dim=ACT_DIM)
    torch.save({"config": model.config, "state_dict": model.state_dict(), "residual": 0.1}, ck)
    out = tmp_path / "s.json"
    stress_main(["--checkpoint", str(ck), "--themes", "forest", "--difficulties", "0.3", "--per", "1",
                 "--conditions", "default", "latency", "--json", str(out)])
    data = json.loads(out.read_text())
    assert set(data["conditions"]) == {"default", "latency"}
    assert "p" in data["conditions"]["latency"]["vs_baseline"]


def test_random_strength_spans_none_to_the_full_preset():
    real = PRESETS["realistic"]
    m = _easy_maps(1)[0]
    rnd = MissionEnv(256, seed=4, maps=[m] * 256, realism=real, realism_random=True)
    fixed = MissionEnv(4, seed=4, maps=[m] * 4, realism=real)
    rnd.reset(); fixed.reset()
    assert np.all((rnd.real_s >= 0) & (rnd.real_s <= 1)) and rnd.real_s.std() > 0.2
    assert set(np.unique(rnd.latency)) == {0, 1}                 # 0..latency steps, both occur
    assert np.all(fixed.real_s == 1) and np.all(fixed.latency == real.latency)
    weak, strong = rnd.real_s < 0.1, rnd.real_s > 0.9
    assert (1 - rnd.wear[weak]).max() <= real.motor_wear * 0.1 + 1e-9
    assert (1 - rnd.wear[strong]).max() > (1 - rnd.wear[weak]).max()
    assert np.allclose(rnd.wind_factor, 1 + (real.wind_scale - 1) * rnd.real_s)


def test_latency_can_differ_between_envs():
    env = CargoEnv(2, seed=0, conditions=Conditions(init_noise=0.0), realism=Realism(latency=2))
    env.reset()
    env.latency[:] = [0, 2]
    push = np.zeros((2, ACT_DIM)); push[:, 0::4] = 1.0
    env.step(push)
    per_env = env.thrust.reshape(2, -1).sum(1)
    assert per_env[0] > 1.5 * per_env[1]                         # no delay: the push lands at once


def test_realism_without_payload_spread_keeps_the_training_mass():
    env = MissionEnv(64, seed=5, difficulty=1.0, realism=Realism(latency=1), realism_random=True)
    env.reset()
    assert env.cargo_mass.std() > 0.05 and env.rope_len.std() > 0.02


def test_single_effects_at_a_named_preset_level():
    assert condition("latency@realistic") == Realism(latency=PRESETS["realistic"].latency)
    assert condition("wind_scale@harsh") == condition("wind_scale")
    import pytest
    with pytest.raises(ValueError):
        condition("latency@nowhere")


def test_parallel_stress_gives_the_same_scores(tmp_path):
    import torch
    from dronelab.mission.env import MISSION_OBS_DIM
    from dronelab.policy import ActorCritic
    ck = tmp_path / "p.pt"
    model = ActorCritic(obs_dim=MISSION_OBS_DIM, act_dim=ACT_DIM)
    torch.save({"config": model.config, "state_dict": model.state_dict(), "residual": 0.1}, ck)
    common = ["--checkpoint", str(ck), "--themes", "forest", "--difficulties", "0.3", "--per", "1",
              "--conditions", "default", "latency@realistic"]
    stress_main(common + ["--json", str(tmp_path / "a.json")])
    stress_main(common + ["--jobs", "2", "--json", str(tmp_path / "b.json")])
    a, b = (json.loads((tmp_path / f).read_text())["conditions"] for f in ("a.json", "b.json"))
    assert a == b


def test_leave_one_out_conditions_switch_one_effect_off():
    from dataclasses import replace
    assert condition("realistic-wind_scale") == replace(PRESETS["realistic"], wind_scale=1.0)
    assert condition("harsh-latency") == replace(PRESETS["harsh"], latency=0)


def test_realism_max_draws_strength_beyond_the_preset():
    real = PRESETS["realistic"]
    env = MissionEnv(256, seed=6, maps=[_easy_maps(1)[0]] * 256, realism=real, realism_random=True, realism_max=1.5)
    env.reset()
    assert env.real_s.max() <= 1.5 and env.real_s.max() > 1.3
    assert env.latency.max() == real.latency                 # delay never exceeds the preset
    assert env.wind_factor.max() > real.wind_scale
