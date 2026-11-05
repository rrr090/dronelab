import numpy as np

from drone_mi.env import ACT_DIM, OBS_DIM, PlanarQuadrotorEnv, TARGET_Z


def test_shapes():
    env = PlanarQuadrotorEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (OBS_DIM,)
    obs, r, term, trunc, _ = env.step(np.zeros(ACT_DIM))
    assert obs.shape == (OBS_DIM,)
    assert isinstance(r, float)


def test_hover_thrust_keeps_height():
    # Thrust that exactly cancels gravity should keep z roughly constant
    # from a level, at-rest start at the target height.
    env = PlanarQuadrotorEnv(seed=0, init_noise=0.0)
    env.reset()
    hover_action = np.full(ACT_DIM, 2 * (9.81 * 0.5 / 2) / 6.0 - 1.0)
    for _ in range(50):
        env.step(hover_action)
    assert abs(env.state[1] - TARGET_Z) < 0.05


def test_falling_terminates():
    env = PlanarQuadrotorEnv(seed=0, init_noise=0.0)
    env.reset()
    done = False
    for _ in range(1000):
        _, _, done, _, _ = env.step(np.full(ACT_DIM, -1.0))
        if done:
            break
    assert done
