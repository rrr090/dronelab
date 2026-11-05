# dronelab

Reinforcement learning for a team of drones carrying slung cargo, from a
single hovering drone to a three-drone team delivering a load through
procedurally generated maps under wind and gusts. Everything is simulated in
numpy and trained with PPO in PyTorch, and every stage has a test suite,
a hand-written baseline and an interactive 3D viewer.

![Cargo team flying a generated forest map](docs/images/mission-forest.png)

## Stages

| Stage | Task | Model |
|---|---|---|
| **hover** | One planar quadrotor holds a hover point. | 6 inputs → 2 motor thrusts |
| **cargo** | Three 3D quadrotors hold one cargo on elastic ropes under gusts, wind, motor faults and cargo kicks. | 63 inputs → 12 attitude commands |
| **mission** | The cargo team flies a generated map (forest, city, gates, storm, mixed) to a goal pad. Score = cargo health on delivery. | 93 inputs → 12 attitude commands |

Each stage builds on the one before. Mission training warm-starts from the
trained cargo network.

## Quick start

```bash
git clone <repo-url> && cd dronelab
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

For GPU training, install the CUDA build of PyTorch first
(https://pytorch.org/get-started/locally/). The physics runs on the CPU, so
a GPU only speeds up the network updates.

Every stage has the same three commands:

```bash
dronelab <task> train        # train a policy with PPO   -> checkpoints/
dronelab <task> evaluate     # score it on the test suite
dronelab <task> visualize    # open the 3D viewer         -> outputs/
```

`python -m dronelab ...` works too, without installing. Add `--help` to any
command to see its options.

```bash
dronelab cargo train                           # ~20 min on a laptop CPU
dronelab cargo visualize --policy rl formation # trained vs baseline, side by side
dronelab mission train                         # warm-starts from checkpoints/cargo_mlp.pt
dronelab mission visualize --themes storm city --difficulty 1.0
```

The viewers are single self-contained HTML files. They load three.js from a
CDN, so they need an internet connection.

## How it works

### Physics

- **Drones** (`dronelab/cargo/env.py`) are rigid bodies with full 3D attitude
  and four motors each in an X layout, integrated at 500 Hz.
- **Cargo** is a point mass hanging from three ropes. Each rope is a
  spring-damper that only pulls when stretched, so it can go slack and then
  snap tight.
- **Wind** is steady wind plus an Ornstein–Uhlenbeck gust process, with a
  component shared by all bodies and a smaller per-body component. The cargo
  catches more wind than a drone.
- **Vectorization:** every environment is vectorized over a batch, so PPO
  runs 64 environments in parallel.

### Control

The policy acts at 50 Hz. For each drone it outputs collective thrust, roll,
pitch and yaw rate. An onboard geometric attitude controller turns these into
motor thrusts, the same split as a real flight controller (PX4, Betaflight).

### Missions

- **Maps** (`dronelab/mission/maps.py`) are generated from a seed, a theme and
  a difficulty. They contain trees, buildings, walls with a gap, low crates
  the cargo can clip, and wind zones (side winds, downdrafts, updrafts).
- **Routes:** an A* planner on a clearance grid finds a route wide enough for
  the formation, avoiding strong wind where it can.
- **Reference:** a reference point moves along the route. It slows in turns,
  brakes before the goal, and waits if the team falls behind.
- **Sensing:** the policy gets obstacle distance and direction for every
  drone, the cargo and each rope.
- **Cargo health** starts at 100. It drops with rope jerks, high g-forces,
  cargo impacts and ropes rubbing on obstacles.

### Training

- **Algorithm:** PPO (`dronelab/ppo.py`) with observation normalization, GAE
  and a learning-rate decay.
- **Curriculum:** domain randomization (wind, gusts, cargo mass, rope length,
  motor wear, sensor noise, map density) ramps from calm to full difficulty
  over the first 60% of training.
- **Mission warm start:** mission training starts from the cargo network with
  low exploration noise. At the default noise level, random thrust jerks the
  ropes hard enough to destroy the cargo before anything is learned.

## Results

### Cargo hold

Results from `dronelab cargo evaluate`, 8 episodes per scenario, after 400
training updates (~20 min on a laptop CPU):

| Scenario | Trained policy | Formation baseline |
|---|---|---|
| storm (stronger than training) | **75%** success | 12% success |
| steady wind, cargo error | **0.04 m** | 0.17 m |
| gusts, cargo error | **0.06 m** | 0.12 m |
| motor fault, cargo error | **0.02 m** | 0.09 m |
| calm air, cargo error | 0.02 m | **~0.00 m** |

![Cargo team in a storm](docs/images/cargo-storm.png)

### Missions

Results from `dronelab mission evaluate` on 30 unseen maps:

| | Trained policy | Formation baseline |
|---|---|---|
| Average score | 69.8 | **72.6** |
| Delivered | 73% | **87%** |
| Cargo health when delivered | **95.2** | 83.8 |
| Rope-jerk damage | **7.8** | 24.9 |

The trained team handles the cargo much more gently than the baseline. It
still fails more missions, because it lags the planned route and clips
obstacles. That is the main open problem (see the roadmap).

![Mission viewer, city map](docs/images/mission-city.png)

## Project layout

```
dronelab/
├── __main__.py        CLI: dronelab <task> <command>
├── policy.py          actor-critic MLP (with running observation normalization)
├── ppo.py             vectorized PPO trainer
├── viz.py             frame conversion and HTML writer for the viewers
├── hover/             planar single drone: env, tests, viewer, CLI
├── cargo/             3D three-drone cargo: env, scenarios, baseline, viewer, CLI
└── mission/           map generator + planner, mission env, scoring, viewer, CLI
tests/                 pytest suite (physics, planning, scoring, viewer export)
docs/images/           screenshots for this README
```

Each task folder contains the same pieces: `env.py` (physics and reward),
`testing.py` (scenarios, metrics, baseline controller), `viewer.py` and
`viewer.html` (3D viewer), plus `train.py`, `evaluate.py` and `visualize.py`.

## Tests

```bash
pytest -q
```

CI runs the suite on every push (`.github/workflows/tests.yml`).

## Roadmap

- **Missions:** reward obstacle clearance more strongly, plan routes with
  wider margins, and add a safety layer that falls back to the baseline near
  obstacles.
- **Realism:** add motor lag, state-estimation noise, a rigid-body cargo that
  can spin, and direct motor control without the attitude controller.
- **Interpretability:** log hidden activations along trajectories, find the
  neurons that track cargo swing, rope tension and obstacles, and ablate them.

## Limitations

This is a research and learning project. The physics is simplified: perfect
state observation, instant motors, point-mass cargo, and no aerodynamic
interaction between drones. Results are not a validated flight model.

## License

MIT, see [LICENSE](LICENSE).
