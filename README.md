<div align="center">

# dronelab

**Reinforcement learning for drone teams carrying slung cargo through wind, gusts and generated obstacle maps.**

[![Tests](https://github.com/rrr090/dronelab/actions/workflows/tests.yml/badge.svg)](https://github.com/rrr090/dronelab/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/physics-NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![three.js](https://img.shields.io/badge/viewer-three.js-000000?logo=threedotjs&logoColor=white)](https://threejs.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
<br>
[![GitHub stars](https://img.shields.io/github/stars/rrr090/dronelab?style=social)](https://github.com/rrr090/dronelab/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/rrr090/dronelab?style=social)](https://github.com/rrr090/dronelab/network/members)
[![Last commit](https://img.shields.io/github/last-commit/rrr090/dronelab)](https://github.com/rrr090/dronelab/commits/main)
[![Repo size](https://img.shields.io/github/repo-size/rrr090/dronelab)](https://github.com/rrr090/dronelab)

<img src="docs/images/mission-demo.gif" alt="Three drones carrying a cargo through a generated city map to the goal pad" width="100%">

<sub>A trained three-drone team flies a cargo through an unseen city map, with wind zones and gusts, and delivers it with 98% health. The recording comes from the built-in 3D viewer.</sub>

**[Live demo](https://rrr090.github.io/dronelab/)** · [Quick start](#quick-start) · [How it works](#how-it-works) · [Results](#results) · [Project layout](#project-layout) · [Roadmap](#roadmap)

</div>

---

## Highlights

- **Full 3D multi-drone physics in NumPy.** Rigid-body quadrotors with four motors each, a cargo on elastic ropes that can go slack and snap tight, and turbulent gusts. Everything is vectorized over a batch of environments.
- **Procedural missions.** Forest, city, gate, storm and mixed maps are generated from a seed, with an A\* route planner and a cargo-health score.
- **PPO from scratch.** One small, readable trainer with observation normalization, GAE, a curriculum and warm starts between stages.
- **A baseline and a test suite at every stage.** Every policy is scored against a hand-written controller on unseen scenarios.
- **Hardware stress test.** Motor lag, control delay, sensor noise and bias, battery sag, motor wear, stronger wind, and payload and rope variation can be switched on one at a time or together, and the policy can be trained with them.
- **Interactive 3D viewers.** Each viewer is a single self-contained HTML file with chase, follow, overview and top cameras, a health HUD and a damage chart.

## Stages

| Stage | Task | Network |
|---|---|---|
| **hover** | One planar quadrotor holds a hover point under pushes and motor faults. | 6 → 2 motor thrusts |
| **cargo** | Three 3D quadrotors hold one cargo on ropes under gusts, wind, motor faults and cargo kicks. | 63 → 12 attitude commands |
| **mission** | The team flies the cargo through a generated map to a goal pad. Score = cargo health on delivery. | 123 → 12 attitude commands |

Each stage builds on the one before it.

<table>
  <tr>
    <td width="33%"><img src="docs/images/hover-lanes.png" alt="Hover test scenarios"></td>
    <td width="33%"><img src="docs/images/cargo-storm.png" alt="Cargo team in a storm"></td>
    <td width="33%"><img src="docs/images/mission-forest.png" alt="Mission in a forest map"></td>
  </tr>
  <tr>
    <td align="center"><sub><b>hover:</b> 12 disturbance scenarios</sub></td>
    <td align="center"><sub><b>cargo:</b> holding a load in a storm</sub></td>
    <td align="center"><sub><b>mission:</b> forest map with wind zones</sub></td>
  </tr>
</table>

## Quick start

```bash
git clone https://github.com/rrr090/dronelab.git && cd dronelab
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

> [!TIP]
> For GPU training, install the CUDA build of PyTorch first
> ([pytorch.org/get-started](https://pytorch.org/get-started/locally/)). The physics runs
> on the CPU, so a GPU only speeds up the network updates.

Every stage has the same three commands:

```bash
dronelab <task> train        # train a policy with PPO   -> checkpoints/
dronelab <task> evaluate     # score it on the test suite
dronelab <task> visualize    # open the 3D viewer         -> outputs/
```

For example:

```bash
dronelab cargo train                           # ~20 min on a laptop CPU
dronelab cargo visualize --policy rl formation # trained vs baseline, side by side
dronelab mission train                         # learns corrections on top of the baseline
dronelab mission visualize --themes storm city --difficulty 1.0
dronelab mission stress --checkpoint checkpoints/chosen.pt   # fly it on realistic hardware
dronelab mission site                          # rebuild the project site    -> site/
```

`python -m dronelab ...` works without installing. Add `--help` to any command to see its options. The viewers load three.js from a CDN, so they need an internet connection.

## How it works

<details open>
<summary><b>Physics</b></summary>

- **Drones** (`dronelab/cargo/env.py`) are rigid bodies with full 3D attitude and four motors each in an X layout, integrated at 500 Hz.
- **Cargo** is a point mass hanging from three ropes. Each rope is a spring-damper that only pulls when stretched, so it can go slack and then snap tight.
- **Wind** is steady wind plus an Ornstein–Uhlenbeck gust process. One gust component is shared by all bodies, and a smaller one differs per body. The cargo catches more wind than a drone does.
- **Vectorization:** every environment is vectorized, so PPO trains on 64 environments in parallel.

</details>

<details open>
<summary><b>Control</b></summary>

The policy acts at 50 Hz. For each drone it outputs collective thrust, roll, pitch and yaw rate. An onboard geometric attitude controller on SO(3) turns these commands into motor thrusts. Real flight controllers such as PX4 and Betaflight split the work the same way.

</details>

<details open>
<summary><b>Missions</b></summary>

- **Maps** (`dronelab/mission/maps.py`) contain trees, buildings, walls with a gap, low crates the cargo can clip, and wind zones (side winds, downdrafts and updrafts).
- **Routes:** an A\* planner on a clearance grid finds a route wide enough for the formation. Where it can, the route avoids strong wind.
- **Reference:** a reference point moves along the route. It slows in turns, brakes before the goal, and waits if the team falls behind.
- **Sensing:** the policy gets obstacle distance and direction for every drone, the cargo and each rope.
- **Cargo health** starts at 100. It drops with rope jerks, high g-forces, impacts and ropes rubbing on obstacles.

</details>

<details open>
<summary><b>Training</b></summary>

- **Algorithm:** PPO (`dronelab/ppo.py`) with observation normalization, GAE and learning-rate decay.
- **Curriculum:** domain randomization ramps from calm to full difficulty over the first 60% of training. It covers wind, gusts, cargo mass, rope length, motor wear, sensor noise and map density.
- **Residual missions:** the formation baseline keeps flying the route. The network learns a bounded correction on top of it, so training starts at the baseline's level instead of from zero. The chosen policy also sees peak rope tension and past frames, and is penalised as tension nears the damage level.
- **Realism training:** with `--realism realistic`, each training episode draws a strength from none up to the full realistic hardware preset (see [Realistic hardware](#realistic-hardware)). The best checkpoint is then chosen on the mean validation score with and without the preset, so a gain on realistic hardware cannot hide a loss on the default one.
- **Checkpoint selection:** the best network is chosen on validation maps, which are separate from the training and test maps.

</details>

## Results

### Cargo hold

Results from `dronelab cargo evaluate`, 8 episodes per scenario, after 400 training updates (~20 min on a laptop CPU):

| Scenario | Trained policy | Formation baseline |
|---|---|---|
| storm (stronger than training) | **75%** success | 12% success |
| steady wind, cargo error | **0.04 m** | 0.17 m |
| gusts, cargo error | **0.06 m** | 0.12 m |
| motor fault, cargo error | **0.02 m** | 0.09 m |
| calm air, cargo error | 0.02 m | **~0.00 m** |

### Missions

The chosen policy on 100 new maps, used only for this final report and not for training or for choosing a model:

| | Chosen policy | Formation baseline |
|---|---|---|
| Average score | **96.3** | 74.9 |
| Delivered | **100%** | 91% |
| Cargo health when delivered | **96.3** | 82.3 |
| Rope-jerk damage | **3.4** | 22.7 |

The policy gains **+21.5 points, 95% interval +16.6 to +26.8**, paired map by map, and scores higher on 98 of the 100 maps. Flight time is 42.9 s against 42.0 s, so the gain does not come from flying slower. The interval covers which maps were drawn, not training luck: this is one model, picked as the best of several runs. Scores from different map sets are not comparable, because some sets are easier.

How it got here:

| Policy | Score vs baseline | Delivered vs baseline | Maps |
|---|---|---|---|
| Full actions, no baseline | 64.9 vs 74.2 | 69% vs 89% | test set |
| Residual, correction 0.3 | 65.4 vs 74.2 (seeds 49, 78, 69) | 83% vs 89% | test set |
| Residual, correction 0.1, best validation checkpoint | 76.0 vs 74.2 | 91% vs 89% | test set |
| Same policy, on new maps | 76.4 vs 74.9 | 92% vs 91% | 100 new maps |
| History inputs and rope-jerk penalty, three seeds | 79.1 vs 72.9 | 96% vs 94% | 100 new maps |
| Best of those seeds, trained further | 91.1 vs 78.0 | 98% vs 95% | 100 new maps |
| That policy, trained further (release v0.2.0) | 92.9 vs 76.7 | 99% vs 95% | 100 new maps |
| Two more rounds of continued training | 94.9 vs 76.0 | 99.5% vs 93% | 200 new maps |
| Two rounds of realism training (chosen policy) | 96.3 vs 74.9 | 100% vs 91% | 100 new maps |

Building on the baseline removed most obstacle crashes, but a large correction shook the cargo apart through rope jerk, so the correction was made smaller. Rope jerk was still the main source of damage, so the policy was given peak tension and past frames as inputs, plus a penalty that starts before damage does. All three seeds beat the baseline, by +6.2 points on average (interval +4.3 to +8.5). Both changes went in together, so this does not show how much each one contributed.

Three further training runs then continued from the best seed at full difficulty. All three beat their starting point on unused maps. The best was promoted, and on the final maps it gains **+5.1 points (interval +3.6 to +6.8)** over the model it replaced, with the same delivery rate: the cargo simply arrives in better condition. Each step was judged on maps not used for the previous one.

Six more runs continued from that model, three of them with a doubled rope-jerk penalty. All six beat it on unused maps, and doubling the penalty added nothing. The best of the six was released as v0.2.0. On its final maps it gained **+2.8 points (interval +1.7 to +4.4)** over the model it replaced, mostly through less rope jerk (6.6 against 9.2).

Two more rounds of continued training each gained one to two points on unused maps. The first added a capped penalty on rope-tension spikes (`--spike-weight 2.0`). Its best run won on unused maps, but on average the penalty did not beat continued training without it, so the gain is credited to continued training. `--resume` keeps the penalty, so every later round trained with it too. The second round tested a critic that also sees the wind and gust state, against the same continuation with three training seeds per arm, and it made no difference. The last two rounds trained with realistic hardware (see below). Against v0.2.0 on the final maps, the chosen policy gains **+2.8 points (interval +1.7 to +4.1)** on the default hardware and **+10.2 points (interval +5.4 to +14.9)** on realistic hardware.

The chosen policy is available as a [release download](https://github.com/rrr090/dronelab/releases/tag/v0.3.0), so the final result can be checked without training:

```bash
curl -L --create-dirs -o checkpoints/chosen.pt https://github.com/rrr090/dronelab/releases/download/v0.3.0/chosen.pt
dronelab mission evaluate --checkpoint checkpoints/chosen.pt --seed 20000000
dronelab mission stress --checkpoint checkpoints/chosen.pt --conditions default realistic --seed 20000000
```

To reproduce the steps:

```bash
dronelab mission train --history --jerk-weight 1.0 --seed 1 --out checkpoints/jerk_s1.pt
dronelab mission train --resume checkpoints/jerk_s1_best.pt --curriculum 0 --critic-warmup 0 --lr 5e-5 --seed 0 --out checkpoints/champ_s0.pt
dronelab mission train --resume checkpoints/champ_s0_best.pt --curriculum 0 --critic-warmup 0 --lr 5e-5 --seed 0 --out checkpoints/next_s0.pt
dronelab mission train --resume checkpoints/g4_s6_best.pt --realism realistic --curriculum 0 --critic-warmup 10 --lr 5e-5 --seed 11 --out checkpoints/rz_s11.pt
dronelab mission train --resume checkpoints/rz_s11_best.pt --curriculum 0 --critic-warmup 10 --lr 5e-5 --seed 13 --out checkpoints/rz2_s13.pt
```

The two rounds between `next_s0` and `g4_s6` follow the same pattern from the previous best (`--resume`, `--curriculum 0`, `--lr 5e-5`), with `--spike-weight 2.0` added in the first. `--resume` carries the spike penalty, the realism setting and the jerk weight forward, so the last command needs no extra flags.

![Mission viewer, city map](docs/images/mission-city.png)

### Realistic hardware

The default simulator has ideal hardware: motors respond instantly, commands act at once, sensors are exact, and every cargo and rope is the same. `dronelab mission stress` flies the same unseen maps with those effects switched on. The **realistic** preset is a guess at typical small-drone hardware, not a measured one: 30 ms motor lag, one 20 ms step of control delay, sensor noise and a constant sensor bias, 10% thrust loss per minute of flight, up to 10% motor wear, 1.25× wind and gusts, and payload, rope length and rope stiffness that vary between flights.

On the final 100 maps:

| Condition | Chosen policy | v0.2.0 policy | Formation baseline |
|---|---|---|---|
| Default hardware | **96.3** (100%) | 93.5 (100%) | 74.9 (91%) |
| Realistic preset | **74.0** (85%) | 63.8 (79%) | 34.2 (48%) |
| 20 ms control delay only | **95.9** (100%) | 90.1 (100%) | 69.4 (86%) |
| 1.25× wind only | **85.7** (100%) | 72.3 (92%) | 46.7 (65%) |

Cells are average score (delivered share). Every policy loses a lot on realistic hardware, and most of the loss is cargo destroyed by rope jerk. Switching the effects off one at a time shows why. Stronger wind and slower actuation, meaning delay and motor lag, do far more harm together than apart. Removing the 20 ms delay from the full preset recovered 8.2 points, although the delay on its own cost only 0.2. Removing the stronger wind recovered 17.8. That attribution was measured on the first realism-trained model.

Realism training draws a strength for each episode, from no effects up to the full preset. In the first round, three runs with it beat three runs of continued training without it under the realistic preset by **+1.8 points (interval +0.7 to +3.1)**, with no loss on the default hardware. Training on effects up to 1.5× the preset was worse than up to 1×, by 2.0 points on the preset, so the extra range did not help. A second round of realism training gained another **+3.0 points (interval +2.4 to +3.6)** on realistic hardware over 500 unused maps and three gust seeds.

The baseline was not retuned for delay or motor lag, so its realistic-hardware score understates what a tuned controller could do. The fair comparison under realism is between the learned policies.

### What the network learned

`dronelab mission interpret` looks for neurons that track cargo swing, rope tension and obstacles, always compared with untrained networks of the same shape. The answers below are for the v0.2.0 policy, before realism training. They have not been rerun on the current one:

- **Single neurons mostly do not track these concepts.** No neuron beats the untrained networks for swing or tension. Obstacles are the exception, with 1 neuron in the first hidden layer and 6 in the second.
- **Tension-tuned neurons have a measurable effect in flight.** Holding the top 32 at their average costs **2.3 points** (interval −3.6 to −1.2) and raises rope-jerk damage from 5.8 to 8.0, against 0.7 points for each of two sets of neurons with similar influence. The previous chosen model showed the same pattern (3.5 points against 0.9 and 1.2), but the current one was trained from it, so this is not an independent replication.
- **Swing is suggestive, and obstacles show nothing.** Swing neurons cost 1.9 points (interval −3.1 to −0.9), against no loss and 0.8 points for their controls. In the previous model the swing effect could not be told apart from its controls, so this needs more models. Removing obstacle neurons costs 0.1 points, less than the controls, and clearance does not change.

This is one model, 60 maps and two control sets per concept, so treat it as evidence, not proof.

## Project layout

```
dronelab/
├── __main__.py        CLI: dronelab <task> <command>
├── policy.py          actor-critic MLP with running observation normalization
├── ppo.py             vectorized PPO trainer
├── viz.py             frame conversion and HTML writer for the viewers
├── hover/             planar single drone
├── cargo/             3D three-drone cargo hold
└── mission/           map generator + planner, mission env, scoring, hardware stress test
tests/                 pytest suite: physics, planning, scoring, viewer export
site/                  generated project site, deployed to GitHub Pages
docs/images/           README media
```

Each task folder has the same pieces:
- `env.py`: physics and reward
- `testing.py`: scenarios, metrics and the baseline controller
- `viewer.py` and `viewer.html`: the 3D viewer
- `train.py`, `evaluate.py` and `visualize.py`: the three commands

## Tests

```bash
pytest -q
```

CI runs the suite on every push.

## Roadmap

- [x] Planar hover with disturbance test suite
- [x] Three-drone 3D cargo hold under gusts and faults
- [x] Procedural missions with route planning and cargo health
- [x] Residual mission policy that clearly beats the baseline
- [x] Project site with 3D replays and measured results
- [x] Interpretability, first pass: test for neurons tracking swing, tension and obstacles, and ablate them in flight (one model)
- [x] Realism, first pass: motor lag, control delay, sensor noise and bias, battery sag, motor wear, wind, payload and rope variation, with a stress test and training under them
- [ ] Interpretability: replicate the rope-tension effect across models
- [ ] GPU-vectorized physics in PyTorch, for thousands of parallel environments
- [ ] Realism: slower position updates and state-estimation delay, a rigid-body cargo, direct motor control

## Limitations

This is a research and learning project. The physics is simplified: a point-mass cargo, no aerodynamic interaction between drones, and state updates at the full 50 Hz control rate. The default results use ideal hardware. The realism layer adds hardware effects, but its values are plausible guesses, not measurements of a real drone. Results are not a validated flight model.

## License

MIT, see [LICENSE](LICENSE). If you find the project useful, a ⭐ helps others find it.
