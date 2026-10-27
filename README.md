# Adaptive Gait — Velocity Tracking on Complex Terrain

> 📄 [中文文档](README_zh.md)

## Overview

This is a reinforcement learning project for training robust velocity-tracking locomotion policies on the Unitree Go2 quadruped robot, built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) v2.3.2 and trained with [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (PPO).

The core objective is to train a generalizable gait controller that can follow velocity commands across diverse real-world terrains — flat ground, slopes, stairs, and rough surfaces — rather than overfitting to a single idealized terrain.

### Training Demo

![](assets/demo.gif)

---

## Key Features

### 🏔️ Mixed Terrain Training for Real-World Generalization

The agent is trained on a procedurally-generated composite terrain (`COBBLESTONE_ROAD_CFG`) that combines 7 terrain types in a single environment:

| Terrain Type | Proportion | Description |
|---|---|---|
| Flat plane | 20% | Baseline locomotion |
| Smooth slope | 15% | Uphill / downhill |
| Inverted slope | 10% | Reverse pyramid slope |
| Random rough | 15% | Stochastic height field |
| Pyramid stairs | 20% | Regular stair climbing |
| Inverted stairs | 20% | Descending step terrain |
| Discrete obstacles| (optional) | Scattered obstacles |

Training on diverse terrains simultaneously encourages the policy to learn robust, generalizable behaviors that transfer well to real-world deployment.

---

### 📈 Curriculum Learning

To progressively increase training difficulty and improve learning efficiency, two curriculum strategies are applied:

1. **Terrain Curriculum** (`terrain_levels_vel`): Robots that successfully traverse a terrain sub-tile are promoted to harder difficulty levels; robots that fail are demoted. This prevents the policy from being overwhelmed early in training.

2. **Velocity Command Curriculum** (`lin_vel_cmd_levels`): The commanded velocity range starts conservatively and expands as the policy's tracking performance improves, allowing the agent to incrementally learn to run faster.

---

### 🔄 Transfer Learning via Pretrained Models

Rather than training from scratch on complex terrain, the policy is initialized from a pretrained model obtained by training on flat ground. This **two-stage curriculum**:

- **Stage 1**: Train on flat terrain to acquire basic locomotion primitives (gait rhythm, balance, velocity tracking).
- **Stage 2**: Load the flat-terrain checkpoint as a pretrained model and fine-tune on the mixed terrain environment.

To load a pretrained checkpoint during training:

```bash
python scripts/rsl_rl/train.py \
    --task=<TASK_NAME> \
    --pretrained=/path/to/flat_terrain_checkpoint.pt
```

---

### 🔧 RSL-RL v5.x Compatibility Layer

RSL-RL v5.0+ introduced significant breaking changes to its configuration API and checkpoint format. Since the original Isaac Lab template was written against older versions, this project includes a local compatibility shim in `scripts/rsl_rl/train.py`.

#### Configuration Compatibility (`handle_deprecated_rsl_rl_cfg`)

Automatically remaps the old `policy`-based agent configuration dict into the new `actor`/`critic` split format expected by `rsl-rl-lib >= 5.0`:

```python
# Old format (< v5.0):
cfg["policy"] = {"actor_hidden_dims": [...], "init_noise_std": 1.0, ...}

# New format (v5.x):
cfg["actor"] = {"class_name": "MLPModel", "hidden_dims": [...], ...}
cfg["critic"] = {"class_name": "MLPModel", "hidden_dims": [...], ...}
```

#### Checkpoint Compatibility (`handle_deprecated_rsl_rl_checkpoint`)

Automatically converts old-format `.pt` checkpoint files (single `model_state_dict`) into the v5.x format (separate `actor_state_dict` / `critic_state_dict`), including key remapping for `MLPModel`'s internal structure:

```
Old key                 →  New key (MLPModel)
────────────────────────────────────────────────
"0.weight"              →  "mlp.0.weight"
"std"                   →  "distribution.std_param"
```

These adapters run automatically and require no manual intervention.

---

## AER Reward Shaping

This project implements the **AER (Adaptive Exponential Reward)** reward structure from the locomotion research literature. Rather than summing all reward terms linearly, penalties are applied multiplicatively via an exponential gate:

```
R_total = R_positive × exp(R_negative / σ)
```

- **Positive terms**: `track_lin_vel_xy`, `track_ang_vel_z`, `energy_efficiency`
- **Negative terms**: joint torque, velocity, acceleration penalties; foot slip; orientation error; body height deviation

This design encourages the robot to prioritize task completion while softly penalizing physically undesirable behaviors.

---

## Installation

1. Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

2. Clone this repository **outside** the IsaacLab directory:
   ```bash
   git clone <repo-url>
   cd <project-dir>
   ```

3. Install the project extension in editable mode:
   ```bash
   python -m pip install -e source/go2_demo
   ```

4. Verify installation by listing available tasks:
   ```bash
   python scripts/list_envs.py
   ```

---

## Training

### Standard Training (Mixed Terrain)

```bash
cd /home/css/work/unitree/rl/zhouzhiguang_repo/my_branchh/Go2_Rough_Curriculum
source ~/miniconda3/etc/profile.d/conda.sh && conda activate go2_rough_curriculum
export OMNI_KIT_ACCEPT_EULA=1

# Train with 4096 environments in headless mode; --run_name is appended to the log directory name
python scripts/rsl_rl/train.py \
    --task=Go2-velocity-v0 \
    --num_envs 4096 \
    --headless \
    --run_name curriculum_fix_v2
```

For long runs, detach it and redirect the log:

```bash
nohup python scripts/rsl_rl/train.py \
    --task=Go2-velocity-v0 --num_envs 4096 --headless \
    --run_name curriculum_fix_v2 > train_curriculum_v2.log 2>&1 &
```

Monitor:

```bash
tensorboard --logdir logs/rsl_rl
```

> Default config (`go2_demo_velocity.py`) now generates terrains in ascending difficulty
> (`terrain_generator.curriculum=True`), keeps the 187 height-scan rays (observation dim **247**,
> same as historical runs), `lin_vel_y ∈ [-0.1, 0.1]`, checkpoints terrain levels to
> `logs/rsl_rl/go2_Complex_terrain/<RUN>/terrain_levels.pt`, and logs the per-episode terrain
> variation range as `Terrain_Scan/*`. The tightened `track_lin_vel_xy` std and the new
> `lin_vel_y` range change training dynamics, so old checkpoints belong to a different
> experiment group — retrain rather than resume.

### Training with Pretrained Flat-Terrain Model
```bash
python scripts/rsl_rl/train.py \
    --task=<TASK_NAME> \
    --num_envs=4096 \
    --pretrained=/path/to/flat_terrain/model_XXXX.pt
```

### Resume Training from Checkpoint
```bash
python scripts/rsl_rl/train.py --task=<TASK_NAME> --resume
```

### Playback / Evaluation
```bash
python scripts/rsl_rl/play.py --task=<TASK_NAME> --num_envs=16
```

### Checking Training Curves (TensorBoard)

```bash
tensorboard --logdir logs/rsl_rl/go2_Complex_terrain
```

Key scalars: `Curriculum/terrain_levels_vel` (mean terrain level, should keep rising),
`Curriculum/terrain_levels_vel/{up,down}` (promotion / demotion rate),
`Curriculum/terrain_levels_vel/level_<sub_terrain>` (per-terrain-type mean level, enabled by
`log_per_terrain=True`), `Train/mean_reward`, `Episode_Reward/track_lin_vel_xy` and the
`Episode_Termination/*` rates.

### Test Method: Verifying the Height Scanner

Two levels of testing: **A static** (standing at spawn, validates the sensor itself) and **B dynamic**
(during training / playback, validates what the policy actually receives). The two differ by more
than 10x — that is expected, they measure different time scales.

**Do not mix the three layers:**

| Layer | Source | Statistics | Stairs (measured) |
|---|---|---|---|
| ① Static | `verify_height_scanner.py` | one frame at spawn, std of 187 rays | **0.000 – 0.024 m** |
| ② Dynamic frame | `Terrain_Scan/frame_range_*` | per-step frame range, averaged over the episode | **0.14 – 0.20 m** |
| ③ Dynamic cumulative | `Terrain_Scan/range_*` | global `max − min` over the whole episode | **0.27 – 0.66 m** |

> Why ③ reaches 0.66 m: the robot walks ~14 m in 20 s while each terrain block is only 8 m, so it
> crosses 1–2 blocks, plus the pyramid stairs accumulate 6 steps × 0.055 m ≈ 0.33 m. It measures
> “how much terrain the robot went through”, **not** the sensor resolution. Use ① or ② for that.

**Test A — static**

```bash
python scripts/verify_height_scanner.py --task=Go2-velocity-v0 --headless --levels 0,3,6,9
```

```bash
# measured with size=[1.6,1.0], offset=(0,0,20)
flat 0.0000 | smooth_slope 0.0000 | smooth_slope_inv 0.0000 | random_rough 0.0212–0.0222
pyramid_stairs 0.0000–0.0096 | pyramid_stairs_inv 0.0056–0.0242 | discrete_obstacles 0.0000
height_scan range [-0.384, -0.092], clip hits 0%
```

Pass criteria: observation not saturated (< 1% at clip bounds); `random_rough ≥ 0.01` with
`flat ≈ 0`; at least some stair/obstacle cells > 0.01. Known blind spot: `platform_width = 3.0 m`
(half-width 1.5 m) exceeds the 0.8 m forward view, so stairs are invisible while standing at spawn.

**Test B — dynamic**

| Scalar | Layer | Typical value |
|---|---|---|
| `Terrain_Scan/frame_range_mean` | ② | 0.07 – 0.09 |
| `Terrain_Scan/frame_std_mean` | ② | 0.019 – 0.026 |
| `Terrain_Scan/frame_range_<sub_terrain>` | ② | stairs 0.14 – 0.20 |
| `Terrain_Scan/range_mean` | ③ | 0.13 – 0.30 |
| `Terrain_Scan/range_<sub_terrain>` | ③ | flat ≈ 0, rough 0.14 – 0.16, stairs **0.27 – 0.66** |

Pass criteria: `frame_range_flat ≈ 0` while `frame_range_pyramid_stairs > 0.05`; ordering
stairs > rough > slope > flat; `frame_range_mean` rises with `Curriculum/terrain_levels_vel/mean`;
no case where `range_mean ≈ 0` together with a high `Train/mean_reward` (reward hacking).

Playback (online debugging):

```bash
python scripts/rsl_rl/play.py --task=Go2-velocity-v0 --num_envs 8 --headless \
    --terrain_level 5 --scan_debug 15 --max_steps 300 \
    --checkpoint logs/rsl_rl/go2_Complex_terrain/<RUN>/model_XXXX.pt
```

Disable (read-only logging, does not affect training results):

```bash
LOG_TERRAIN_SCAN=0 python scripts/rsl_rl/train.py --task=Go2-velocity-v0 --num_envs 4096 --headless
```

Note on `RayCasterCfg.offset.pos`: the offset is applied to the **ray start**, not to the sensor
pose. `z = 20.0` only lifts the ray origins above the terrain; `mdp.height_scan` returns
`base_z - hit_z - 0.5` (ground height relative to the body, ≈ -0.23 m on flat ground) and does
**not** saturate.

### Key Curriculum / Observation Changes (retraining required)

| Change | Where | Notes |
|---|---|---|
| Terrain curriculum switch fixed | `go2_demo_velocity.py` | Checked `curriculum.terrain_levels` (never exists) → `terrain_generator.curriculum` stayed `False`, terrains were random in difficulty; now also accepts `terrain_levels_vel` |
| Promotion / demotion uses accumulated path length | `mdp/curriculums.py` + `aer_env.py` | `use_path_length=True` uses `∫|v_xy|dt`. Displacement-based metrics (Euclidean or projected) break down when the command is resampled every 10 s: the net displacement cancels out and robots that keep walking get demoted (measured demotion rate 81% before the fix) |
| Ratio-based thresholds | `mdp/curriculums.py` | `up_ratio=0.5`, `down_ratio=0.25`, floored at half / quarter of a terrain block (8 m → 4 m / 2 m) |
| Terrain-level checkpointing | `mdp/curriculums.py` + `train.py` | Writes `terrain_levels.pt`; `--resume` restores it so the curriculum does not drop back to level 0 |
| Tighter velocity tracking | `go2_demo_velocity.py` | `track_lin_vel_xy` std `sqrt(0.25)=0.5` → `sqrt(0.1)≈0.316` |
| Lateral velocity command | `go2_demo_velocity.py` | `lin_vel_y` `(0,0)` → `(-0.1, 0.1)` |
| Terrain variation logging | `aer_env.py` + `play.py` | Per-episode height-envelope scalars `Terrain_Scan/*`; `play --scan_debug N` prints them |

### Playback at a Fixed Curriculum Level

```bash
python scripts/rsl_rl/play.py --task=Go2-velocity-v0 --num_envs 8 \
    --terrain_level 5 --terrain_name pyramid_stairs \
    --scan_debug 15 --max_steps 45 --headless \
    --checkpoint logs/rsl_rl/go2_Complex_terrain/<RUN>/model_XXXX.pt
```

| Argument | Meaning |
|---|---|
| `--terrain_level` | Pins the curriculum level (terrain **row**, 0-9) and freezes the curriculum |
| `--terrain_name` | Pins the terrain type by **name** (e.g. `pyramid_stairs`); comma-separated values are assigned round-robin, `env_0` always gets the first one |
| `--terrain_type` | Pins the terrain type by **column index** (comma-separated, e.g. `10,14`) |
| `--scan_debug N` | Prints height-scanner statistics every N steps |
| `--max_steps N` | Exits automatically after N steps (headless regression runs) |

**Why `--terrain_level` alone never shows stairs**: it only pins the row (difficulty). The column
(terrain type) is still whatever `terrain_types` was initialised to; with `--num_envs 4` the default
columns are `0/5/10/15`, and since the viewport camera follows **env_0 only** (`viewer.env_index = 0`),
you see column `0 = flat`. Use `--terrain_name`/`--terrain_type` to force the column — the script
assigns it round-robin so `env_0` lands on the requested terrain.

| Terrain type | Column (col) |
|---|---|
| flat | 0 ~ 3 |
| smooth_slope | 4 ~ 5 |
| smooth_slope_inv | 6 ~ 7 |
| random_rough | 8 ~ 9 |
| **pyramid_stairs** | **10 ~ 13** |
| **pyramid_stairs_inv** | **14 ~ 16** |
| discrete_obstacles | 17 ~ 19 |

### Dummy Agent Testing
```bash
# Zero-action agent (verify environment setup)
python scripts/zero_agent.py --task=<TASK_NAME>

# Random-action agent
python scripts/random_agent.py --task=<TASK_NAME>
```

---

## Project Structure

```
<project-root>/
├── scripts/
│   ├── verify_height_scanner.py  # Verify the height scanner can sense terrain changes
│   └── rsl_rl/
│       ├── train.py          # Training entry point (+ RSL-RL v5.x compat layer)
│       └── play.py           # Playback / evaluation script (--terrain_level support)
└── source/
    └── go2_demo/
        └── go2_demo/
            ├── assets/
            │   └── robot/unitree.py       # Go2 robot asset config
            └── tasks/manager_based/go2_demo/
                ├── go2_demo_velocity.py   # Main env config (terrains, obs, rewards)
                ├── aer_env.py             # AER reward manager
                └── mdp/
                    ├── curriculums.py     # Terrain & velocity curriculum
                    └── rewards.py         # Custom reward functions
```
