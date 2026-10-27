# 复杂地形速度跟随步态训练

> 📄 [English Documentation](README.md)

## 项目概述

本项目基于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) v2.3.2，使用 PPO 算法（[RSL-RL](https://github.com/leggedrobotics/rsl_rl) 框架）训练宇树 Go2 四足机器人在复杂地形中的速度跟随运动控制策略。

项目的核心目标是训练出可以泛化到真实世界各种地形（平地、斜坡、台阶、粗糙地面等）的通用步态控制器，而非仅针对单一理想地形的过拟合策略。

### 训练演示

![](assets/demo.gif)

---

## 核心特性

### 🏔️ 混合地形训练 · 提升真实世界泛化能力

智能体在程序化生成的复合地形（`COBBLESTONE_ROAD_CFG`）中训练，七种地形类型在同一场景中混合出现：

| 地形类型 | 占比 | 说明 |
|---|---|---|
| 平坦地面 | 20% | 基础步行 |
| 平滑斜坡 | 15% | 上坡 / 下坡 |
| 反向斜坡 | 10% | 倒置金字塔斜面 |
| 随机粗糙面 | 15% | 随机高度场，模拟碎石路 |
| 金字塔台阶 | 20% | 规则台阶攀爬 |
| 反向台阶 | 20% | 下行台阶地形 |
| 离散障碍物 | 可选 | 散布障碍物 |

在多样化地形中同时训练，可以引导策略学习出鲁棒的、可迁移的运动行为，从而在实际部署中取得更好的泛化效果。

---

### 📈 课程学习 · 逐步提升训练效率

为了防止策略被过难任务淹没，同时充分挖掘学习潜力，项目采用了两级课程学习机制：

#### 1. 地形难度课程（`terrain_levels_vel`）

机器人根据每个 episode 内的行走距离动态调整地形难度：
- **走得足够远** → 晋升到更难地形
- **走得不足一半** → 降级到更简单地形

这种机制使训练初期从简单地形开始积累基础能力，再逐步接触高难度场景。

#### 2. 速度指令课程（`lin_vel_cmd_levels`）

速度指令范围随策略表现动态扩展：
- 初始阶段速度范围保守（如 0.2 ~ 1.2 m/s）
- 当跟踪奖励达到阈值后，自动扩大速度范围（最高可达 6.0 m/s）

这使策略能够循序渐进地学习从慢走到快跑的全速域控制。

---

### 🔄 迁移学习 · 加载平地预训练模型

在复杂地形上从零训练难度极大。本项目采用**两阶段训练策略**：

- **第一阶段**：在平坦地形上训练，让策略先掌握基础步态节律、平衡控制和速度跟踪能力。
- **第二阶段**：将平地训练的 checkpoint 作为预训练权重加载，在混合地形上微调，快速迁移已有能力并适应复杂场景。

加载预训练模型的训练命令：

```bash
python scripts/rsl_rl/train.py \
    --task=<TASK_NAME> \
    --num_envs=4096 \
    --pretrained=/path/to/flat_terrain/model_XXXX.pt
```

两阶段策略相比从零训练可以显著缩短收敛时间，并降低策略陷入局部最优的风险。

---

### 🔧 RSL-RL v5.x 兼容适配层

RSL-RL v5.0 对配置 API 和 checkpoint 格式进行了重大重构，与旧版不兼容。为了在不修改 Isaac Lab 官方模板的前提下支持新版本，本项目在 `scripts/rsl_rl/train.py` 中实现了一套本地兼容适配层。

#### 配置格式适配（`handle_deprecated_rsl_rl_cfg`）

自动将旧版 `policy` 字段格式转换为 v5.x 所需的 `actor` / `critic` 分离格式：

```python
# 旧版格式（< v5.0）：
cfg["policy"] = {
    "actor_hidden_dims": [512, 256, 128],
    "init_noise_std": 1.0,
    ...
}

# 新版格式（v5.x）：
cfg["actor"] = {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], ...}
cfg["critic"] = {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], ...}
```

同时自动处理 `GaussianDistribution` / `HeteroscedasticGaussianDistribution` 等分布配置的映射。

#### Checkpoint 格式适配（`handle_deprecated_rsl_rl_checkpoint`）

自动将旧版扁平权重文件（单一 `model_state_dict`）转换为 v5.x 所需的 `actor_state_dict` / `critic_state_dict` 分离格式，并对 `MLPModel` 内部键名做精确重映射：

| 旧版权重键名 | v5.x MLPModel 期望键名 |
|---|---|
| `"0.weight"` | `"mlp.0.weight"` |
| `"0.bias"` | `"mlp.0.bias"` |
| `"std"` | `"distribution.std_param"` |
| `"actor.X.weight"` | `"mlp.X.weight"` |

以上适配均在运行时自动完成，无需手动转换权重文件。

---

## AER 奖励设计

本项目实现了论文中的 **AER（Adaptive Exponential Reward，自适应指数奖励）** 结构。有别于线性加权求和，负向惩罚项通过指数门控乘到正向奖励上：

```
R_总奖励 = R_正向 × exp(R_负向 / σ)
```

- **正向项**：速度跟踪（线速度、角速度）、能量效率奖励
- **负向项**：关节力矩过大、关节速度过大、关节加速度过大、动作变化率过大、足部打滑、躯干姿态偏差、机身高度偏差等

这种设计在任务完成度和运动质量之间形成自然的权衡机制，而非简单的线性累加，能更有效地引导策略同时优化运动效果和物理合理性。

---

## 安装

**前置条件**：已安装 Isaac Lab v2.3.2，推荐使用 conda 环境。

```bash
# 1. 克隆项目（需在 IsaacLab 目录之外）
git clone <repo-url>
cd <project-dir>

# 2. 激活 conda 环境并安装项目扩展
conda activate go2_rough_curriculum
python -m pip install -e source/go2_demo

# 3. 验证安装
python scripts/list_envs.py
```

---

## 训练与评估

### 标准训练（混合地形，推荐命令）

```bash
cd /home/css/work/unitree/rl/zhouzhiguang_repo/my_branchh/Go2_Rough_Curriculum
source ~/miniconda3/etc/profile.d/conda.sh && conda activate go2_rough_curriculum
export OMNI_KIT_ACCEPT_EULA=1

# 训练（14096 个环境，无头模式；--run_name 会拼在 logs 目录名后面，便于区分实验）
python scripts/rsl_rl/train.py \
    --task=Go2-velocity-v0 \
    --num_envs 10096 \
    --headless \
    --run_name curriculum_fix_v2
```

长时间训练建议放后台并重定向日志：

```bash
nohup python scripts/rsl_rl/train.py \
    --task=Go2-velocity-v0 --num_envs 14096 --headless \
    --run_name curriculum_fix_v2 > train_curriculum_v2.log 2>&1 &
```

监控曲线：

```bash
tensorboard --logdir logs/rsl_rl   # 必须是 logs/rsl_rl 本身，见下方 TEXT 说明
```

> 当前默认配置（`go2_demo_velocity.py`）：地形课程按难度递增生成（`terrain_generator.curriculum=True`），
> 高度扫描 187 条射线（观测维度 **247**，与历史实验一致），`lin_vel_y ∈ [-0.1, 0.1]`，
> 地形等级自动存档到 `logs/rsl_rl/go2_Complex_terrain/<RUN>/terrain_levels.pt`，
> 每个 episode 记录地形变化范围 `Terrain_Scan/*`。
> `track_lin_vel_xy` 的 `std` 已收紧、`lin_vel_y` 已放开，这两项会改变训练动力学，
> 与 2026-10-26 的历史 checkpoint 相比属于**不同实验组**，建议重训。

### 基于预训练模型的迁移学习训练

```bash
python scripts/rsl_rl/train.py \
    --task=<TASK_NAME> \
    --num_envs=4096 \
    --pretrained=/path/to/flat_terrain/model_XXXX.pt
```

### 加载 checkpoint 继续训练（--resume）

```bash
python scripts/rsl_rl/train.py \
    --task=<TASK_NAME> \
    --resume
```

### 评估 / 可视化推理

```bash
cd /home/css/work/unitree/rl/zhouzhiguang_repo/my_branchh/Go2_Rough_Curriculum
source ~/miniconda3/etc/profile.d/conda.sh && conda activate go2_rough_curriculum
export OMNI_KIT_ACCEPT_EULA=1

# ① 指定 checkpoint 文件
python scripts/rsl_rl/play.py --task=Go2-velocity-v0 --num_envs 4 \
    --checkpoint logs/rsl_rl/go2_Complex_terrain/2026-10-26_20-42-22/model_8500.pt

# ② 只给 run 目录，自动取最新 checkpoint
python scripts/rsl_rl/play.py --task=Go2-velocity-v0 --num_envs 4 \
    --load_run 2026-10-27_20-53-29_warmstart_8500_v2

# ③ 指定课程等级 + 地形类型（推荐用 --terrain_name，不用记列号）
# 只看上台阶：4 只狗全部在楼梯，镜头跟随的 env_0 也在楼梯
python scripts/rsl_rl/play.py --task=Go2-velocity-v0 --num_envs 4 \
    --terrain_level 5 --terrain_name pyramid_stairs \
    --load_run 2026-10-27_20-53-29_warmstart_8500_v2

# 上下台阶一起看，env_0 = 上台阶
python scripts/rsl_rl/play.py --task=Go2-velocity-v0 --num_envs 4 \
    --terrain_level 5 --terrain_name pyramid_stairs,pyramid_stairs_inv \
    --load_run 2026-10-27_20-53-29_warmstart_8500_v2


# ④ 无头模式 + 录制视频（不开窗，适合远程 / SSH）
python scripts/rsl_rl/play.py --task=Go2-velocity-v0 --num_envs 4 --headless \
    --video --video_length 400 \
    --checkpoint logs/rsl_rl/go2_Complex_terrain/2026-10-26_20-42-22/model_8500.pt
```

#### 为什么只加 `--terrain_level` 看不到楼梯

`--terrain_level` 只决定**行（row = 难度）**，**列（col = 地形类型）**仍由初始化时的
`terrain_types` 决定。`--num_envs 4` 时默认列是 `0 / 5 / 10 / 15`
（flat 平地 / smooth_slope 斜坡 / pyramid_stairs 上台阶 / pyramid_stairs_inv 下台阶），
虽然其中两个是台阶，但**视口相机只跟随 env_0**（`viewer.env_index = 0`），而 env_0 的列是 `0 = flat`，
所以画面里永远是平地。

解决：用 `--terrain_type`（列号）或 `--terrain_name`（地形名）指定列。脚本会把目标列
**循环分配给所有 env**，保证 `env_0` 一定落在第一个目标地形上：

```bash
# 只看上台阶，4 只狗都在楼梯上
--terrain_level 5 --terrain_name pyramid_stairs

# 同时看上下台阶，env_0 = 上台阶
--terrain_level 5 --terrain_name pyramid_stairs,pyramid_stairs_inv

# 等价写法：用列号（可逗号分隔多个）
--terrain_level 5 --terrain_type 10,14
```

#### 地形列（col）映射表（当前配置 `num_cols=20`）

| 地形类型（terrain type） | 列 col | 说明 |
|---|---|---|
| flat | 0 ~ 3 | 平坦地面 |
| smooth_slope | 4 ~ 5 | 平滑上坡 |
| smooth_slope_inv | 6 ~ 7 | 平滑下坡（倒金字塔斜面） |
| random_rough | 8 ~ 9 | 随机粗糙面 |
| **pyramid_stairs** | **10 ~ 13** | **上台阶（金字塔台阶）** |
| **pyramid_stairs_inv** | **14 ~ 16** | **下台阶（倒金字塔台阶）** |
| discrete_obstacles | 17 ~ 19 | 离散障碍物 |

> 该映射由 `proportion` 归一化后按 IsaacLab `_generate_curriculum_terrains` 的规则计算，
> 修改 `COBBLESTONE_ROAD_CFG` 里任何 `proportion` 都会改变列分布，届时用
> `scripts/verify_height_scanner.py` 打印的表格可自查最新的列 → 地形名对应关系。

### 检查训练曲线（TensorBoard，张量板）

```bash
tensorboard --logdir logs/rsl_rl/go2_Complex_terrain
```

重点看这几条曲线（指标名称 → 含义）：

| 曲线 | 含义 | 健康表现 |
|---|---|---|
| `Curriculum/terrain_levels_vel`（mean，均值） | 平均地形等级（0~9，行 row） | 持续上升，最终 > 7 |
| `Curriculum/terrain_levels_vel/up`、`/down` | 升级率 / 降级率 | up ≈ down 说明课程进入平衡，长期 down > up 说明卡住 |
| `Curriculum/terrain_levels_vel/level_<地形名>` | 每种地形类型（terrain type，列 col）的平均等级 | 各类型差距 < 2，否则说明某类地形没人走得动 |
| `Curriculum/terrain_levels_vel/up_<地形名>` | 每种地形的升级率 | 接近 0 的地形就是瓶颈地形 |
| `Train/mean_reward`（平均回报） | 总回报（AER 自适应指数奖励，adaptive exponential reward） | 上升后应继续缓慢上升；走平即饱和 |
| `Episode_Reward/track_lin_vel_xy`（线速度跟踪） | 线速度跟踪奖励 | 上限 = weight（1.0），长时间 ≈ 0.96 表示已饱和 |
| `Episode_Termination/bad_orientation`（姿态异常终止） | 姿态超限终止率 | < 0.01 |
| `Episode_Termination/base_contact`（机身触地终止） | 躯干触地终止率 | ≈ 0 |
| `Terrain_Scan/range_mean` | **地形变化范围**：一个 episode 内 height_scanner 射线命中高度的包络（peak-to-peak，最大−最小），单位 m | 随地形等级上升，0.05 ~ 0.30 |
| `Terrain_Scan/range_<地形名>` | 每种地形类型的地形变化范围（`range_pyramid_stairs` 等） | 台阶类最大（0.15~0.25），平地 ≈ 0 |
| `Terrain_Scan/range_{min,max}`、`num_envs` | 本批 episode 的最小/最大起伏、参与统计的 env 数 | 用于判断方差是否异常 |

> `Terrain_Scan/*` 由 `AERManagerBasedRLEnv`（`aer_env.py`）在**每个 episode 结束时**汇总写入，
> 记录的是「整个 episode 里 height_scanner 看到过的地形高度包络」，因此机器人走动起来后数值会明显大于
> 静止站在出生点时的瞬时值。设环境变量 `LOG_TERRAIN_SCAN=0` 可关闭该日志（消融对照用）。

> 注：`log_per_terrain=True`（已默认开启）只新增日志，不改变课程行为，不影响消融对照。

#### 课程与观测的关键改动（需重训，旧 checkpoint 不兼容）

| 改动 | 位置 | 说明 |
|---|---|---|
| 地形课程开关修复 | `go2_demo_velocity.py` | 原来检查 `curriculum.terrain_levels`（名字不存在）→ `terrain_generator.curriculum` 永远 False，地形按随机难度生成；现改为同时识别 `terrain_levels_vel` |
| 升降级改用累积路程 | `mdp/curriculums.py` + `aer_env.py` | `use_path_length=True`：用 `∫|v_xy|dt` 累积路程。位移类判据（欧氏或投影）在「指令每 10 s 重采样」时会因正负抵消而失真，实测修复前降级率高达 81% |
| 升降级阈值比例化 | `mdp/curriculums.py` | 升级 `up_ratio=0.5`、降级 `down_ratio=0.25`，且至少半块 / 四分之一块地形（8 m → 4 m / 2 m），消除原来「4~12 m 死区」 |
| 地形等级存档 / 恢复 | `mdp/curriculums.py` + `train.py` | 训练时写 `terrain_levels.pt`，`--resume` 自动读回，课程不掉级 |
| 地形变化范围记录 | `aer_env.py` + `play.py` | 每个 episode 汇总 height_scanner 的高度包络到 `Terrain_Scan/*`；`play --scan_debug N` 同时打印这些值 |
| 速度跟踪收紧 | `go2_demo_velocity.py` | `track_lin_vel_xy` 的 `std` 由 `sqrt(0.25)=0.5` 收紧到 `sqrt(0.1)≈0.316`，避免早饱和 |
| 侧向速度指令 | `go2_demo_velocity.py` | `lin_vel_y` 由 `(0,0)` 改为 `(-0.1, 0.1)`，训练侧移能力 |

### 测试方法：验证 height_scanner 能否「看到」地形变化

分两级测试：**A 静态**（出生点站立，验证传感器本身）与 **B 动态**（训练 / play 中，验证实际训练时的感知）。
两级测出来的数值差 10 倍以上属于正常，因为度量的是不同时间尺度。

#### 三层指标不要混用

| 层级 | 指标来源 | 统计方式 | 台阶类实测 |
|---|---|---|---|
| ① 静态站立 | `verify_height_scanner.py` | 出生点、单帧、187 条射线的 std | **0.000 ~ 0.024 m** |
| ② 动态单帧 | `Terrain_Scan/frame_range_*` | 每走一步算一次单帧落差，再对 episode 求平均 | **0.14 ~ 0.20 m** |
| ③ 动态累积 | `Terrain_Scan/range_*` | 整个 episode 所有帧的全局 `max − min` | **0.27 ~ 0.66 m** |

> ③ 之所以高达 0.66 m：机器人 20 s 走约 14 m，而每块地形只有 8 m，会跨越 1~2 个地块，
> 叠加金字塔台阶 6 级 × 0.055 m ≈ 0.33 m 的累计落差。它衡量的是「这段经历里地形变化有多大」，
> **不是**「传感器分辨率」。要判断传感器看得清不清楚，请看 ① 或 ②。

#### 测试 A：静态（传感器本身是否有效）

```bash
python scripts/verify_height_scanner.py --task=Go2-velocity-v0 --headless --levels 0,3,6,9
```

脚本把仿真强制排布成「地形等级（level）× 地形类型（terrain type）」网格，机器人零动作站立，
输出每个地块脚下 187 条射线的起伏。实测（当前配置 `size=[1.6,1.0]`、`offset=(0,0,20)`）：

| 地形 | level 0 | level 3 | level 6 | level 9 |
|---|---|---|---|---|
| flat 平坦 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| smooth_slope 上坡 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| smooth_slope_inv 下坡 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_rough 粗糙 | 0.0212 | 0.0222 | 0.0215 | 0.0218 |
| pyramid_stairs 上台阶 | 0.0078 | 0.0096 | 0.0000 | 0.0000 |
| pyramid_stairs_inv 下台阶 | 0.0056 | 0.0000 | 0.0199 | 0.0242 |
| discrete_obstacles 障碍 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

观测值全局范围 `[-0.384, -0.092]`，clip 命中率 0%（未饱和）。

**判定标准**

| 检查项 | PASS 条件 | 不满足时的处理 |
|---|---|---|
| 观测未饱和 | `height_scan` 贴 clip 边界比例 < 1% | 检查 `PatternCfg`/`offset` 是否让射线跑到地形外 |
| 能区分噪声地形 | `random_rough` ≥ 0.01 且 `flat` ≈ 0 | 射线未命中地形，检查 `mesh_prim_paths` |
| 台阶/障碍可见 | 至少部分地块 > 0.01 | 出生点被大平台包住，见下方「已知盲区」 |

**已知盲区**：台阶地形 `platform_width=3.0 m`（半宽 1.5 m），而视场向前只覆盖 0.8 m，
机器人 reset 在地块中心时扫不到台阶边缘，静态值退化到 ≈0。这不影响训练——机器人一走起来就到了 ②/③ 的量级。

**关于 `RayCasterCfg.offset.pos` 的正确理解**（容易踩坑）：
该 offset 加在**射线起点（ray start）**上，不是加在传感器位姿上。因此
`offset.pos.z = 20.0` 只是把射线起点抬到地形之上 20 m（保证不会从地形内部起射），
`mdp.height_scan` 输出 = `base_z - hit_z - 0.5`，语义是「地面相对机体的高度」，
平地约 -0.23 m，**不会**因为 z=20 而饱和。

#### 测试 B：动态（训练 / play 中实际感知）

训练和 play 都会自动记录，不需要额外脚本，直接看 TensorBoard（张量板）的 `Terrain_Scan` 分组。

| 曲线 | 层级 | 含义 | 实测典型值 |
|---|---|---|---|
| `Terrain_Scan/frame_range_mean` | ② 单帧 | 每步单帧落差（max−min）的平均 | 0.07 ~ 0.09 |
| `Terrain_Scan/frame_std_mean` | ② 单帧 | 每步单帧 std 的平均 | 0.019 ~ 0.026 |
| `Terrain_Scan/frame_range_<地形名>` | ② 单帧 | 每种地形的单帧落差，如 `frame_range_pyramid_stairs` | 台阶 0.14 ~ 0.20 |
| `Terrain_Scan/range_mean` | ③ 累积 | episode 内全局包络均值 | 0.13 ~ 0.30 |
| `Terrain_Scan/range_min` / `range_max` | ③ 累积 | 本批 episode 的最小 / 最大包络 | max 可达 1.1 |
| `Terrain_Scan/range_flat` | ③ 累积 | 平坦地形 | 0.000 ~ 0.005 |
| `Terrain_Scan/range_smooth_slope` / `_inv` | ③ 累积 | 上 / 下斜坡 | 0.12 ~ 0.13 |
| `Terrain_Scan/range_random_rough` | ③ 累积 | 随机粗糙面 | 0.14 ~ 0.16 |
| `Terrain_Scan/range_pyramid_stairs` / `_inv` | ③ 累积 | 上 / 下台阶 | **0.27 ~ 0.66** |
| `Terrain_Scan/range_discrete_obstacles` | ③ 累积 | 离散障碍 | 0.10 ~ 0.15 |
| `Terrain_Scan/num_envs` | — | 参与统计的 env 数（同一批 reset 的个数） | — |

**判定标准**

| 检查项 | PASS 条件 |
|---|---|
| 传感器正常工作 | `frame_range_flat` ≈ 0 且 `frame_range_pyramid_stairs` > 0.05 |
| 能区分地形类型 | `frame_range_<地形>` 之间有明显排序：台阶 > 粗糙 > 斜坡 > 平坦 |
| 感知随难度增强 | `Curriculum/terrain_levels_vel/mean` 上升时，`frame_range_mean` 单调上升 |
| 无刷分嫌疑 | 不允许出现 `range_mean` 长期 ≈ 0（机器人原地不动）而 `Train/mean_reward` 很高 |

**四步诊断流程**

1. **确认在看地形**：`frame_range_flat ≈ 0`、`frame_range_pyramid_stairs` 明显 > 0 → 正常；
   两者都 ≈ 0 → 扫描窗口完全没覆盖到结构（检查 `PatternCfg.size` 与 `platform_width`）。
2. **对照课程**：`frame_range_mean` 应随 `terrain_levels_vel/mean` 同步上升；
   若等级上升但单帧值不变，说明高等级地块的结构没进入视场。
3. **定位瓶颈地形**：某地形的 `range_<地形>` 低、`up_<地形>` 也接近 0 → 该地形是瓶颈
   （既看不到结构也升不上去）。
4. **查刷分**：`range_mean` 长期 ≈ 0 而回报高 → 机器人在平坦区刷分。

#### play 时查看（在线调试）

```bash
python scripts/rsl_rl/play.py --task=Go2-velocity-v0 --num_envs 8 --headless \
    --terrain_level 5 --scan_debug 15 --max_steps 300 \
    --checkpoint logs/rsl_rl/go2_Complex_terrain/<RUN>/model_XXXX.pt
```

`--scan_debug N` 每 N 步打印一次瞬时扫描统计 + 最近一批 episode 的 `Terrain_Scan/*`：

```
[SCAN] step=15 | raw_hit_z mean=0.012 std=0.370 ... | obs_scan mean=-0.174 std=0.015 ...
        range_mean=0.162  frame_range_mean=0.090  frame_std_mean=0.026 ...
```

#### 关闭 / 消融

```bash
LOG_TERRAIN_SCAN=0 python scripts/rsl_rl/train.py --task=Go2-velocity-v0 --num_envs 4096 --headless
```

该日志只读取传感器数据、不影响动作与奖励，关闭与否**不改变训练结果**，仅影响日志开销（约 1~2%）。

### play（可视化推理）时指定课程等级

```bash
# 固定地形等级 5（0~9），冻结课程，便于分难度评估
python scripts/rsl_rl/play.py --task=Go2-velocity-v0 --num_envs 8 \
    --terrain_level 5 \
    --checkpoint logs/rsl_rl/go2_Complex_terrain/<RUN>/model_XXXX.pt

# 再固定地形类型（列 col，0~19），只测试某一种地形
python scripts/rsl_rl/play.py --task=Go2-velocity-v0 --num_envs 8 \
    --terrain_level 5 --terrain_type 12

# 无头调试 + 打印高度扫描统计 + 跑满 45 步自动退出
python scripts/rsl_rl/play.py --task=Go2-velocity-v0 --num_envs 8 --headless \
    --terrain_level 5 --scan_debug 15 --max_steps 45 \
    --checkpoint logs/rsl_rl/go2_Complex_terrain/<RUN>/model_XXXX.pt
```

参数说明：`--terrain_level`（课程等级，即地形行 row）、`--terrain_type`（地形列 col）、
`--scan_debug N`（每 N 步打印一次 height_scanner 统计）、`--max_steps N`（跑满 N 步自动退出）。

### 调试用 Dummy Agent

```bash
# 零动作（验证环境配置是否正常）
python scripts/zero_agent.py --task=<TASK_NAME>

# 随机动作
python scripts/random_agent.py --task=<TASK_NAME>
```

---

## 项目结构

```
<project-root>/
├── README.md                  # 英文文档
├── README_zh.md               # 中文文档（本文件）
├── scripts/
│   ├── list_envs.py
│   ├── zero_agent.py
│   ├── random_agent.py
│   ├── verify_height_scanner.py  # 验证高度扫描传感器能否感知地形变化
│   └── rsl_rl/
│       ├── train.py           # 训练入口（含 RSL-RL v5.x 兼容层）
│       └── play.py            # 评估推理脚本（支持 --terrain_level 固定课程等级）
└── source/
    └── go2_demo/
        └── go2_demo/
            ├── assets/
            │   └── robot/unitree.py          # Go2 机器人资产配置
            └── tasks/manager_based/go2_demo/
                ├── go2_demo_velocity.py      # 主环境配置（地形/观测/奖励）
                ├── aer_env.py                # AER 奖励管理器
                └── mdp/
                    ├── curriculums.py        # 地形与速度课程
                    └── rewards.py            # 自定义奖励函数
```
