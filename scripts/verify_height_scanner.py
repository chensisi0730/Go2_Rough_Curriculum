# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""验证 height_scanner（高度扫描传感器，height scanner）能否真正"看到"地形变化。

本脚本不加载策略，机器人保持默认站姿（零动作，zero action），只做三件事：

1. 把环境强制放到指定 terrain_level（地形等级，terrain level = 地形网格的 row 行索引）
   和 terrain_type（地形类型，terrain type = 地形网格的 col 列索引）上；
2. 统计 height_scanner 的原始射线命中点（raw ray hits，世界系 z）与观测项
   height_scan（高度扫描观测，即 mdp.height_scan 的输出，真正送进策略的量）；
3. 判定：
   - 观测项是否被 clip（裁剪）饱和成常数 —— 若饱和，策略等于"没有地形感知"；
   - 观测项内部起伏（单 env 内 187 条射线的标准差）能否区分地形类型与难度。

用法：
    python scripts/verify_height_scanner.py --task=Go2-velocity-v0 --headless
    python scripts/verify_height_scanner.py --task=Go2-velocity-v0 --headless --levels 0,3,6,9 --steps 40
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify whether the height scanner can sense terrain changes.")
parser.add_argument("--task", type=str, default="Go2-velocity-v0", help="Name of the task.")
parser.add_argument("--levels", type=str, default="0,3,6,9", help="Comma separated terrain levels (rows) to test.")
parser.add_argument("--steps", type=int, default=40, help="Number of settle steps before sampling.")
parser.add_argument("--num_samples", type=int, default=5, help="Number of sampled steps to average over.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 清空 sys.argv 中已消费的参数，避免 Hydra 误把本脚本参数当作 override
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_demo.tasks  # noqa: F401
from go2_demo.tasks.manager_based.go2_demo import mdp


def sub_terrain_names(gen_cfg) -> list[str]:
    """复现 IsaacLab TerrainGenerator._generate_curriculum_terrains 的「列 -> 子地形」映射。"""
    names = list(gen_cfg.sub_terrains.keys())
    proportions = np.array([cfg.proportion for cfg in gen_cfg.sub_terrains.values()], dtype=np.float64)
    proportions /= np.sum(proportions)
    cumsum = np.cumsum(proportions)
    out = []
    for col in range(gen_cfg.num_cols):
        idx = int(np.min(np.where(col / gen_cfg.num_cols + 0.001 < cumsum)[0]))
        out.append(names[idx])
    return out


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    levels = [int(x) for x in args_cli.levels.split(",") if x.strip() != ""]
    gen_cfg = env_cfg.scene.terrain.terrain_generator
    num_cols = gen_cfg.num_cols
    col_names = sub_terrain_names(gen_cfg)
    order = list(dict.fromkeys(col_names))  # 去重后的地形类型顺序

    # 每个 (level, col) 组合分配一个 env，保证地形类型 x 地形等级全覆盖
    num_envs = len(levels) * num_cols
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = 0  # 关闭随机化，保证唯一变量是地形
    env_cfg.observations.policy.enable_corruption = False
    orig_clip = tuple(env_cfg.observations.policy.height_scanner.clip)  # 原始裁剪范围，用于判断饱和
    env_cfg.observations.policy.height_scanner.clip = None  # 关掉裁剪，直接看原始观测
    env_cfg.scene.height_scanner.debug_vis = False

    with gym.make(args_cli.task, cfg=env_cfg) as gym_env:
        env = gym_env.unwrapped
        terrain = env.scene.terrain
        sensor = env.scene.sensors["height_scanner"]

        # 强制指定每个 env 的 terrain level / terrain type
        lv = torch.tensor([[l] * num_cols for l in levels], device=env.device).reshape(-1)
        tp = torch.tensor([list(range(num_cols))] * len(levels), device=env.device).reshape(-1)
        terrain.terrain_levels[:] = lv
        terrain.terrain_types[:] = tp
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]

        # 冻结课程，避免测试过程中地形等级被 curriculum 改写
        env.curriculum_manager.active_terms.clear()

        env.reset()

        actions = torch.zeros(env.action_space.shape, device=env.device)
        sensor_cfg = SceneEntityCfg("height_scanner")
        for _ in range(args_cli.steps):
            env.step(actions)

        raw_list, scan_list = [], []
        for _ in range(args_cli.num_samples):
            env.step(actions)
            raw_list.append(sensor.data.ray_hits_w[..., 2].clone())
            scan_list.append(mdp.height_scan(env, sensor_cfg=sensor_cfg).clone())

        raw = torch.stack(raw_list).mean(0)  # (num_envs, num_rays) 射线命中点世界 z
        scan = torch.stack(scan_list).mean(0)  # (num_envs, num_rays) 送进策略的观测值

        robot = env.scene["robot"]
        root_z = robot.data.root_pos_w[:, 2]
        sensor_z = sensor.data.pos_w[:, 2]
        nan_inf = int((~torch.isfinite(raw)).sum())
        off_xy = torch.linalg.norm(robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)

        # ---- 明细表 ----
        print("\n" + "=" * 122)
        print(
            f"{'地形类型':<20}{'col':>4}{'level':>7}{'scan_mean':>11}{'起伏(std)':>11}"
            f"{'scan_min':>10}{'scan_max':>10}{'robot离原点(m)':>14}{'非有限值':>9}"
        )
        print("=" * 122)
        table = {}
        for li, level in enumerate(levels):
            for col in range(num_cols):
                idx = li * num_cols + col
                s = scan[idx]
                name = col_names[col]
                table[(level, name)] = float(s.std())
                print(
                    f"{name:<20}{col:>4}{level:>7}{float(s.mean()):>11.4f}{float(s.std()):>11.4f}"
                    f"{float(s.min()):>10.3f}{float(s.max()):>10.3f}{float(off_xy[idx]):>14.3f}{nan_inf:>9d}"
                )
        print("=" * 122)

        # ---- 结论 1：传感器位姿（pose）与观测偏移 ----
        pat = sensor.cfg.pattern_cfg
        off = sensor.cfg.offset.pos
        print("\n[1] 传感器配置与观测偏移")
        print(f"  pattern size={tuple(pat.size)}, resolution={pat.resolution} -> {scan.shape[1]} 条射线")
        print(f"  offset.pos={tuple(off)}（注意：该 offset 加在射线起点 ray start 上，不是传感器位姿）")
        print(f"  sensor_z - robot_root_z = {float((sensor_z - root_z).mean()):.3f} m "
              f"（不含 offset.z，说明 z 只抬高了射线起点）")
        print(f"  观测值 scan = base_z - hit_z - 0.5，全局取值 [{float(scan.min()):.3f}, {float(scan.max()):.3f}]")
        clipped_max = float(orig_clip[1])
        frac_sat = float((scan >= clipped_max - 1e-3).float().mean())
        print(f"  贴上界(clip 上界={clipped_max})的比例 = {frac_sat:.2%}；"
              f"贴下界({orig_clip[0]})比例 = {float((scan <= float(orig_clip[0]) + 1e-3).float().mean()):.2%}")

        # ---- 结论 2：能否区分地形类型（同一 level 下不同地形的脚下起伏）----
        print(f"\n[2] 单 env 脚下 {scan.shape[1]} 条射线的标准差（起伏，越大说明越能看见地形结构）——按地形类型")
        header = f"  {'level':<7}" + "".join(f"{n[:14]:>16}" for n in order)
        print(header)
        for level in levels:
            line = f"  {level:<7}" + "".join(f"{table[(level, n)]:>16.4f}" for n in order)
            print(line)

        # ---- 结论 3：能否区分难度（同一地形类型随 level 的起伏变化）----
        print("\n[3] 同一地形类型随地形等级（level）的起伏变化")
        for name in order:
            seq = [table[(level, name)] for level in levels]
            print(f"  {name:<20}" + "".join(f"L{l}={v:.4f}  " for l, v in zip(levels, seq))
                  + ("  <-- 随难度上升" if seq[-1] > seq[0] * 1.5 + 1e-3 else ""))

        # ---- 总结判定 ----
        print("\n[结论]")
        flat_std = min(table.get((levels[0], n), 0.0) for n in order)
        max_std = max(table.values())
        print(f"  * 观测项 clip 饱和：{'FAIL（策略拿不到地形信息）' if frac_sat > 0.5 else 'PASS（未饱和，保留地形起伏）'}")
        print(f"  * 能否检测到地形变化：{'PASS' if max_std > 0.01 else 'FAIL'} "
              f"（起伏从 {flat_std:.4f} m 到 {max_std:.4f} m）")
        print(f"  * 观测几何：视场 {tuple(pat.size)} m，射线起点 x 偏移 {off[0]:.2f} m -> "
              f"向机体前方最多可看到 {off[0] + pat.size[0] / 2:.2f} m；")
        print("    楼梯地形 platform_width=3.0 m（半宽 1.5 m），机器人 reset 在地块中心时，"
              "若扫描窗口前方覆盖距离 < 1.5 m 就扫不到任何一级台阶，脚下起伏会退化到 ≈0。")
        print("    注意：这里测的是静止站立值，机器人行走后实际包络更大（见 TensorBoard 的 Terrain_Scan/*）。")
        print()


if __name__ == "__main__":
    main()
    simulation_app.close()
