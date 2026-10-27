from __future__ import annotations

import os

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def _sub_terrain_col_names(terrain_gen_cfg) -> list[str]:
    """复现 IsaacLab ``TerrainGenerator._generate_curriculum_terrains`` 的「列 -> 子地形名」映射。

    课程地形是「行 = 难度等级（level），列 = 地形类型（type）」的网格布局，
    这里把列索引翻译成可读的地形名字，便于在 TensorBoard 上按地形分组看曲线。
    """
    names = list(terrain_gen_cfg.sub_terrains.keys())
    proportions = np.array([cfg.proportion for cfg in terrain_gen_cfg.sub_terrains.values()], dtype=np.float64)
    proportions /= np.sum(proportions)
    cumsum = np.cumsum(proportions)
    return [
        names[int(np.min(np.where(col / terrain_gen_cfg.num_cols + 0.001 < cumsum)[0]))]
        for col in range(terrain_gen_cfg.num_cols)
    ]


def _maybe_save_terrain_levels(env: ManagerBasedRLEnv, terrain: TerrainImporter, interval: int = 200) -> None:
    """周期性把地形等级（terrain levels）存档到 ``env.cfg.log_dir/terrain_levels.pt``。

    配合 ``scripts/rsl_rl/train.py --resume`` 使用：恢复训练时会从该文件读回地形等级，
    避免每恢复一次课程就从 0 重来（地形难度掉级）。
    """
    log_dir = getattr(env.cfg, "log_dir", None)
    if not log_dir:
        return
    count = getattr(env, "_terrain_curriculum_save_count", 0) + 1
    env._terrain_curriculum_save_count = count
    # 第一次就存一份（保证短训练也有存档），之后每 interval 次存一次
    if count != 1 and count % interval != 0:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        torch.save(terrain.terrain_levels.cpu(), os.path.join(log_dir, "terrain_levels.pt"))
    except OSError:
        pass


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    log_per_terrain: bool = False,
    up_ratio: float = 0.5,
    down_ratio: float = 0.25,
    use_path_length: bool = True,
    use_projection: bool = True,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Args:
        log_per_terrain: 为 ``True`` 时返回 ``dict``，按地形类型（terrain type，即地形网格的列 col）
            分别记录平均等级与升降级率，曲线名形如 ``Curriculum/terrain_levels_vel/level_<sub_terrain>``；
            为 ``False`` 时保持原行为，只返回地形等级均值（兼容历史实验曲线）。
        up_ratio: 升级（promotion）阈值比例，要求实际行进距离 >= ``up_ratio * 指令期望距离``。
        down_ratio: 降级（demotion）阈值比例，实际行进距离 < ``down_ratio * 指令期望距离`` 则降级。
        use_path_length: 为 ``True``（默认）时用「累积路程 ∫|v_xy|dt」作为行进距离。
            这是唯一在「指令每 10 s 重采样」的任务里可靠的度量：位移类指标会在换向后
            因正负抵消而接近 0，导致机器人明明在走却被判成没走动（实测降级率高达 81%）。
        use_projection: 仅在 ``use_path_length=False`` 的回退路径下生效，
            用「沿指令方向的投影位移」代替「欧氏直线位移」。

    Returns:
        The mean terrain level for the given environment ids (or a dict of logging entries).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    # 统一索引（curriculum manager 可能传 slice 或 tensor）
    if isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)
    else:
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)

    cmd_xy = command[ids, :2]
    cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
    # 指令要求走完的距离 = 指令速度 * episode 时长
    expected_dist = cmd_speed * env.max_episode_length_s

    # ---- 行进距离的度量：优先用「累积路程（path length）」----
    # 位移类判据（欧氏位移 / 沿指令的投影位移）在「指令每 10 s 重采样一次」的任务里都会失效：
    # episode 内换向后总位移可能接近 0（甚至投影为负），机器人明明一直在走却被判成没走动。
    # 因此默认使用 AERManagerBasedRLEnv 每步累积的路程 ∫|v_xy|dt。
    travelled = None
    if use_path_length and getattr(env, "_travel_ready", False):
        travelled = env._travelled[ids].clone()
        env._travelled[ids] = 0.0

    if travelled is None:
        # 回退（fallback）：位移类判据，仅在环境没有提供路程 buffer 时使用
        disp = asset.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2]
        if use_projection:
            proj = torch.sum(disp * cmd_xy, dim=1) / torch.clamp(cmd_speed, min=1e-6)
            travelled = torch.where(cmd_speed > 0.05, proj, torch.linalg.norm(disp, dim=1))
        else:
            travelled = torch.linalg.norm(disp, dim=1)

    # 阈值：至少要走完半块地形才升级，至少 1/4 块地才不降级（地块边长 8 m -> 4 m / 2 m）
    block_size = terrain.cfg.terrain_generator.size[0]
    up_thr = torch.clamp(up_ratio * expected_dist, min=0.5 * block_size)
    down_thr = torch.clamp(down_ratio * expected_dist, min=0.25 * block_size)

    # robots that walked far enough progress to harder terrains
    move_up = travelled > up_thr
    # robots that walked less than their required distance go to simpler terrains
    move_down = travelled < down_thr
    move_down *= ~move_up
    # 站立指令（rel_standing_envs）不参与降级，避免"站着"被当成"走不动"
    move_down &= cmd_speed > 0.05

    # update terrain levels
    terrain.update_env_origins(ids, move_up, move_down)
    # 周期性存档地形等级，供 --resume 时恢复
    _maybe_save_terrain_levels(env, terrain)

    # return the mean terrain level
    mean_level = torch.mean(terrain.terrain_levels.float())

    if not log_per_terrain or terrain.cfg.terrain_generator is None:
        return mean_level

    # ---- 分地形（按列 col = 地形类型）统计，便于诊断「哪种地形卡住了课程」----
    col_names = getattr(env, "_terrain_curriculum_col_names", None)
    if col_names is None:
        col_names = _sub_terrain_col_names(terrain.cfg.terrain_generator)
        env._terrain_curriculum_col_names = col_names
    num_cols = len(col_names)

    # 累计升/降次数（cumulative，累计值），避免单步瞬时值噪声过大
    stat = getattr(env, "_terrain_curriculum_stat", None)
    if stat is None:
        stat = torch.zeros((num_cols, 3), device=env.device)
        env._terrain_curriculum_stat = stat
    idx = terrain.terrain_types[ids]
    up_f, down_f = move_up.float(), move_down.float()
    stat[:, 0].index_add_(0, idx, up_f)
    stat[:, 1].index_add_(0, idx, down_f)
    stat[:, 2].index_add_(0, idx, torch.ones_like(up_f))

    types = terrain.terrain_types
    levels = terrain.terrain_levels.float()
    out: dict[str, torch.Tensor] = {
        "mean": mean_level,
        "up": move_up.float().mean(),
        "down": move_down.float().mean(),
    }
    for name in dict.fromkeys(col_names):
        col_mask = torch.zeros(num_cols, dtype=torch.bool, device=env.device)
        col_mask[[c for c, n in enumerate(col_names) if n == name]] = True
        env_mask = col_mask[types]
        out[f"level_{name}"] = levels[env_mask].mean()
        out[f"up_{name}"] = stat[col_mask, 0].sum() / torch.clamp(stat[col_mask, 2].sum(), min=1.0)
        out[f"down_{name}"] = stat[col_mask, 1].sum() / torch.clamp(stat[col_mask, 2].sum(), min=1.0)
    return out
