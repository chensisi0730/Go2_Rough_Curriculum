from __future__ import annotations

import os

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

from .mdp.curriculums import _sub_terrain_col_names

class AERRewardManager(RewardManager):
    positive_terms = {"track_lin_vel_xy", "track_ang_vel_z", "energy_new_actual"}
    sigma_aux = 0.02

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf[:] = 0.0
        positive_reward = torch.zeros_like(self._reward_buf)
        negative_reward = torch.zeros_like(self._reward_buf)
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue

            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            self._episode_sums[name] += value 
            self._step_reward[:, term_idx] = value / dt

            if name in self.positive_terms:
                positive_reward += value
            else:
                negative_reward += value

        self._reward_buf[:] = positive_reward * torch.exp(negative_reward / self.sigma_aux)
        return self._reward_buf
class AERManagerBasedRLEnv(ManagerBasedRLEnv):
    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = AERRewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    """地形变化范围记录（terrain scan range logging，记录 height_scanner 看到的地形起伏）。"""

    def _scan_logging_enabled(self) -> bool:
        """是否记录地形变化范围。默认开启；设环境变量 LOG_TERRAIN_SCAN=0 可关闭（消融对照用）。"""
        return os.environ.get("LOG_TERRAIN_SCAN", "1") != "0"

    def _init_scan_buffers(self) -> None:
        self._scan_min = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._scan_max = torch.full((self.num_envs,), -float("inf"), device=self.device)
        # 单帧（frame-level）统计：反映"此刻脚下"的地形起伏，与行走距离无关
        self._scan_frame_range = torch.zeros((self.num_envs,), device=self.device)
        self._scan_frame_std = torch.zeros((self.num_envs,), device=self.device)
        self._scan_frame_cnt = torch.zeros((self.num_envs,), device=self.device)

    def _update_scan_buffers(self) -> None:
        """累积当前帧的射线统计：episode 内包络（min/max）+ 单帧起伏（range/std）。"""
        if not getattr(self, "_scan_buffers_ready", False):
            if not self._scan_logging_enabled():
                return
            self._init_scan_buffers()
            self._scan_buffers_ready = True
        sensors = self.scene.sensors
        sensor = sensors.get("height_scanner") if sensors else None
        if sensor is None:
            return
        # 射线命中点的世界 z（未受 height_scan 的 offset 影响，直接反映地形绝对高度）
        z = sensor.data.ray_hits_w[..., 2]
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = z.min(dim=1).values, z.max(dim=1).values
        # ① episode 内包络：走动越远、跨越地块越多，值越大
        self._scan_min = torch.minimum(self._scan_min, lo)
        self._scan_max = torch.maximum(self._scan_max, hi)
        # ② 单帧起伏：同一时刻 187 条射线的落差，等价于"传感器这一刻看清的地形起伏"
        self._scan_frame_range += hi - lo
        self._scan_frame_std += z.std(dim=1)
        self._scan_frame_cnt += 1.0

    def _collect_scan_stats(self, env_ids) -> dict:
        """汇总本 episode 的地形起伏：① episode 内包络 ② 单帧平均起伏，按地形类型分组。"""
        if not getattr(self, "_scan_buffers_ready", False):
            return {}
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        rng = (self._scan_max - self._scan_min)[ids]
        cnt = self._scan_frame_cnt[ids]
        # reset 后 buffer 仍为 inf/-inf 的 env（本 episode 没采到数据）剔除
        valid = torch.isfinite(rng) & (cnt > 0)
        if valid.sum() == 0:
            return {}
        rng_valid = rng[valid]
        ids_valid = ids[valid]
        safe_cnt = torch.clamp(cnt[valid], min=1.0)
        frame_range = self._scan_frame_range[ids][valid] / safe_cnt
        frame_std = self._scan_frame_std[ids][valid] / safe_cnt

        log: dict[str, float] = {
            "Terrain_Scan/range_mean": rng_valid.mean(),
            "Terrain_Scan/range_min": rng_valid.min(),
            "Terrain_Scan/range_max": rng_valid.max(),
            "Terrain_Scan/frame_range_mean": frame_range.mean(),
            "Terrain_Scan/frame_std_mean": frame_std.mean(),
            "Terrain_Scan/num_envs": float(ids_valid.numel()),
        }

        # 按地形类型（terrain type，列 col）分组
        terrain = getattr(self.scene, "terrain", None)
        if terrain is not None and getattr(terrain, "terrain_origins", None) is not None:
            names = getattr(self, "_scan_col_names", None)
            if names is None:
                gen_cfg = terrain.cfg.terrain_generator
                names = _sub_terrain_col_names(gen_cfg) if gen_cfg is not None else []
                self._scan_col_names = names
            types = terrain.terrain_types[ids_valid]
            for name in dict.fromkeys(names):
                cols = [c for c, n in enumerate(names) if n == name]
                mask = torch.isin(types, torch.tensor(cols, device=self.device))
                if mask.any():
                    log[f"Terrain_Scan/range_{name}"] = rng_valid[mask].mean()
                    log[f"Terrain_Scan/frame_range_{name}"] = frame_range[mask].mean()

        # 清空已结算 env 的缓冲，开始下一个 episode 的统计
        self._scan_min[ids] = float("inf")
        self._scan_max[ids] = -float("inf")
        self._scan_frame_range[ids] = 0.0
        self._scan_frame_std[ids] = 0.0
        self._scan_frame_cnt[ids] = 0.0
        return log

    def _reset_idx(self, env_ids):
        # 注意：必须在本 episode 的 buffer 被清空前汇总，且要在 scene.reset() 之前读取传感器
        self._ensure_travel_buffer()
        scan_log = self._collect_scan_stats(env_ids)
        super()._reset_idx(env_ids)
        if scan_log:
            self.extras.setdefault("log", {}).update(scan_log)

    """路程累积（path length，世界系平面速度对时间的积分），供地形课程判定使用。"""

    def _ensure_travel_buffer(self) -> None:
        if not getattr(self, "_travel_ready", False):
            self._travelled = torch.zeros(self.num_envs, device=self.device)
            self._travel_ready = True

    def _update_travel_buffer(self) -> None:
        """每步累加 |v_xy| * dt，得到本 episode 走过的路程。

        课程升级/降级不能用「位移」判断：指令每 10 s 重采样一次，episode 内换向后
        位移会正负抵消，机器人一直在走却会被判成没走动。
        """
        self._ensure_travel_buffer()
        vel = self.scene["robot"].data.root_lin_vel_w[:, :2]
        self._travelled += torch.linalg.norm(vel, dim=1) * self.step_dt

    def step(self, action):
        obs, rew, term, trunc, extras = super().step(action)
        # super().step() 内部已完成 reset，这里累积的是当前（可能已是新地形）的数据
        self._update_scan_buffers()
        self._update_travel_buffer()
        return obs, rew, term, trunc, extras