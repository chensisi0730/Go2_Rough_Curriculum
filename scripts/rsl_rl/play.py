# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--terrain_level",
    type=int,
    default=None,
    help="强制所有环境使用指定的课程等级（curriculum level，地形行索引 row，0 最简单）。默认走课程。",
)
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    help="强制地形类型（terrain type，地形列索引 col），可逗号分隔多个值，如 10,14。默认保留原有分配。",
)
parser.add_argument(
    "--terrain_name",
    type=str,
    default=None,
    help=(
        "按地形名指定地形类型（比记列号方便），可逗号分隔多个，如 pyramid_stairs 或"
        " pyramid_stairs,pyramid_stairs_inv。可用名字：flat / smooth_slope / smooth_slope_inv /"
        " random_rough / pyramid_stairs / pyramid_stairs_inv / discrete_obstacles。"
    ),
)
parser.add_argument(
    "--scan_debug",
    type=int,
    default=0,
    help="每 N 步打印一次 height_scanner（高度扫描传感器）统计，0 表示关闭。",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="跑满指定步数后自动退出（0 表示不限制），便于无头（headless）调试与回归测试。",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
import isaaclab.envs.mdp as mdp_core
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_demo.tasks  # noqa: F401
from go2_demo.tasks.manager_based.go2_demo.mdp.curriculums import _sub_terrain_col_names


def handle_deprecated_rsl_rl_cfg(agent_cfg):
    """Convert the legacy policy/algorithm config to the rsl_rl 5.x actor/critic layout."""
    if hasattr(agent_cfg, "to_dict"):
        cfg_dict = agent_cfg.to_dict()
    elif isinstance(agent_cfg, dict):
        cfg_dict = agent_cfg
    else:
        return agent_cfg

    if "policy" in cfg_dict and "actor" not in cfg_dict:
        policy_cfg = cfg_dict.pop("policy")
        cfg_dict["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
        }
        cfg_dict["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
        }
        if "obs_groups" not in cfg_dict:
            cfg_dict["obs_groups"] = {"actor": ["policy"], "critic": ["policy"]}
        init_std = policy_cfg.pop("init_noise_std", 1.0)
        std_type = policy_cfg.pop("noise_std_type", "scalar")
        state_dependent = policy_cfg.pop("state_dependent_std", False)

        policy_cfg.pop("stochastic", None)
        policy_cfg.pop("is_stochastic", None)

        if state_dependent:
            cfg_dict["actor"]["distribution_cfg"] = {
                "class_name": "HeteroscedasticGaussianDistribution",
                "init_std": init_std,
                "std_type": std_type,
            }
        else:
            cfg_dict["actor"]["distribution_cfg"] = {
                "class_name": "GaussianDistribution",
                "init_std": init_std,
                "std_type": std_type,
            }
        cfg_dict["critic"]["distribution_cfg"] = None

    return cfg_dict


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, handle_deprecated_rsl_rl_cfg(agent_cfg), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(
        resume_path,
        load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
    )

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.get_policy()

    # export policy to onnx/jit (rsl_rl 5.x API)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
    runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # ---- 可选：固定课程等级（curriculum level）/ 地形类型（terrain type）----
    # 用途：消融实验时把机器人只放在「指定难度」的地块上评估，排除课程漂移的干扰。
    if (
        args_cli.terrain_level is not None
        or args_cli.terrain_type is not None
        or args_cli.terrain_name is not None
    ):
        _terrain = env.unwrapped.scene.terrain
        if getattr(_terrain, "terrain_origins", None) is None:
            raise RuntimeError("当前任务不是 grid 地形（terrain_origins 为空），无法指定课程等级。")
        if args_cli.terrain_level is not None:
            assert 0 <= args_cli.terrain_level < _terrain.max_terrain_level, (
                f"terrain_level 必须在 [0, {_terrain.max_terrain_level - 1}] 内。"
            )
            _terrain.terrain_levels[:] = args_cli.terrain_level

        # 解析目标地形列（col）：--terrain_name 优先，其次 --terrain_type
        _cols = None
        _col_names = None
        gen_cfg = _terrain.cfg.terrain_generator
        if gen_cfg is not None:
            _col_names = _sub_terrain_col_names(gen_cfg)
        if args_cli.terrain_name is not None:
            if _col_names is None:
                raise RuntimeError("当前地形不是 generator 类型，无法按名字指定地形。")
            _wanted = [x.strip() for x in args_cli.terrain_name.split(",") if x.strip()]
            _missing = [w for w in _wanted if w not in _col_names]
            if _missing:
                raise ValueError(
                    f"未知地形名 {_missing}；可用：{sorted(dict.fromkeys(_col_names))}"
                )
            _cols = [i for i, n in enumerate(_col_names) if n in _wanted]
        elif args_cli.terrain_type is not None:
            _cols = [int(x) for x in args_cli.terrain_type.split(",") if x.strip() != ""]
            n_cols = len(_col_names) if _col_names else int(max(_cols)) + 1
            if any(c < 0 or c >= n_cols for c in _cols):
                raise ValueError(f"terrain_type 必须在 [0, {n_cols - 1}] 内，收到 {_cols}")

        if _cols:
            # 循环分配到所有 env，保证 env_0（viewer 跟随的那个）一定落在第一个目标地形上
            n_envs = env.unwrapped.num_envs
            _terrain.terrain_types[:] = torch.tensor(
                [_cols[i % len(_cols)] for i in range(n_envs)],
                device=_terrain.device,
                dtype=_terrain.terrain_types.dtype,
            )
            _desc = (
                [f"col{c}={_col_names[c]}" for c in _cols] if _col_names else [f"col{c}" for c in _cols]
            )
            print(f"[INFO] 地形类型（terrain type）循环分配：{_desc}；env_0 -> {_desc[0]}")

        _terrain.env_origins[:] = _terrain.terrain_origins[_terrain.terrain_levels, _terrain.terrain_types]
        # 冻结课程：清空课程项，避免推理过程中地形等级被 curriculum 改写
        env.unwrapped.curriculum_manager.active_terms.clear()
        env.unwrapped.reset()
        print(
            f"[INFO] 已固定 terrain_level={args_cli.terrain_level}，课程已冻结（curriculum frozen）"
        )

    # reset environment
    obs = env.get_observations()
    timestep = 0
    _dbg_i = 0  # 跟随相机诊断计数器（counter，计数器）
    _scan_i = 0  # height_scanner（高度扫描传感器）诊断计数器
    _step_i = 0  # 总步数计数器（用于 --max_steps 自动退出）
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # 诊断（diagnostic，诊断信息）：确认视口跟随相机（follow camera，跟随镜头）是否始终跟随 env_0 的机器人。
        # 仅在环境变量 DEBUG_FOLLOW_CAM=1 时打印，默认关闭，不影响原有 play 行为。
        if os.environ.get("DEBUG_FOLLOW_CAM"):
            _dbg_i += 1
            if _dbg_i % 50 == 0:
                _vcc = env.unwrapped.viewport_camera_controller
                _root = [round(float(x), 2) for x in env.unwrapped.scene["robot"].data.root_pos_w[0][:3]]
                if _vcc is None:
                    print(f"[FOLLOW_CAM] step={_dbg_i} viewport_camera_controller=None (无 GUI/相机控制器)")
                else:
                    _voc = [round(float(x), 2) for x in _vcc.viewer_origin[:3]]
                    print(
                        f"[FOLLOW_CAM] step={_dbg_i} robot_root={_root} cam_origin={_voc} "
                        f"origin_type={_vcc.cfg.origin_type} eye_offset={tuple(_vcc.cfg.eye)}"
                    )

        # 诊断（diagnostic，诊断）：打印 height_scanner（高度扫描传感器）的原始读数与观测值。
        # 用于验证「策略是否真的看得到地形」，以及观测值是否被 clip（裁剪）饱和。
        if args_cli.scan_debug > 0:
            _scan_i += 1
            if _scan_i % args_cli.scan_debug == 0:
                _sensor = env.unwrapped.scene.sensors.get("height_scanner")
                if _sensor is None:
                    print("[SCAN] 当前任务没有 height_scanner 传感器")
                    args_cli.scan_debug = 0
                else:
                    _raw = _sensor.data.ray_hits_w[..., 2]
                    _scan = mdp_core.height_scan(env.unwrapped, sensor_cfg=SceneEntityCfg("height_scanner"))
                    print(
                        f"[SCAN] step={_scan_i} | raw_hit_z mean={_raw.mean():.3f} std={_raw.std():.3f} "
                        f"min={_raw.min():.3f} max={_raw.max():.3f} | "
                        f"obs_scan mean={_scan.mean():.3f} std={_scan.std():.3f} "
                        f"min={_scan.min():.3f} max={_scan.max():.3f}"
                    )
                    # 地形变化范围（terrain scan range，episode 内射线高度包络），
                    # 由 AERManagerBasedRLEnv 在每个 episode 结束时汇总到 extras["log"]
                    _tlog = {
                        k: v
                        for k, v in env.unwrapped.extras.get("log", {}).items()
                        if k.startswith("Terrain_Scan/")
                    }
                    if _tlog:
                        print(
                            "        "
                            + "  ".join(f"{k.split('/')[-1]}={float(v):.3f}" for k, v in _tlog.items())
                        )

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        # 达到指定步数后自动退出（无头调试 / 回归测试用）
        _step_i += 1
        if args_cli.max_steps > 0 and _step_i >= args_cli.max_steps:
            print(f"[INFO] 已达到 --max_steps={args_cli.max_steps}，退出 play 循环。")
            break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
