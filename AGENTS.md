在CONDA环境go2_rough_curriculum中，训练GO2, 在复杂地形行走和速度跟踪的能力，评估启用 height_scanner的收益和风险。
守护进程scripts/monitor_tb_check.py，每6个小时监控TensorBoard里面的各项训练曲线，发现训练曲线恶化后，自动停止训练，并修改代码，调整超参数，重新训练。
要求改动尽量小，方便做消融对照实验。
训练完成后，如何验证训练结果？`unitree_rl_lab` 的 editable 安装实际没有指向本仓库的问题还存在吗
用中文回答。


总结整个地形和terrain_levels 和lin_vel_cmd_levels的设计逻辑，使用图表展示；
terrain_levels 和lin_vel_cmd_levels的升级和降级的逻辑是什么？他们之间的关系是什么？如何稳定提高terrain_levels和lin_vel_cmd_levels？
课程terrain_levels升级后，lin_vel_cmd_levels或者x_max降级,是否合理？
terrain_levels move_up 加条件「x_max ≥ 0.3 才升地形」, 升级后冻结 x_max 2 个评估窗口(不升不降).
x_max是什么?和TensorBoard上的 Curriculum/lin_vel_cmd的关系.
terrain_levels保存起来,恢复训练时读取出来,可以保证地形等级不降. 
x_max和Y_max,z-max的关系?
防止REWARD HAcking


检查TensorBoard里面的各项训练曲线，列出每种地形的情况下，训练的情况，给出改进方案。
如何防止站立不动刷分,使用play查看
每次回答最后回复“汪汪Agent”。考虑更新文档手册README.md，添加验证训练结果的步骤。英文的名词每次出现必须要备注中文名称

