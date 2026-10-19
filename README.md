# SO101-VLA-Lab

从示范数据到动作块策略，再到 VLA 微调和真实机器人验证的两个月实验项目。

## 目标

- 在 SO101 单臂上完成数据采集、ACT 训练和真机推理
- 在 LIBERO 上跑通一个固定基线
- 在 LIBERO 上微调一个 VLA，优先 SmolVLA
- 将同一套数据记录、预处理和评测规范迁移到一个真实任务
- 记录相机、延迟、标定、数据质量和失败案例

## 项目状态

| 阶段 | 状态 | 结果 |
|---|---|---|
| 极简 Transformer / diffusion / flow matching | 未开始 | |
| SO101 单臂 ACT | 未开始 | |
| LIBERO 基线 | 未开始 | |
| VLA 微调 | 未开始 | |
| 真实任务迁移 | 未开始 | |

## 快速入口

- [环境与版本](docs/setup.md)
- [每周记录](docs/weekly-log/)
- [踩坑记录](docs/troubleshooting/)
- [失败案例分类](docs/failure-taxonomy.md)
- [实验结果](results/)

## 复现实验原则

每个结果都记录 git commit、环境版本、数据集版本、配置、随机种子、checkpoint 和评测 episode 数。大文件不直接提交到 Git 历史，使用 Hugging Face 或 GitHub Release 保存。

## 当前边界

本仓库只报告已经实际运行并保存日志的结果。计划、推测和待验证内容会单独标注。
