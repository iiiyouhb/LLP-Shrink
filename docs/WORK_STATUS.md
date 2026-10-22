# 工作进度与恢复入口

更新时间：2026-10-22（Asia/Shanghai）。本文件用于会话中断后恢复工作，不代表后台任务会自动继续。

- 用户目标仓库：https://github.com/iiiyouhb/LLP-Shrink
- 保留初始提交：`5a5c65a35fd89e3c7e9e83aecb7b4ad948c8a6a0`。
- 用户确认：每类额度 top-confidence；每类独立剩余集合；不归一化；默认 seed 跟随 AHIL，即 3。
- 用户选择：先完成代码和本地验证，再连接 AutoDL。
- 已完成：两篇论文方法/实验阅读、官方代码审计、核心实现、训练/评估/权重保存、实验网格、文档。
- 已通过：21 项自动检查；CIFAR10 合成数据多 worker 端到端训练与重新加载评估；真实 FashionMNIST 的 WRN-28-2 前向、反向、保存和重载预检。
- 正在完成：安装打包检查、最终提交推送。
- GitHub：用户已通过 Git Credential Manager 登录，推送 dry-run 已成功；实际推送状态必须以远端 commit 为准。
- 本机环境：`C:\Anaconda\envs\pyt\python.exe`，PyTorch 2.6.0+cpu / torchvision 0.21.0+cpu；无可用 CUDA。
- 原始材料、下载数据、调试日志位于当前工作区 `work/`；不得将它们误提交到 GitHub。

若会话或额度中断：从本文件、`docs/VALIDATION.md`、`git status` 和实际远程提交状态继续，不必从头重建。
当前没有可用的额度监测或额度刷新后自动唤醒工具，因此不承诺自动检测刷新并恢复；用户在本聊天发送“继续”可沿已保存状态接着做。
