# AutoDL 运行与接管说明

当前用户选择：先完成项目和本地检查，之后再建立 SSH。本项目没有擅自启动付费 GPU 实验。

## 环境预检

```bash
nvidia-smi
python --version
df -h
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

先确认 GPU 型号、驱动、显存、Python 和磁盘，再选择相应 wheel。版本对照以
[PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/#v260) 为准。
以下仅为支持 CUDA 12.4 的环境示例，不自动修改现有服务器环境：

```bash
python -m venv /root/autodl-tmp/llp-env
source /root/autodl-tmp/llp-env/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/iiiyouhb/LLP-Shrink.git /root/autodl-tmp/LLP-Shrink
cd /root/autodl-tmp/LLP-Shrink
pip install -e .
python -m unittest discover -s tests -v
```

## 先验证，再长跑

```bash
# 合成数据只验证 CUDA 流程与 checkpoint；不用于测正式 batch 的显存
python -m llp_shrink.train --config configs/smoke.json --device cuda --output runs/cuda-smoke
# 真实数据与正式 WRN-28-2
python scripts/preflight.py --dataset CIFAR10 --root /root/autodl-tmp/data --download --device cuda
# 正式网络、正式 batch、完整 1024-epoch scheduler，仅先跑完一轮
python -m llp_shrink.train --config configs/paper.json --dataset CIFAR10 --bag-size 16 --root /root/autodl-tmp/data --download --device cuda --output /root/autodl-tmp/llp-runs/CIFAR10/shrink_M16_s3_tau0.95 --stop-after-epoch 1 --log-every 10
```

`--stop-after-epoch 1` 不把总训练改成一个 epoch，因而不会将学习率提前衰减到终点。预检通过后直接接着恢复：

```bash
python -m llp_shrink.train --resume /root/autodl-tmp/llp-runs/CIFAR10/shrink_M16_s3_tau0.95/last.pth --device cuda
```

每个 minibatch 有 1024 个实例，同时构造弱/强增强，前向共 2048 张图像。正式显存和耗时必须在实际 GPU 上测量。
若 OOM，优先核对空闲显存并选择足够显存的 GPU；`--amp` 会改变数值路径，开启后需要单独检查并记录。
不要不加说明地减小 batch、换小模型或用跨袋微批次冒充原实验；BatchNorm 和每 batch 的 λ 会随之变化。
本版为单进程、单 GPU 训练，不实现 DDP；多张 GPU 可分别运行独立配置，并指定不同输出目录。

## 保持训练与保存结果

可以在已有的 `tmux` 会话中运行；若服务器未安装 tmux，也可使用 `nohup`：

```bash
mkdir -p /root/autodl-tmp/llp-runs
nohup python -m llp_shrink.train --resume /root/autodl-tmp/llp-runs/CIFAR10/shrink_M16_s3_tau0.95/last.pth --device cuda > /root/autodl-tmp/llp-runs/launcher.log 2>&1 &
echo $!
```

保存 PID，并通过对应运行目录的 `.log` 和 `summary.json` 检查进度。
`SIGTERM` 请求会在当前 epoch 完成并保存后退出；强制结束或断电最多丢失当前未完成 epoch。
`last.pth` 使用临时文件 + 原子替换；续训恢复模型、EMA、优化器、调度器和随机状态。
复制完整运行目录回本地，保留 `.log`、JSON、`bags.npz`、checkpoint，不只复制最终纯权重。

## SSH 协作需要的信息

- 主机、端口、用户名，以及可使用的本地密钥路径；不要在聊天中粘贴私钥或密码。
- GPU 数量、可运行时间、预算/何时停止、服务器项目与数据目录。
- 首次连接需核对主机指纹；保持 SSH 的主机密钥检查。

连接建立后可以通过 SSH 完成服务器环境检查、部署、单步/一轮验证、启动和检查长训练。
本地 CPU 检查不能代替这一步；本版未宣称已通过 AutoDL、CUDA、AMP 或 1024-epoch 收敛验证。
