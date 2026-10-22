# LLP-Shrink

**LEARNING FROM LABEL PROPORTIONS WITH SHRINKING BAG**

依据 Lu.pdf 第 3 节公式 (1)–(7)、第 4 节实验设置构建，基于
[TianhaoMa5/LLP-AHIL](https://github.com/TianhaoMa5/LLP-AHIL/tree/65c4bfa201f4a4d820452e5be9de51c5f6c4b326)
的 WRN 和 RandAugment 修改。包含训练、测试、模型权重、逐样本选择权重导出、断点续训、实验网格和自动检查。

**核心实现：每袋每类按置信度筛选且不超过类别计数；每类构造独立剩余集合；λ 使用 minibatch 实际选择比例。**
不是把 AHIL 的连续权重简单换成阈值，也不是删除所有已选样本后使用统一剩余袋。

项目覆盖论文中 **LLP-Shrink 的三个数据集 × 两种袋大小，以及 CIFAR-10 阈值消融**。
附带 DLLP、AHIL、全监督诊断入口；未声称实现表 1 的全部对比方法，LLPFC、LLP-VAT、ROT 不在本项目的已验证范围内。
本地验证与完整 GPU 实验的区别见 [验证报告](docs/VALIDATION.md)。不承诺复现论文数值。

## 安装

建议使用独立 Python 3.10/3.11 环境。先安装适合服务器驱动的 PyTorch，再安装项目；本地测试版本为 torch 2.6.0 / torchvision 0.21.0。
安装命令参考 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/#v260)。

```bash
git clone https://github.com/iiiyouhb/LLP-Shrink.git
cd LLP-Shrink
python -m venv .venv
source .venv/bin/activate
# CPU 检查环境；GPU 训练需换成适配驱动的 CUDA wheel。
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Windows 激活命令为 `.venv\Scripts\Activate.ps1`。AutoDL 配置见 [服务器运行指南](docs/AUTODL.md)。

## 先做检查

```bash
python -W error::RuntimeWarning -W error::DeprecationWarning -m unittest discover -s tests -v
python -m llp_shrink.train --config configs/smoke.json --output runs/smoke
python -m llp_shrink.evaluate runs/smoke/last.pth --device cpu --ema
```

`smoke.json` 使用合成图像、小网络、小 batch，只验证程序；日志明确标注 `SYNTHETIC`，不能作为论文结果。
正式网络和真实数据的单步预检：

```bash
python scripts/preflight.py --dataset FashionMNIST --root data --download --device cpu
```

## 正式训练

```bash
# CIFAR-10，M=16；64 个袋 × 16 个实例 = 1024 个实例 / minibatch
python -m llp_shrink.train --config configs/paper.json --dataset CIFAR10 --bag-size 16 --root data --download --device cuda --output runs/CIFAR10/shrink_M16_s3_tau0.95

# CIFAR-10，M=128；8 个袋 × 128 个实例 = 1024 个实例 / minibatch
python -m llp_shrink.train --config configs/paper.json --dataset CIFAR10 --bag-size 128 --root data --download --device cuda --output runs/CIFAR10/shrink_M128_s3_tau0.95

# 同一入口支持 SVHN / FashionMNIST
python -m llp_shrink.train --config configs/paper.json --dataset SVHN --bag-size 16 --root data --download --device cuda
python -m llp_shrink.train --config configs/paper.json --dataset FashionMNIST --bag-size 128 --root data --download --device cuda
```

默认采用上游 `seed=3`。数据使用 torchvision 标准目录；SVHN 只使用 train/test，不使用 extra。
训练只将真实标签聚合为袋计数。若希望日志包含与旧日志相同的 `n_correct_u` 数字，可加 `--audit-labels`；这些标签只用于训练后诊断，不进入 Shrink/AHIL/DLLP 损失。

`--batch-instances` 是**原始实例数**；弱、强增强拼接后，一次网络前向实际处理两倍图像。
默认 1024 个实例、1024 epoch、WRN-28-2、τ=0.95、SGD、lr=0.03、momentum=0.9、weight_decay=5e-4。
采用上游实际生效的 `cos(π step/(2 total_steps))` 调度；另提供 `--scheduler cosine` 和 `--scheduler ahil-paper`，改变调度须单独记录。

## 实验批量运行

```bash
# 先打印六组主实验命令，不启动长时间训练
python scripts/run_experiments.py --suite main
# 执行；已完成的实验跳过，未完成的从 last.pth 恢复
python scripts/run_experiments.py --suite main --root data --download --execute --resume
# 若需要 mean ± sample std，显式指定多个种子；不是上游已给出的重复方案
python scripts/run_experiments.py --suite main --seeds 1 2 3 --root data --execute --resume
# 阈值消融；图中未列出全部精确阈值，网格是可编辑的复现约定
python scripts/run_experiments.py --suite threshold --thresholds 0 0.25 0.5 0.75 0.95 0.96 0.98 1 --execute --resume
python scripts/summarize.py runs --output experiment_summary.json
```

汇总只纳入实际完成的运行，默认排除合成数据；一个种子时标准差写 `null`。默认汇总最终 epoch，最佳测试表现另留在每个 `summary.json` 中，不把它冒充独立验证集选模结果。

## 日志和权重

每个运行目录输出：

| 文件 | 内容 |
| --- | --- |
| `YYYY-MM-DD_HH_MM_SS_*.log` | 与给定 logs_copy 类似的时间戳、INFO、epoch、loss_u、loss_prop、Acc、Ema-Acc、best_acc；另加 λ、Selected |
| `metrics.jsonl` | 机器可读逐 epoch 指标，续训时以 checkpoint 历史去重 |
| `config.json` / `environment.json` | 完整参数及环境 |
| `bags.npz` / `data_manifest.json` | 固定袋成员、类别计数、未成袋样本、数据指纹和丢弃数量 |
| `last.pth` | 每个完整 epoch 原子保存：模型、EMA、优化器、scheduler、RNG、指标历史 |
| `best_raw.pth` / `best_ema.pth` | 最佳测试表现对应 checkpoint，明确仅作描述性记录 |
| `model_final.pt` / `ema_final.pt` | 完整训练结束后的纯 `state_dict` 模型权重 |
| `summary.json` | 是否完整结束、最终指标、最佳测试指标 |

加 `--export-weights-every 100` 可导出 `sample_weights_epoch_*.npz`，包含实际处理样本的原始 ID、bag ID、伪标签、置信度、0/1 选择权重和当前 batch λ。它们是该 epoch 训练前向时的记录，不是训练结束后重新预测的结果，也不包含当轮丢弃的尾部 minibatch。

```bash
python -m llp_shrink.train --resume runs/CIFAR10/shrink_M16_s3_tau0.95/last.pth --device cuda
python -m llp_shrink.evaluate runs/CIFAR10/shrink_M16_s3_tau0.95/last.pth --root data --device cuda --ema
```

只加载自己生成或可信来源的 checkpoint。已有输出目录不会被新训练覆盖；如果进程被强制杀死留下 `.running.lock`，先确认该任务确实已停止，再删除这个锁文件后续训。

更多细节：[公式与代码对照](docs/METHOD.md)、[上游审计与修改](docs/UPSTREAM_AUDIT.md)、[实验对应关系](docs/EXPERIMENTS.md)、[来源记录](THIRD_PARTY.md)。
