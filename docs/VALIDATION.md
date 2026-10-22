# 验证报告

验证日期：2026-10-22。环境：Windows，Python 3.9.18，PyTorch 2.6.0+cpu，torchvision 0.21.0+cpu，NumPy 1.26.4。
原始自动测试输出见 [test-results-windows.txt](test-results-windows.txt)。

## 已通过的 21 项自动检查

执行：

```bash
python -W error::RuntimeWarning -W error::DeprecationWarning -m unittest discover -s tests -v
```

- 手算三类示例：集合、剩余目标、预测比例、两个损失及总损失。
- 每类额度、最高置信度排序、相同置信度的稳定顺序、严格大于阈值、零类别额度。
- 无选择时退化为 DLLP；全选择时只剩实例项；空类集合、空选择袋均有限。
- 多组随机输入与极端 logits 的约束和梯度；float64 `gradcheck`。
- 不合法计数、阈值、NaN 输入会报错。
- AHIL 权重有界且不参与求导。
- 固定袋划分可重复、互不重叠、计数准确、余数可追踪。
- CIFAR10/SVHN/FashionMNIST 三类输入、数据增强和测试尺寸；训练默认不暴露真实实例标签。
- Windows 0/2 workers 的 bag/sample/label 对齐。
- AutoContrast uint8 取负和 Posterize 位掩码回归检查。
- 实际 WRN-28-2 对三种输入均完成梯度更新。
- 评估最后一个不满 batch 使用正确实例分母。
- EMA 参数加权、BN buffer 复制；学习率匹配上游实际生效定义。
- 完整两轮训练，与“完成一轮、从 last.pth 恢复第二轮”的模型/EMA **逐 tensor 完全相等**；除耗时外的历史指标一致（CPU，workers=0，WRN-10-1 小网络）。
- checkpoint、最终权重、样本选择权重和指标历史实际生成并检查。
- 关闭 EMA 以及 Shrink/DLLP/AHIL/supervised 四种入口的训练烟雾测试。
- 开启/关闭真实标签诊断时模型逐项相同；新任务不能覆盖已有运行目录。

## 实际命令行端到端检查

`configs/smoke.json` 的 CIFAR10 合成图像，用 2 workers 跑完两个 epoch，再用独立评估入口读取 `last.pth` 的 EMA 模型，准确率与训练日志记录一致。
示例：[synthetic-cifar10.log](../examples/synthetic-cifar10.log)、[synthetic-summary.json](../examples/synthetic-summary.json)。
这些文件是**实际执行产生的合成测试日志**，没有手工填入训练指标；模型小、数据假，数值不是论文性能。

## 真实数据预检

下载并读取完整官方 FashionMNIST train/test 文件（60,000 / 10,000），在随机袋中的 32 张真实训练图像上运行 **WRN-28-2**：

- τ=0.95 分支前向有限，初始模型没有高置信度样本是正常现象。
- τ=0 分支确实选中样本；反向传播、梯度有限性检查、SGD 更新通过。
- 保存权重后重新加载，在 19 张真实测试图像上评估结果一致。

原始报告：[real-fashion-preflight.json](real-fashion-preflight.json)。这是数据/网络/损失连通性检查，不是完整训练或收敛验证。

## 实际修复迭代

1. 上游 AutoContrast 和 Posterize 的无符号整数问题：实际数据增强测试触发后修复。
2. 手算测试中的第三类概率加和误写为 0.83，经核对输入应为 0.82；修正预期值，公式实现不变。
3. Windows 训练结束后日志句柄未关闭导致临时目录清理失败：加入 finally 关闭 handler。
4. 复用已有运行目录的保护前移，避免新任务先写入文件；失败的恢复不改写已有环境文件。
5. 批量运行脚本将路径解析为绝对路径，避免从项目外调用时父进程与子进程判断不同目录。

## 尚未验证的边界

- 未执行三个数据集的完整 1024-epoch 实验、mean/std 或阈值消融数值。
- 本机没有可用 CUDA；CUDA/AMP、目标服务器显存/速度、跨 GPU/跨版本逐位可重复性未验证。
- CPU 断点完全一致结论只覆盖本报告明确测试的配置；多 worker 已做数据与端到端检查，尚未声称 GPU 多 worker 断点完全一致。
- CIFAR10/SVHN 的真实完整训练待服务器完成；三种输入的网络与增强路径均已本地检查。
- 尝试下载 CIFAR10 官方档案，本机该源约 60–90 KB/s，已停止此次未完成下载；未将它算作完成的真实数据验证。SVHN 真实数据下载未开始。
- 本项目没有实现与验证表 1 全部对比方法。完整说明见 [EXPERIMENTS.md](EXPERIMENTS.md)。

GitHub Ubuntu CPU CI **实际执行成功**：[运行 39735541442](https://github.com/iiiyouhb/LLP-Shrink/actions/runs/39735541442)，对应代码提交 `9234c9ef45ee56ef94a2c22a634f88e4eaa3551b`。
已通过环境安装、全部单元/集成检查、命令行烟雾训练及 EMA checkpoint 独立评估。该结论仍不包含 CUDA 或长训练。

## 打包与交付检查

`python -m pip wheel --no-deps .` 成功构建 wheel；源码通过 `compileall` 和 `git diff --check`。
正式代码已在 `9234c9e` 提交推送到目标仓库，并核对远端 main 与本地 SHA 一致。
