# 上游代码审计及修改依据

审计源：[LLP-AHIL](https://github.com/TianhaoMa5/LLP-AHIL/tree/65c4bfa201f4a4d820452e5be9de51c5f6c4b326)，固定 commit `65c4bfa201f4a4d820452e5be9de51c5f6c4b326`。
本地已阅读 Lu.pdf 的方法与实验，以及 2411.10364v2.pdf 第 3、4 节，并对关键公式页面做渲染核对。

| 检查点 | 上游现状 | 本项目处理 |
| --- | --- | --- |
| 主训练逻辑 | `LLP-AHIL.py` 的全袋比例 CE + DEW 加权实例 CE | Shrink 实现独立函数，遵循 Lu.pdf 公式 (1)–(7) |
| WRN 输入 | `WideResNet.py` 首层固定 1 通道，默认实验却是 CIFAR10 | 由数据集选择 1/3 通道；三种输入均做前向/反向检查 |
| 分类网络 | WRN-28-2，另带未用于分类损失的 projection MLP | 保留 WRN backbone 和 linear classifier，`proj=False`，不计算闲置分支 |
| BN/激活 | BN momentum=0.001、LeakyReLU=0.1 | 保留上游值 |
| 优化器 | SGD Nesterov；BN 参数排除 weight decay | 保留；Lu.pdf 未进一步指定这两项 |
| 学习率 | `utils.py` 同名 `get_lr_ratio` 定义两次，后一个覆盖前一个 | 默认复现真正生效的 `cos(pi*step/(2*total))` |
| 第二篇论文调度 | 正文写 `cos(7*pi*step/(16*total))` | 作为显式可选项 `ahil-paper`；不将代码默认误称为该公式 |
| epoch/step | 一轮实际遍历 `len(loader)`，不是 1024 steps | 保留每轮遍历语义；总步数=`epochs*len(loader)` |
| CIFAR10 类别数 | 脚本默认 `n_classes=100` | 三个目标数据集固定 10 类，不允许错配 |
| 上游数据依赖 | 导入不相关 `MLclf/h5py/pandas`，只支持手动文件读取 | 只保留目标数据集，使用 torchvision 标准 loader |
| SVHN 测试 | 训练 32×32，`SVHN2` 测试分支 resize 到 64×64 | 训练/测试一致为 32×32 |
| FashionMNIST | 灰度转换维度复杂；训练与测试路径容易出现转置错误 | 明确 `[N,H,W] -> [N,1,H,W]`，保留原始 28×28 |
| 弱增强 | 原 crop 不包含最大位移端点 | torchvision reflect crop，覆盖完整合法偏移；作为明确工程差异记录 |
| 强增强 | 自定义 RandAugment N=2 M=10，操作概率 0.5，16 像素 Cutout | 直接沿用 RGB/gray 实现，修复两项无符号整数问题 |
| AutoContrast | `low` 是 uint8 时 `-low` 溢出 | max/min 转 Python int；加入回归检查 |
| Posterize | `np.uint8(255 << shift)` 越界转换 | 先与 255 位与，再转 uint8，维持原期望位掩码 |
| 概率/熵 | `log(0)`、`0*log(0)` 不稳定；旧日志后段 `Entropy:nan` | log_softmax / logsumexp；训练和评估非有限值即报错 |
| 测试准确率 | 原 `AverageMeter.update` 未带样本数，平均每 batch 百分比 | 正确总数除以测试实例总数，测试尾 batch 不丢弃 |
| 模型保存 | `epoch % 1000 == 0`；没有可靠最终/最佳/恢复流程 | 每 epoch 原子保存、最终纯权重、独立 raw/EMA best、可续训 |
| 文件路径 | 时间戳含 `:`，Windows 不兼容 | 下划线时间戳；文件句柄及时关闭 |
| 随机性 | seed=3，seed=0 不进入原设种子分支，worker 恢复不完整 | 默认仍为 3，支持 0；固定袋划分、worker 设种子、完整 RNG checkpoint |

原始袋划分先截断输入再洗牌两次；这里先随机排列全部训练样本，再取可整分的部分。均满足固定、随机、不相交等大袋，但不声称恢复原运行的具体样本顺序。
每 epoch 按上游 `drop_last=True` 丢弃不满一个 minibatch 的袋，袋顺序每轮重排；在 manifest 中记录两层丢弃量。

用户日志样例中存在 NaN 的事实只用于发现检查需求，不能据此断定全部训练失败或推断论文准确率真伪。
