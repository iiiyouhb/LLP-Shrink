# 公式到代码：以 Lu.pdf 第 3 节为准

实现入口为 `llp_shrink/losses.py::shrink_loss`。用户在本次复现中确认了两项解释：

1. 每袋每类超过 `m[i,k]` 个高置信度候选时，按置信度从高到低取前 `m[i,k]` 个。
2. 采用公式 (3)–(5) 的**每类独立集合**，不采用图 1 的统一剩余袋示意，也不额外将各类比例归一化到和为 1。

## 输入与伪标签

- `weak`, `strong`：形状 `[B,M,C]` 的弱/强增强 logits。
- `counts`：形状 `[B,C]` 的整数袋计数 `m[i,k]=M*alpha[i,k]`，逐袋总和为 M。
- `p=softmax(weak)`，`pseudo=argmax(p)`，`confidence=max(p)`。
- 候选必须满足严格的 `confidence > tau`。先排序，再按每类额度筛选。相同置信度按袋内原始位置稳定排序。
- `selected[i,j]` 为最终实际入选的 0/1 权重；伪标签、筛选、计数和 λ 均不参与求导。

不读取训练实例真实标签来修正伪标签、确定排序或计算优化损失。

## 实例损失：公式 (2)

令 `s_i=sum_j selected[i,j]`，先在每个袋的已选样本中平均，再对 batch 中所有袋平均：

```text
Li = (1/B) sum_i [ sum_j selected[i,j] * CE(strong[i,j], pseudo[i,j]) / max(s_i,1) ]
```

空选择袋贡献 0，仍包含在外层 B 个袋的分母中。这是对论文未定义 `|Bs_i|=0` 时的最小数值扩展。
不能改成“把所有袋的已选样本拼一起再平均”，那会改变每个袋的权重。

## 每类剩余集合：公式 (3)–(6)

```text
remove[i,j,k] = selected[i,j] AND (pseudo[i,j] == k)
keep[i,j,k]   = NOT remove[i,j,k]
r[i,k]       = sum_j keep[i,j,k] = M - selected_count[i,k]
alpha'[i,k]  = (counts[i,k] - selected_count[i,k]) / r[i,k]
alpha_bar[i,k] = sum_j keep[i,j,k] * p[i,j,k] / r[i,k]
Lb = -(1/B) sum_i sum_k alpha'[i,k] * log(alpha_bar[i,k])
```

同一个已选样本只从其伪标签对应的类集合中移除，仍在其他类的集合中。因此各类分母可能不同，`sum_k alpha'[i,k]` 和 `sum_k alpha_bar[i,k]` 都不保证等于 1。
实现保留论文的 `-sum alpha' log alpha_bar` 表达式；不使用二元交叉熵、不额外加入 `(1-alpha') log(1-alpha_bar)`、不做第二次 softmax、不进行跨类归一化。

每类限额保证剩余类别计数非负。若 `r[i,k]=0`，则整袋都被选为该类，剩余计数也为 0；该坐标对袋损失贡献定义为 0。
代码在 log 空间使用 `logsumexp`，并单独处理空集合，避免 `0*log(0)`、`logsumexp(-inf,...)` 的反向 NaN。

## 总损失与更新：公式 (7)、实验第 4.2 节

```text
lambda = selected.sum() / (B*M)
loss = lambda*Li + (1-lambda)*Lb
```

λ 是经过额度筛选后的**实际入选比例**，每个 minibatch 重算。不是 epoch 进度、不是预设常数，也不是未经限额筛选的候选比例。

弱/强视图由同一个网络联合前向得到。实例损失梯度进入强增强分支；袋损失通过剩余预测进入弱增强分支；共同更新同一网络。EMA 只用于评估，不代替论文中的弱增强模型生成伪标签。
每轮原始袋成员固定；每次前向重新计算集合，不跨 epoch 永久移除样本。

## 可手算的例子

一袋 4 个样本，3 类，类别计数 `[2,1,1]`，弱预测为：

```text
[0.98, 0.01, 0.01]  -> 入选类 0
[0.02, 0.97, 0.01]  -> 入选类 1
[0.80, 0.10, 0.10]  -> 不入选
[0.10, 0.20, 0.70]  -> 不入选
```

τ=0.95 时，剩余集合大小是 `[3,3,4]`，新目标为 `[1/3,0,1/4]`，预测比例为 `[0.92/3,0.31/3,0.82/4]`，λ=1/2。
这与统一剩余袋产生的目标 `[1/2,0,1/2]` 不同。
该例的实例损失、袋损失、总损失和梯度都进入自动测试。

## 与 AHIL 的关系

AHIL 为每个实例计算 `exp(-H_i²/beta_i) * exp(-(H_b-log m[i,k])²/beta_b)`，用于全袋样本平均的伪标签 CE；原袋比例不缩减，总损失为 `Lb + lambda_AHIL*Li`。
Shrink 替换的是这一整套选择/目标/混合方式，不继续叠加 AHIL 的 DEW 权重。
可选 `--method ahil` 仅用于同一工程框架中的诊断对照，使用第二篇论文的 `beta_b=beta_i=1, lambda_AHIL=5`，不同于上游 CLI 的 `5,5,0.5`。缺失类别的 AHIL 权重定义为 0。
