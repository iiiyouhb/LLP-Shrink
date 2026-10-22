# Method and implementation

The objective is implemented in `llp_shrink/losses.py::shrink_loss`.
The equations below follow Sections 3.1–3.3 of *Learning from Label Proportions with Shrinking Bag*.

## Inputs and selection

Weak and strong logits have shape `[B, M, C]`, where `B` is the number of bags, `M` the bag size, and `C` the number of classes. Integer `counts[B,C]` sum to `M` within each bag.

Pseudo-labels are the argmax of the weak-view prediction. Candidates satisfy the strict condition `confidence > threshold`. Within each bag and predicted class, candidates are ranked by confidence and selected up to that class's known count. Ties retain bag-position order.

Pseudo-labels, selection masks, target counts, and loss mixing ratios are detached from the gradient computation.

## Instance-level loss

With `selected[i,j]` the binary selection mask and `s_i` its count in bag `i`:

```text
Li = mean_i( sum_j selected[i,j] * CE(strong[i,j], pseudo[i,j]) / max(s_i, 1) )
```

The loss is averaged within each selected bag and then across all bags. A bag with no selected instances contributes zero and remains in the outer average.

## Class-specific residual bag loss

Each class has its own residual set. An instance selected as class `k` is removed from that class's set and retained in the sets for other classes:

```text
remove[i,j,k] = selected[i,j] AND (pseudo[i,j] == k)
keep[i,j,k] = NOT remove[i,j,k]
r[i,k] = sum_j keep[i,j,k]
target[i,k] = (counts[i,k] - sum_j remove[i,j,k]) / r[i,k]
prediction[i,k] = sum_j keep[i,j,k] * softmax(weak[i,j])[k] / r[i,k]
Lb = -mean_i( sum_k target[i,k] * log(prediction[i,k]) )
```

These coordinates use different instance sets; they are not normalized again across classes. Per-class selection limits keep residual counts nonnegative. An empty residual set contributes zero. `log_softmax` and masked `logsumexp` provide stable log probabilities and gradients.

## Combined objective

```text
lambda = selected.sum() / (B*M)
loss = lambda * Li + (1-lambda) * Lb
```

The ratio uses actual selections after applying class limits and is recalculated for every minibatch. The instance term trains strong-view predictions; the bag term trains weak-view predictions. Both update the same classifier. EMA is used for evaluation.

Bag membership remains fixed across epochs. Selection and residual sets are recomputed at each forward pass, with no permanent removal of instances.

## Numerical example

Consider one bag with class counts `[2,1,1]`, threshold `0.95`, and weak probabilities:

```text
[0.98, 0.01, 0.01]
[0.02, 0.97, 0.01]
[0.80, 0.10, 0.10]
[0.10, 0.20, 0.70]
```

The first two instances are selected. Residual set sizes are `[3,3,4]`, targets are `[1/3,0,1/4]`, predicted proportions are `[0.92/3,0.31/3,0.82/4]`, and `lambda=1/2`.
`tests/test_losses.py` checks this example, quota handling, empty sets, and gradient correctness.
