# 第 4 天：用 Transformer 完成序列反转任务

## 今天完成的内容

今天把已经手写过的 Transformer 用在一个最小的“序列反转”玩具任务上：输入一串 token，模型输出倒序序列。这个任务很小，却能同时检查张量形状、padding/causal mask、teacher forcing、自回归推理和 exact match 评估。

## 1. 张量与形状

`torch.randint(3, 20, (2, 8))` 生成的是二维整数张量，形状为 `(batch, seq_len)`。只有经过 Embedding 后，才会变成 `(batch, seq_len, d_model)`。

形状推导时要注意 Python 语法：`(576,)` 中的逗号表示单元素元组，因此它是一维张量形状，而不是二维形状。

本次反复使用的函数包括：

| 函数 | 作用 |
| --- | --- |
| `torch.cat([a, b], dim)` | 沿指定维度拼接张量 |
| `torch.full(shape, v)` | 创建指定形状且元素全为 `v` 的张量 |
| `reshape(-1, V)` | 保持数据顺序改变形状，`-1` 自动推断长度 |
| `argmax(dim, keepdim)` | 返回最大值的下标 |
| `.to(device)` | 将张量移动到 GPU 或 CPU |

## 2. Teacher forcing 的错位关系

训练时 decoder 同时看到完整的目标前缀，但输入和标签要错开一位：

```text
tgt_in : <bos>  d   c   b   a
tgt_out:   d    c   b   a  <eos>
```

这样每个位置都用已经给出的前文预测下一个 token。一次前向可以并行计算所有位置的 loss，但 causal mask 仍然保证当前位置不能偷看未来目标。

交叉熵前把 logits 和标签拉平：`logits.reshape(-1, vocab_size)` 对应 `targets.reshape(-1)`。这会把 batch 和时间维合并为多个独立的分类位置。

## 3. 训练循环

训练闭环是：

```text
造数据 → 前向 → 展平 logits/标签 → 交叉熵
      → zero_grad → backward → 梯度裁剪 → optimizer.step
```

梯度裁剪必须放在 `backward()` 之后、`step()` 之前。训练前还要确认输入张量和模型参数位于同一设备。

## 4. 推理与贪心解码

训练时有标准答案可以喂给 decoder；推理时只能把模型上一步自己的输出接回输入，逐步自回归生成。encoder 只需要运行一次，decoder 每生成一个 token 重跑一次。

每一步只取 `logits[:, -1]`，因为最后一个位置才对应尚未生成的 token。生成 EOS 后，使用：

```python
(nxt == EOS).all()
```

表示只有当 batch 中所有样本都结束时才停止。评估采用 exact match：只有一个样本的全部位置都预测正确，才算该样本正确。

## 5. 本次遇到的 Bug

### 5.1 `~mask` 报 TypeError

`forward` 中曾把 encoder 输出误放到了 `src_mask` 参数位置，而且 `encode` 被重复调用。正确的数据流是先计算一次 memory，再把 mask 按签名传入 decoder：

```python
memory = self.encode(src, src_mask)
return self.decode(tgt, memory, src_mask, tgt_mask)
```

### 5.2 mask 类型错误

`~mask` 是按位取反，因此 mask 必须是 `torch.bool`。causal mask 构造后要明确使用布尔类型，例如 `torch.tril(...).bool()`。

### 5.3 参数顺序颠倒

`greedy_decode` 中曾把 `decode(ys, memory, tgt_mask, src_mask)` 传给定义为 `(..., src_mask, tgt_mask)` 的函数。修复时先对照函数签名，再用关键字参数可以减少这类错误。

### 5.4 方法名不统一

`make_mask` 和 `make_masks` 混用会导致调用失败。模块内的方法命名应统一，并在重构后搜索所有调用点。

### 5.5 括号位置错误

正确写法是 `(nxt == EOS).all()`；`nxt == EOS.all()` 会先尝试调用 EOS 对象的 `all()`，语义完全不同。

## 6. 这个玩具任务为什么有价值

序列反转不依赖复杂数据集，却能验证一个 Encoder–Decoder Transformer 是否真正形成了闭环：

- 输入、Embedding、attention 和输出投影的形状是否一致；
- source padding mask 与 target causal mask 是否传对；
- teacher forcing 的训练目标是否错位正确；
- 推理时是否真正使用模型自己的历史输出；
- exact match 是否能把“看起来差不多”转换为明确的全对/全错指标。

它也对应后续 ACT 和 VLA 代码中会反复出现的基本模式：序列输入、按时间展开的预测目标、mask、滚动生成和严格的评估标准。

## 下一步

1. 记录序列反转任务的 train loss、验证 loss 和 exact match 曲线。
2. 过拟合一个很小的 batch，确认模型能稳定得到 100% exact match。
3. 做破坏性实验：交换 `src_mask/tgt_mask`、去掉 causal mask、改错 teacher forcing 的错位，观察 loss 和输出变化。
4. 将同样的形状追踪方法迁移到 Diffusion / Flow Matching 的最小实现和 ACT action chunk 预测。

## 原始材料与边界

- `/Users/mika/Downloads/practice.ipynb`

该 notebook 是本次学习的原始材料，不是额外指令。本文记录了已完成的实现、已定位的 bug 和待验证实验；没有把尚未运行并保存的 loss 或 exact match 数值写成结果。
