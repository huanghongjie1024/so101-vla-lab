# 第 1 天：Transformer 基础、数值稳定性与注意力机制

## 今天完成的内容

今天围绕 Transformer 的底层计算，重点理解了 Softmax、交叉熵、Scaled Dot-Product Attention、mask、Causal Mask、Multi-Head Attention 和 Dropout 的行为。

## 1. 数值稳定的 Softmax

减去最大值不会改变 Softmax 的结果，因为分子和分母同时乘以同一个 `exp(-c)` 后会约掉。这个操作主要防止上溢和下溢：`exp(x)` 超过 float32 上限时会产生 `inf`，而很小的指数可能全部变成 0，最终出现 `inf / inf` 或 `0 / 0`。

测试 `[[1, 2], [3, 4]]` 看不出差异，是因为数值规模太小。使用 1000 和 -1000 一类的输入，稳定版本和朴素版本的差异会直接出现。

## 2. 交叉熵直接接收 logits

先 Softmax 再取 log 可能让小概率下溢成 0，进而得到 `log(0) = -inf`。`log_softmax` 使用 log-sum-exp 技巧保持数值稳定，梯度可以简化为 `p - one_hot`。

实践规则：`F.cross_entropy` 直接传入 logits，不要提前手动 Softmax。

## 3. Scaled Dot-Product Attention 与 mask

Attention 的核心公式是：

```text
softmax(QK^T / sqrt(d_k)) V
```

`masked_fill` 是 `torch.Tensor` 自带的方法。使用 `scores.masked_fill(~mask, -inf)` 可以把不允许访问的位置设成负无穷，Softmax 后这些位置的权重为 0。

本次采用的约定是：`mask=True` 表示允许访问。mask 必须是布尔类型，否则 `~` 会变成按位取反。整行都被 mask 时会产生 `nan`，这是需要单独防范的边界情况。

## 4. Causal Mask

```python
torch.tril(torch.ones(T, T, dtype=torch.bool))
```

下三角 mask 中，第 `i` 行表示位置 `i` 能够看到哪些位置。对 `(B, H, T, T)` 的 attention score，`(T, T)` mask 可以通过广播使用；广播从最右侧维度对齐，缺少的维度按 1 补齐。

验证 causal mask 的标准方法是：只修改未来位置的输入，检查过去位置的输出是否保持不变。

## 5. Multi-Head Attention

Multi-Head Attention 的流程是：

1. 对输入做线性投影；
2. 拆分成多个 head；
3. 每个 head 独立计算 attention；
4. 合并 head；
5. 通过输出投影 `W_O`。

`__init__` 中放置有参数的层，保证参数只创建一次并参与保存和训练；`forward` 中放置数据计算流程。把 `nn.Linear` 写进 `forward` 会导致每次重新创建随机层，模型无法正常学习。

`transpose` 后如果要使用 `view`，需要先调用 `.contiguous()`。`mask[None, None]` 是显式补出 batch 和 head 维度，得到 `(1, 1, T, T)`；对于依赖 batch 的 padding mask，需要按照实际数据形状显式构造。

## 6. Dropout 与 `eval()`

训练模式下 Dropout 会随机丢弃元素，因此同一个输入的两次输出可能不同。做验证、推理和确定性断言时使用 `model.eval()`；训练时使用 `model.train()`。当 `dropout=0.0` 时也会失去随机性，但正式推理仍应显式切换到 `eval()`。

## Notebook 中记录的几个收获点

| 问题 | 结论 |
| --- | --- |
| `self.dropout = drop_out` | 只保存了数字；要写成 `nn.Dropout(p)` 才会真正执行 Dropout |
| 类名拼写 | `MutleHeadAttention` 与 `multiheadattention` 应统一命名 |
| mask 形状 | 建议明确写成 `(1, 1, Tq, Tk)` |
| 注释中的 `t` | 用 `Tq`、`Tk` 区分 query 和 key/value 长度，cross-attention 中两者可以不同 |

## 下一步

1. 补完 Encoder 层：FFN、LayerNorm 和 Pre-LN 残差结构；
2. 加入位置编码并搭建 Decoder：self-attention + cross-attention；
3. 在序列反转任务上验证整个模型是否收敛；
4. 做破坏性实验：去掉 `~`、`.contiguous()` 或 `sqrt(d_k)`，记录对应错误；
5. 不看代码，在 15 分钟内默写带 mask 的 MHA，并与 `F.scaled_dot_product_attention` 对齐。

## 原始材料

- [原始 Transformer notebook](https://github.com/huanghongjie1024/so101-vla-lab/blob/main/notebooks/transformer_text.ipynb)

这篇记录整理自当天的学习笔记。Notebook 中的实验代码保留原样，后续会单独修正尚未完成的实现并补充测试结果。
