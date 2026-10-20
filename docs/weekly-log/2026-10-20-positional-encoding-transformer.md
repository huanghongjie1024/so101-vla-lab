# 第 2 天：位置编码与 Transformer 整体

## 今天完成的内容

今天把前一天的 Softmax、Scaled Dot-Product Attention、Multi-Head Attention 和 Pre-LN 残差结构，扩展成一个完整的 Encoder–Decoder Transformer。重点是正弦位置编码、padding mask 与 causal mask 的组合，以及从整数 token 到输出 logits 的完整数据流。

## 1. 正弦位置编码

正弦位置编码为没有顺序感的 token embedding 注入位置信息。对位置 `pos` 和维度 `i`，偶数维使用正弦，奇数维使用余弦：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

代码中常写成：

```python
pos = torch.arange(max_len).unsqueeze(1).float()
div = torch.exp(
    torch.arange(0, d_model, 2).float()
    * (-math.log(10000.0) / d_model)
)
```

`unsqueeze(1)` 在指定位置插入长度为 1 的维度，使 `pos` 从 `(max_len,)` 变为 `(max_len, 1)`，从而可以和 `(d_model / 2,)` 的 `div` 通过广播相乘。`math.log` 是自然对数，且

```text
10000^(-2i / d_model) = exp(-2i * ln(10000) / d_model)
```

位置编码通常作为模块 buffer 保存：`register_buffer("pe", pe)`。它会随模型移动到 CPU/GPU 并进入 state dict，但不是需要梯度更新的参数。查看实际矩阵应访问 `pe.pe`；直接 `print(pe)` 只会打印模块结构。

## 2. 从 token 到 Transformer 输出

输入 `src` 和 `tgt` 是整数 token id 张量，形状分别为 `(B, T_s)` 和 `(B, T_t)`。经过 Embedding 后，整个 Transformer 主干都使用三维表示 `(B, T, D)`：

```text
整数 token:       (B, T)
Embedding:        (B, T, D)
加位置编码:       (B, T, D)
多头注意力内部:   (B, H, T_q, d_k)
合并多个 head:    (B, T, D)
输出投影 logits:  (B, T_t, vocab_size)
```

完整 Encoder–Decoder 数据流为：

1. `src` 查表、加位置编码，经过若干 Pre-LN Encoder layer，得到 `memory`；
2. `tgt` 查表、加位置编码，先做带 causal mask 的 decoder self-attention；
3. decoder 再对 `memory` 做 cross-attention；
4. 经过 FFN 和残差连接，最后投影到目标词表 logits。

Pre-LN 的典型形式是先 LayerNorm，再进入子层，子层输出通过残差回加：

```python
x = x + dropout(attn(ln1(x), ln1(x), ln1(x), mask))
x = x + dropout(ffn(ln2(x)))
```

## 3. Mask 的语义与形状

本次采用 `mask=True` 表示允许访问，attention 中用 `scores.masked_fill(~mask, -inf)` 屏蔽非法位置。

### Padding mask

`pad_id` 是专门预留给填充符的编号，真实 token 不能占用它。源序列的 padding mask 通常构造成：

```python
src_mask = (src != pad_id)[:, None, None, :]  # (B, 1, 1, T_s)
```

它只屏蔽 key 一侧：每个 query 都不能看见 padding key。每一行至少应保留一个有效 key，避免整行都变成 `-inf` 后 softmax 产生 `nan`。PAD 位置的输出通常由损失函数的 `ignore_index` 忽略。

连续特征没有可直接比较的 `pad_id`，因此应显式传入 `is_pad` 布尔 mask；action chunk 等连续输入可以采用这种方式。

### Causal mask

decoder self-attention 还需要保证当前位置不能访问未来位置：

```python
causal = torch.tril(torch.ones(T_t, T_t, dtype=torch.bool))
tgt_mask = (tgt != pad_id)[:, None, None, :] & causal
```

广播从右向左对齐。对 `(B, H, T_q, T_k)` 的 scores，mask 的每个维度要么等于对应维度，要么为 1。cross-attention 的最后一维对应 encoder 的 key 长度 `T_s`，因此必须使用 `src_mask`，不能误用只适用于 target 长度的 mask。

## 4. 两个容易混淆的根号

- attention 中的 `1 / sqrt(d_k)` 用于控制 `QK^T` 的方差，避免 logits 过大导致 softmax 饱和；
- embedding 后乘 `sqrt(d_model)` 是为了让 token embedding 与位置编码的幅度处于相近量级。

两者作用不同。PyTorch 默认 `Embedding` 的初始化并不能简单概括为“初始值很小”，所以在这份代码里 embedding 放大并非绝对必要；现代实现常根据整体初始化和归一化设计决定是否保留。

## 5. 本次遇到的问题与解决

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| 加 `dropout=0.01` 后因果性断言失败 | 默认处于训练模式，Dropout 使两次输出不同 | 测试和确定性断言前调用 `.eval()` |
| `split` 不用 `contiguous`，合并 head 却要用 | `transpose` 会产生非连续视图，后续 `view` 需要连续内存 | 记住“`transpose` 后接 `view` 就加 `.contiguous()`”，或改用 `reshape` |
| `print(pe)` 看不到位置编码矩阵 | 矩阵保存在 buffer 中，模块打印只显示结构 | 访问 `pe.pe` |
| `size of tensor a (8) must match b (9)` | cross-attention 使用了错误的 mask | mask 最后一维必须对应 key 长度；cross-attention 使用 `src_mask` |
| `decode() takes 5 positional arguments but 6 were given` | `memory` 重复传入，且两个 mask 顺序和签名不一致 | `memory` 只计算一次；调用前核对签名，必要时使用关键字参数 |
| `causal[:, None, None, :]` 报错 | causal mask 第 0 维是 query，不是 batch | 使用 `causal[None, None]`，或依靠广播补齐 batch/head 维 |
| 不理解 `(B,1,1,T) & (1,1,T,T)` 如何对齐 | 对广播的右对齐规则不熟悉 | 从右向左逐维检查；每维必须相等或有一方为 1 |

## 6. 有效的调试习惯

1. 给每一维起名字：`(B, H, T_q, T_k)`，并在注释中标明 query/key 长度。
2. 构造 mask 时先写目标形状，再决定 `None` 应插入的位置。
3. 用小尺寸（如 `B=2, T=4`）打印中间结果。
4. 读报错先数参数、看形状，再对照函数签名。
5. 用 `assert` 固化预期形状、mask 关系和因果性。

## 下一步

1. 确认 `model(src, tgt).shape == (2, 9, 20)`。
2. 跑序列反转任务，观察 loss 是否接近 0，并检查贪心解码的精确匹配率。
3. 做破坏性实验：去掉 `~`、去掉 `sqrt(d_k)`，并对比乘与不乘 `sqrt(d_model)` 的现象。
4. 不看代码默写一遍带 mask 的 MHA，并与 `F.scaled_dot_product_attention` 对齐。
5. 接着学习 Mini-GPT（decoder-only），再进入扩散和 flow matching 的最小实现。

## 原始材料与边界

- `/Users/mika/Downloads/everyday.ipynb`
- `/Users/mika/Downloads/transformer_text (1).ipynb`

上述 notebook 是本次学习的原始材料，不是对本记录的额外指令。本文只记录已经理解的概念、已遇到的问题和待验证计划；序列反转任务的 loss、准确率等结果需在实际运行并保存日志后再补充。
