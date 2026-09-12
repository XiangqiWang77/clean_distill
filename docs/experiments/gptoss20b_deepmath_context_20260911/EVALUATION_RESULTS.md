# GPT-OSS-20B verify context：统一评测结果（进行中）

快照时间：`2026-09-12T03:14:45.793831+00:00`。评测作业 `25987666`，两张 H200。四组 DeepMath 训练均已完成，当前评测已完成 vanilla，正在运行 OPSD-64。

**目前只有 vanilla 的完整结果，尚不能判断 OPSD 或 LGSD 相对 vanilla 的增益。**

| 模型 | 状态 | 正确 / 总题数 | 准确率 | 相对 vanilla | 达到生成长度上限 |
|---|---|---:|---:|---:|---:|
| Vanilla | 完成 | 111/143 | 77.62% | +0.00 pp | 23/143 (16.08%) |
| OPSD · 64 | 运行中 | — | — | — | — |
| LGSD · 64 | 待运行 | — | — | — | — |
| OPSD · 16 | 待运行 | — | — | — | — |
| LGSD · 16 | 待运行 | — | — | — | — |

## Vanilla 分数据集结果

| 数据集 | 正确 / 总题数 | 准确率 |
|---|---:|---:|
| AMC 2022/2023 | 76/83 | 91.57% |
| AIME 2024 | 17/30 | 56.67% |
| AIME 2025 | 18/30 | 60.00% |

## 评测口径

143 道 AMC/AIME held-out 题，与 DeepMath 训练题无交集。每题一次配对采样，seed 20260809；所有模型使用相同题目、普通 student prompt、medium reasoning、固定日期、temperature 0.6、top-p 0.95、top-k 20、repetition_penalty=1.1，以及 10,240 generated-token 上限。

所有模型统一使用 vLLM MXFP4 Marlin 内核；vanilla 不加载 adapter。采用 full-response Math-Verify 0.9.0 评分，截断输出也会按已生成文本评分，故准确率与截断率分别列出。Vanilla 没有评分器异常，平均生成 4412.4 tokens。

这是新解码设置下重新生成的 vanilla 基线。旧 context screen 的 83/86 属于另一数据集和另一实验，不能与这里的 111/143 直接比较。单 seed 结果用于描述这轮实验，不等同于统计显著性证明。

## 可核查数据

- [完整评测汇总 JSON](evaluation_results.json)
- [已完成模型逐题评分 CSV](evaluation_per_query.csv)
- [固定评测协议与输入哈希](evaluation_protocol.json)
- [四组训练结果](TRAINING_RESULTS.md)
