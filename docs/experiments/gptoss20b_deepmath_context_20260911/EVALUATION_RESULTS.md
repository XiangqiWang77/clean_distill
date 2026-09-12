# GPT-OSS-20B verify context：统一评测结果

快照时间：`2026-09-12T05:20:20.372243+00:00`。评测作业 `25987666`，两张 H200；四组 DeepMath 训练均已完成。评测已完成 5/5 组。

| 模型 | 状态 | 正确 / 总题数 | 准确率 | 相对 vanilla | 达到生成长度上限 |
|---|---|---:|---:|---:|---:|
| Vanilla | 完成 | 111/143 | 77.62% | +0.00 pp | 23/143 (16.08%) |
| OPSD · 16 | 完成 | 108/143 | 75.52% | -2.10 pp | 22/143 (15.38%) |
| LGSD · 16 | 完成 | 106/143 | 74.13% | -3.50 pp | 19/143 (13.29%) |
| OPSD · 64 | 完成 | 92/143 | 64.34% | -13.29 pp | 71/143 (49.65%) |
| LGSD · 64 | 完成 | 90/143 | 62.94% | -14.69 pp | 91/143 (63.64%) |

**本轮四个训练后模型的总体准确率均低于 vanilla，没有得到 OPSD > vanilla 或 LGSD > vanilla 的结果。**

## 哪些局部改善，哪些整体退步

- OPSD · 16 在 AIME 2024 局部改善：17/30 → 19/30（+6.67 个百分点），但不能替代总体结果。
- OPSD · 16 的截断回答较少：23 → 22 条；截断减少也没有自动转化为总体准确率提升。
- LGSD · 16 在 AIME 2024 局部改善：17/30 → 19/30（+6.67 个百分点），但不能替代总体结果。
- LGSD · 16 的截断回答较少：23 → 19 条；截断减少也没有自动转化为总体准确率提升。

| 方法 | 16 步准确率 → 64 步准确率 | 变化 | 截断回答 16 步 → 64 步 |
|---|---:|---:|---:|
| OPSD | 75.52% → 64.34% | -11.19 pp | 22 → 71 |
| LGSD | 74.13% → 62.94% | -11.19 pp | 19 → 91 |

两种方法在 64 步时均比各自的 16 步模型更低，伴随更长的输出和更多截断；16 步和 64 步训练分别从相同 base 初始化。

| 训练步数 | OPSD 正确数 | LGSD 正确数 | OPSD − LGSD |
|---|---:|---:|---:|
| 16 | 108/143 | 106/143 | +1.40 pp |
| 64 | 92/143 | 90/143 | +1.40 pp |

这是同一 seed、相同题目上的观测差异，不能把 2 题的差值解释成已证明稳定优越。


## 已完成模型的改善与退步

| 模型 | 救回 vanilla 错题 | 损失 vanilla 对题 | 净正确题变化 | 平均生成 tokens | 评分器异常 |
|---|---:|---:|---:|---:|---:|
| Vanilla | 0 | 0 | +0 | 4412.4 | 0 |
| OPSD · 16 | 8 | 11 | -3 | 4201.8 | 0 |
| LGSD · 16 | 5 | 10 | -5 | 4245.5 | 0 |
| OPSD · 64 | 2 | 21 | -19 | 7129.2 | 0 |
| LGSD · 64 | 1 | 22 | -21 | 8116.1 | 0 |

OPSD · 16 相对 vanilla 下降 2.10 个百分点；救回 8 题、损失 11 题。

LGSD · 16 相对 vanilla 下降 3.50 个百分点；救回 5 题、损失 10 题。

OPSD · 64 相对 vanilla 下降 13.29 个百分点；救回 2 题、损失 21 题。

LGSD · 64 相对 vanilla 下降 14.69 个百分点；救回 1 题、损失 22 题。

准确率变化与输出长度、截断情况一并报告；这些观测本身不能证明截断是退步的唯一原因。

## 分数据集结果

| 模型 | AMC 2022/2023（83 题） | AIME 2024（30 题） | AIME 2025（30 题） |
|---|---:|---:|---:|
| Vanilla | 76/83 (91.57%) | 17/30 (56.67%) | 18/30 (60.00%) |
| OPSD · 16 | 72/83 (86.75%) | 19/30 (63.33%) | 17/30 (56.67%) |
| LGSD · 16 | 72/83 (86.75%) | 19/30 (63.33%) | 15/30 (50.00%) |
| OPSD · 64 | 65/83 (78.31%) | 15/30 (50.00%) | 12/30 (40.00%) |
| LGSD · 64 | 65/83 (78.31%) | 13/30 (43.33%) | 12/30 (40.00%) |

## 哪些题改善、哪些题退步

完整配对（包括不变题目）保存在 [逐题前后对比 CSV](evaluation_paired.csv)。下表列出所有已完成模型的变化题目；截断栏为 vanilla → 训练后模型。

| 模型 | Query ID | 数据集 | 变化 | 截断前后 |
|---|---|---|---|---|
| OPSD · 16 | `aime24:10c80ba48d1803c48df550827e6e2c3b4b8ef3e0aba30c507798eabf30aeea2e` | aime24 | 错 → 对 | 否 → 否 |
| OPSD · 16 | `aime24:21db44a1eb19aba4568c77a3a1c623e7e57485539a76265b43902887364f6ef3` | aime24 | 错 → 对 | 否 → 否 |
| OPSD · 16 | `aime24:357603fe2192e4dc2b2a52cc90ade2f1f6a3fd452ceec23c194e28a3748e579b` | aime24 | 对 → 错 | 否 → 是 |
| OPSD · 16 | `aime24:692d35427467a74f07dff2b65afbcc2874995945b5ee044320ff04af46ca3187` | aime24 | 对 → 错 | 否 → 是 |
| OPSD · 16 | `aime24:6c7d2070f985c4e7be6467b4bf365f372d64e3b314183113880bb2e4a07f32c2` | aime24 | 错 → 对 | 是 → 是 |
| OPSD · 16 | `aime24:89a3d7b0f4c73a63119f6c1a2bec8ed68f39a8f5d4a3e5ebb12d89eb8b47e857` | aime24 | 错 → 对 | 否 → 否 |
| OPSD · 16 | `aime25:1352bde57c17a969982b9342711da1bea4dbabf0221ae015d0c206ec2f9e30f6` | aime25 | 对 → 错 | 否 → 否 |
| OPSD · 16 | `aime25:656d1b5d0105f71eb14aced1dfa5b1eaa8d9a087cc8e434da877ad2c01870de5` | aime25 | 对 → 错 | 否 → 否 |
| OPSD · 16 | `aime25:70629b6e7c5db3e713f2672a9355856d2fc7b17285add34b1f7fd1d289c9e173` | aime25 | 错 → 对 | 否 → 否 |
| OPSD · 16 | `aime25:757329975fcc4a0df83dd88a44c25217dc18c4aab9c3e2b3c4efe98ef4fcb22b` | aime25 | 错 → 对 | 是 → 是 |
| OPSD · 16 | `aime25:aa6399130b64e6a0447fc114a07afb7125a71237472ea6f12051bf36f62f9d77` | aime25 | 对 → 错 | 否 → 否 |
| OPSD · 16 | `amc23:07b9c484882e267d20422574c9c903f676c255259b9ca652377701c5359e052b` | amc23 | 对 → 错 | 否 → 否 |
| OPSD · 16 | `amc23:2a63d16d2b0a8fd7612aa93f22439b54aa86f21a6301662771b31c5d1979eea0` | amc23 | 对 → 错 | 否 → 否 |
| OPSD · 16 | `amc23:34b7e15dd5e80ce908acd5ffc9ef6872ad9a431acd41064f96910a3a21cda033` | amc23 | 对 → 错 | 否 → 否 |
| OPSD · 16 | `amc23:41eb3f5f38c5931eaf64054a5654ff42abafa6d50eb6dccf93159cabdf2435fe` | amc23 | 错 → 对 | 是 → 是 |
| OPSD · 16 | `amc23:43a47a3682574d0013250823ee0b4e09f230a09d5c3cea9702509249574b0f61` | amc23 | 对 → 错 | 否 → 否 |
| OPSD · 16 | `amc23:d31000037cb3644393d26e571c65e04594f86638450b81264d7afaa99a381424` | amc23 | 对 → 错 | 否 → 否 |
| OPSD · 16 | `amc23:ead36adb680e3bc2f013d37f49c85242ff58e44c43942c0e420bc945062da169` | amc23 | 对 → 错 | 否 → 是 |
| OPSD · 16 | `amc23:ee87beeb37a38aaf69fbae16f50f1a29b67ba132258875a444b04974a83a1f36` | amc23 | 错 → 对 | 是 → 否 |
| LGSD · 16 | `aime24:21db44a1eb19aba4568c77a3a1c623e7e57485539a76265b43902887364f6ef3` | aime24 | 错 → 对 | 否 → 否 |
| LGSD · 16 | `aime24:692d35427467a74f07dff2b65afbcc2874995945b5ee044320ff04af46ca3187` | aime24 | 对 → 错 | 否 → 是 |
| LGSD · 16 | `aime24:89a3d7b0f4c73a63119f6c1a2bec8ed68f39a8f5d4a3e5ebb12d89eb8b47e857` | aime24 | 错 → 对 | 否 → 否 |
| LGSD · 16 | `aime24:d8c514307d5e703048cb5b6165cf10c78d771e2166f107f48b586d9289d2e736` | aime24 | 错 → 对 | 是 → 否 |
| LGSD · 16 | `aime25:1352bde57c17a969982b9342711da1bea4dbabf0221ae015d0c206ec2f9e30f6` | aime25 | 对 → 错 | 否 → 否 |
| LGSD · 16 | `aime25:755f37ac64c45cd71e3e6d556bed121a74b02b22dca1855b3d76ce7e44e1964b` | aime25 | 对 → 错 | 否 → 否 |
| LGSD · 16 | `aime25:9592e82151960bdda3419ee55f677a8fb7a54bbbc762295146f0c43653c7ae08` | aime25 | 对 → 错 | 是 → 否 |
| LGSD · 16 | `amc23:1b0d9cd3c430e32006453ba2e3355555109007e679f9ef9ae8480e4573e0bcfd` | amc23 | 对 → 错 | 是 → 是 |
| LGSD · 16 | `amc23:2a63d16d2b0a8fd7612aa93f22439b54aa86f21a6301662771b31c5d1979eea0` | amc23 | 对 → 错 | 否 → 否 |
| LGSD · 16 | `amc23:34b7e15dd5e80ce908acd5ffc9ef6872ad9a431acd41064f96910a3a21cda033` | amc23 | 对 → 错 | 否 → 否 |
| LGSD · 16 | `amc23:41eb3f5f38c5931eaf64054a5654ff42abafa6d50eb6dccf93159cabdf2435fe` | amc23 | 错 → 对 | 是 → 否 |
| LGSD · 16 | `amc23:76e2a609540741d3f60c8cfe8249ff0fe44f0ffd12c8a99b86aac0b540c6a572` | amc23 | 对 → 错 | 否 → 否 |
| LGSD · 16 | `amc23:d529b529d9b906fd26d8434344905f5db7c6951fd28681385a6016dbe63d8b6c` | amc23 | 对 → 错 | 是 → 否 |
| LGSD · 16 | `amc23:ead36adb680e3bc2f013d37f49c85242ff58e44c43942c0e420bc945062da169` | amc23 | 对 → 错 | 否 → 否 |
| LGSD · 16 | `amc23:ee87beeb37a38aaf69fbae16f50f1a29b67ba132258875a444b04974a83a1f36` | amc23 | 错 → 对 | 是 → 否 |
| OPSD · 64 | `aime24:10c80ba48d1803c48df550827e6e2c3b4b8ef3e0aba30c507798eabf30aeea2e` | aime24 | 错 → 对 | 否 → 是 |
| OPSD · 64 | `aime24:21db44a1eb19aba4568c77a3a1c623e7e57485539a76265b43902887364f6ef3` | aime24 | 错 → 对 | 否 → 是 |
| OPSD · 64 | `aime24:5650d2a0767a53a577e51aab7a9b3cafb4f91562ad2544bb088c6cfc577cf364` | aime24 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `aime24:692d35427467a74f07dff2b65afbcc2874995945b5ee044320ff04af46ca3187` | aime24 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `aime24:7f3d2be0982807560fb814e16f111ef6e38f04e32a108355530732c9c64c22fb` | aime24 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `aime24:bb57bbb4f19d36094fcb0392d8ef63461045fac1aa02c1bc063102c072f3452a` | aime24 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `aime25:1352bde57c17a969982b9342711da1bea4dbabf0221ae015d0c206ec2f9e30f6` | aime25 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `aime25:656d1b5d0105f71eb14aced1dfa5b1eaa8d9a087cc8e434da877ad2c01870de5` | aime25 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `aime25:7528e62e74eabdd96e6c6e3f542e9076010c55bf9be70e28509b4c8672b20141` | aime25 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `aime25:755f37ac64c45cd71e3e6d556bed121a74b02b22dca1855b3d76ce7e44e1964b` | aime25 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `aime25:981c7f2ec58bbc629a520abdcc2d49c69ec86d2c6b2bc574b71d50c2d1dcce0e` | aime25 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `aime25:a7f58aabb0a8393f9c5d00fb3ba6df269ecd0fca3e5f447b0402ab0facb6d3ad` | aime25 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `amc23:1b0d9cd3c430e32006453ba2e3355555109007e679f9ef9ae8480e4573e0bcfd` | amc23 | 对 → 错 | 是 → 是 |
| OPSD · 64 | `amc23:34055b747dd7589da7426630fa1d3913c41de6c93329b6a544119e24eaaad6a5` | amc23 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `amc23:44e8c69a75fc70f09d64f2b16fb15c6b19f9783318264438b48a8b3191247436` | amc23 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `amc23:556fcb2237639a503bd11388a346f5ea95e1ec98ba7753497362539b5ec4c57e` | amc23 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `amc23:6529281306a5556f2db9b18525e46452f6f434440ea85593b77c4178f76b4486` | amc23 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `amc23:76e2a609540741d3f60c8cfe8249ff0fe44f0ffd12c8a99b86aac0b540c6a572` | amc23 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `amc23:7e499e5f0882293d9fda2549375fc120f08583c1bde929b8427c21998cbad5e2` | amc23 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `amc23:a19e295c1cae405e3b0b84c59959b1f70176f8c06a1b240c9f59915fd678b522` | amc23 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `amc23:b81a8c8267075f99502d08dc4804a698f5ad7408cd5b690fa20138019fb98055` | amc23 | 对 → 错 | 否 → 否 |
| OPSD · 64 | `amc23:bc3ba284e49260cf117c0de67d0e8298b140d1b0c85998e84a01813dd192d43a` | amc23 | 对 → 错 | 否 → 是 |
| OPSD · 64 | `amc23:ead36adb680e3bc2f013d37f49c85242ff58e44c43942c0e420bc945062da169` | amc23 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `aime24:357603fe2192e4dc2b2a52cc90ade2f1f6a3fd452ceec23c194e28a3748e579b` | aime24 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `aime24:5650d2a0767a53a577e51aab7a9b3cafb4f91562ad2544bb088c6cfc577cf364` | aime24 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `aime24:692d35427467a74f07dff2b65afbcc2874995945b5ee044320ff04af46ca3187` | aime24 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `aime24:7f3d2be0982807560fb814e16f111ef6e38f04e32a108355530732c9c64c22fb` | aime24 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `aime25:1352bde57c17a969982b9342711da1bea4dbabf0221ae015d0c206ec2f9e30f6` | aime25 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `aime25:596ba4660f86c56a92b3ee64c2caf803865c1589deb8c2a1a6e471ed83695401` | aime25 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `aime25:5997376b7b910b0e745067c7631f89d007c8f520a11ba656e15181eb0d45e3ab` | aime25 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `aime25:656d1b5d0105f71eb14aced1dfa5b1eaa8d9a087cc8e434da877ad2c01870de5` | aime25 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `aime25:755f37ac64c45cd71e3e6d556bed121a74b02b22dca1855b3d76ce7e44e1964b` | aime25 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `aime25:951031f375bc05ca792f45e75f51e81f67054c08c48c30225d3c269cba80e733` | aime25 | 错 → 对 | 否 → 是 |
| LGSD · 64 | `aime25:981c7f2ec58bbc629a520abdcc2d49c69ec86d2c6b2bc574b71d50c2d1dcce0e` | aime25 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `aime25:ed910555a62ce77895b83babe2779d049772d20cc13c1fdcecc116d97abd9ae4` | aime25 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `amc23:2bcd935ef65a0ce135babeb7581864fb17098e66cd325cb4de41897d1b4b2fad` | amc23 | 对 → 错 | 是 → 是 |
| LGSD · 64 | `amc23:2fef41e34422fcff8a92f23a1e8cefe0c84eeb54096005d945069a62f6a73b7d` | amc23 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `amc23:34055b747dd7589da7426630fa1d3913c41de6c93329b6a544119e24eaaad6a5` | amc23 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `amc23:44e8c69a75fc70f09d64f2b16fb15c6b19f9783318264438b48a8b3191247436` | amc23 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `amc23:556fcb2237639a503bd11388a346f5ea95e1ec98ba7753497362539b5ec4c57e` | amc23 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `amc23:76e2a609540741d3f60c8cfe8249ff0fe44f0ffd12c8a99b86aac0b540c6a572` | amc23 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `amc23:7e499e5f0882293d9fda2549375fc120f08583c1bde929b8427c21998cbad5e2` | amc23 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `amc23:a19e295c1cae405e3b0b84c59959b1f70176f8c06a1b240c9f59915fd678b522` | amc23 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `amc23:bc3ba284e49260cf117c0de67d0e8298b140d1b0c85998e84a01813dd192d43a` | amc23 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `amc23:d31000037cb3644393d26e571c65e04594f86638450b81264d7afaa99a381424` | amc23 | 对 → 错 | 否 → 是 |
| LGSD · 64 | `amc23:ead36adb680e3bc2f013d37f49c85242ff58e44c43942c0e420bc945062da169` | amc23 | 对 → 错 | 否 → 是 |

## 评测口径

143 道 AMC/AIME held-out 题，与 DeepMath 训练题无交集。每题一次配对采样，seed 20260809；所有模型使用相同题目、普通 student prompt、medium reasoning、固定日期、temperature 0.6、top-p 0.95、top-k 20、repetition_penalty=1.1，以及 10,240 generated-token 上限。

所有模型统一使用 vLLM MXFP4 Marlin 内核；vanilla 不加载 adapter。采用 full-response Math-Verify 0.9.0 评分，截断输出也会按已生成文本评分，因此准确率与截断率分别列出。

Vanilla 使用这轮已经完成的 111/143 基线，不会再次生成。旧 context screen 的 83/86 与 verify 的 84/86 是另一数据集上的 teacher 提示筛选结果，不能当作本次训练后的 student 表现，也不能保证在 AMC/AIME 上改善。单 seed 结果用于描述这轮实验，不等同于统计显著性证明。

## 可核查数据

- [完整评测汇总 JSON](evaluation_results.json)
- [已完成模型逐题评分 CSV](evaluation_per_query.csv)
- [逐题前后对比 CSV](evaluation_paired.csv)
- [固定评测协议与输入哈希](evaluation_protocol.json)
- [四组训练结果](TRAINING_RESULTS.md)
