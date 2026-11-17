# 2026-11-17 量化组补证：分工、版本与适用范围

状态：**AproAkk 已核对**本文件列出的文件、哈希和数值；**aly16-k、jxd1111 的分工与
配方/制品确认待本人签字**。本文件不代替本人确认，截止 2026-11-17 18:00 前由本人
回帖或在本文档确认。

## 1. 分工与核对范围

| 责任人 | 负责范围 | 配方 | 制品 | 吞吐出处 | 精度出处 | KV 预算 | 状态 |
|---|---|---|---|---|---|---|---|
| **AproAkk** | gcy W4A8 线；P1-2/P1-3/P2-1/P2-2/P2-3；MMLU-Pro；长上下文；benchmark 归档 | `qwen3_5_35b_a3b_w4a8_experts.yaml`（sha256 `3dd1102e…`） | `/data/gcy/Qwen3.5-35B-A3B-W4A8`（25,099,795,178 B） | `swe_results/ab-*`；`data/results/perf/*` | `mmlu-pro-accuracy/`；`p1-2-layer-error/`；`p1-3-dequant-probe/`；`p2-1-routing-coupling/`；`p2-2-long-context/`；`p2-3-long-context-stability/`；`data/results/longctx*` | 18.85 GiB 正式；32.56/33.49 GiB 容量；19.22 GiB 附录；8 GiB 长上下文 | 已核对 |
| **jxd1111** | GDN-W8A8 制品；toolkit 0.4.2/0.5.0；artifact record；发布 | `qwen35-35b-a3b-w4a8-gdnw8a8.yaml`（sha256 `0bcef0f5…`） | `/data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot`（24,096,662,520 B） | `data/results/perf/{w8a8,w4a8gdn}-18.85g/`、`*-19.22g/`、`*-33.49g/` | `data/results/ppl/`；`data/results/mmlu/`；`data/results/hle/`；`data/results/frontierscience/` | 18.85 / 19.22 / 33.49 GiB | **待签字** |
| **aly16-k** | 待本人确认（建议：官方 W8A8 对照、toolkit 发布校验） | 待确认 | 待确认 | 待确认 | 待确认 | 待确认 | **待签字** |

说明：`owner` 按现有记录中的 `recorded_by`/发布主体预填；不是对个人贡献的最终认定。
AproAkk 的"已核对"仅指 benchmark 包中的文件存在、哈希一致、数值与原始记录一致；
jxd1111/aly16-k 的配方、制品和发布事实仍需本人确认。

## 2. 版本与对应关系

| 版本 | 配方 | Checkpoint | 发布 | 运行时对应 | 备注 |
|---|---|---|---|---|---|
| **0.4.2** | `qwen25-w8a8`（sha256 `9c9385da…`，tag `toolkit-v0.4.2` 核验） | 无 Qwen3.5 checkpoint | PyPI 0.4.2（wheel `57964380…`、sdist `5ff91e53…`）；GitHub Release `411366927` | 记录写 `vllm-ascend-quant-ext 0.5.0`，但 0.5.0 README 写 frozen at 0.4.1a4；**待 jxd1111 确认** | 仅发布/安装验证；无吞吐/PPL 记录 |
| **0.5.0** | `qwen25-w8a8`；`qwen35-35b-a3b-w4a8-gdnw8a8`（sha256 `0bcef0f5…`） | GDN-W8A8：6 分片，24,047,130,272 B | PyPI 0.5.0（wheel `ad3b320f…` 48,923 B；sdist `ea065055…` 66,144 B，2026-11-12）；**无 GitHub Release** | 0.5.0 toolkit 分支 `b93c073` | PyPI wheel 与本地 dist wheel 哈希不同，不强行对应到 `b93c073` |
| **gcy 线** | `qwen3_5_35b_a3b_w4a8_experts.yaml`（sha256 `3dd1102e…`） | gcy W4A8：6 分片，25,099,795,178 B | 非 toolkit 版本；#547 实验线 | 与 0.5.0 GDN 线分开 | 专家 int4、GDN bf16；不要与 GDN-W8A8 混称"W4A8" |
| **官方 W8A8** | `qwen3_5_35b_a3b_w8a8.yaml`（sha256 `b18286e8…`） | W8A8：39,810,890,505 B | 对照基线 | 与 0.5.0 同运行时 | MMLU/PPL/吞吐对照 |

## 3. 适用范围与口径

### 3.1 设备与并行
- 吞吐、TP1 PPL、MMLU-Pro、长上下文：**1×910B3，TP1**。
- TP2 PPL、HLE、FrontierScience、BF16 基线：**2×910B3，TP2**（HLE 为 TP2+EP2）。
- 不得把任务分数的 TP2 数字写成单卡结果；不得把吞吐的 TP1 数字写成双卡结果。

### 3.2 KV 预算（必须与数值同时引用）
| 组 | KV | 用途 |
|---|---:|---|
| 正式同 KV 对照 | **18.85 GiB**（20,242,376,704 B） | W8A8 vs W4A8/GDN C1–C16；也是 #547 gcy 线 P0-1 |
| 19.22 GiB 附录 | **19.22 GiB**（20,637,317,857 B）；两臂 820,136 tokens | W8A8 绝对上限附录；不作加速结论 |
| 容量组 | **33.49 GiB**（35,956,596,497 B） | W4A8/GDN 单卡容量曲线；不可与 18.85 GiB 直接比较 |
| gcy 容量组 | **32.56 GiB**（ab-w4a8max） | gcy W4A8 容量组；不可与 18.85 GiB 直接比较 |
| 长上下文 | **8 GiB**（8,589,934,592 B） | P2-3 8K/32K/128K、260K、261K/260.8K、262,144 capacity smoke |
| PPL | 无显式 KV pin（max-model-len 4096） | TP2/TP1 PPL |
| MMLU-Pro | 无显式 KV pin（32K 上下文，gpu util 0.90） | 四臂准确率 |
| 任务分数（HLE/FS） | 无显式 KV pin | TP2 任务评估 |

### 3.3 精度与长上下文（引用时必须带限定）
- 单卡 PPL：W8A8 **7.1785** vs GDN-W8A8 **7.4009**（+3.10%）。
- MMLU-Pro 12,032：BF16 79.862%；W8A8 79.496%；gcy W4A8 79.746%；GDN-W8A8 **79.172%**。
  GDN-W8A8 vs BF16 **−0.69 pp（p=0.0082）**，vs W8A8 −0.32 pp（p=0.238，不显著）。
- 长上下文：8K/32K/128K + 260K 无可见退化；满 262,144 capacity smoke 两臂均通过；
  261K NIAH C1/C2 未触发；**260.8K 长生成 C3 触发**（W4A8 2/5 vs W8A8 5/5，n=5，
  3 条截断；260K 方向相反）。不声称稳定退化，也不把"无退化"外推到 262,144 任务级。
- HLE `formal_acceptance=false`、自评分、单 trial；FS Research n=60。
- 混合精度 artifact：`prepare-runtime` / `export-contract` 按设计拒绝；artifact
  contract 1.1.0 是单 scheme。

## 4. 已知冲突与缺项（不补实验，只改表述）

1. **0.4.2 runtime counterpart**：0.5.0 vs 0.4.1a4 的冲突未解决，待 jxd1111 确认；
   本文不合并 runtime 版本结论。
2. **0.5.0 GitHub Release**：不存在；只有 PyPI。不要把 PyPI 字节写成 GitHub Release。
3. **0.4.2 runtime E2E 原件**：旧 repo 的两份 E2E 文档不在当前 checkout；只链旧
   Release/固定提交，不复制、不重跑。
4. **满 262,144 任务级**：容量已通过；260.8K 长生成 C3 触发（n=5）是负向/高方差结果，
   不能隐藏，也不能据此声称稳定退化。

## 5. 签字

- [x] **AproAkk**：已核对 benchmark 包的文件、SHA256、数值与 KV 预算；已核对本文件
  与证据索引中的 0.4.2/0.5.0 PyPI 字节和 0.4.2 tag 配方哈希。
- [ ] **aly16-k**：待确认分工/配方/制品/吞吐/精度出处。
- [ ] **jxd1111**：待确认 GDN-W8A8 制品、toolkit 版本与 runtime counterpart。

> 截止 2026-11-17 18:00。未签字的项在 benchmark 报告中保留"待本人确认"，
> 不冒充已核对。
