# 0.4.2 / 0.5.0 证据索引（2026-11-17 合并稿）

用途：把 `ascend-quant-toolkit` 0.4.2 与 0.5.0 两个版本的发布证据、配方、checkpoint
和 benchmark 记录串到同一张索引中。只收录可追溯到原始记录的内容；无法核验的原件在
§5 明确写出限制，不用推测值补齐。

状态：AproAkk 已核对本索引与 benchmark 报告中的数值/哈希；aly16-k、jxd1111 的
分工与配方确认待本人签字（见 `contributions.md`）。

## 1. 版本与发布对象

### 1.1 ascend-quant-toolkit 0.4.2（已发布）

| 项 | 值 |
|---|---|
| 版本/日期 | 0.4.2 / 2026-10-13 |
| 源码修订 | `50ba3f8`（`main`）；peeled commit `50ba3f8fdad986fa647b0d55918d13b47a25ec42` |
| Tag | `toolkit-v0.4.2`，annotated tag object `be9393676a0808733302559f6e513672c08dfa1e` |
| GitHub Release | id `411366927`，2026-10-13T06:43:03Z；https://github.com/jxd1111/Ascend-LLM-quant/releases/tag/toolkit-v0.4.2 |
| PyPI wheel | `ascend_quant_toolkit-0.4.2-py3-none-any.whl`，41,003 B，sha256 `57964380f85275e5e7d32d14bc6df94c020d6be70bfe47da0dcc430d946d2470` |
| PyPI sdist | `ascend_quant_toolkit-0.4.2.tar.gz`，78,146 B，sha256 `5ff91e5343a8a3faaf687cac690ab7b90d4afa96a209514a3224765899548781` |
| TestPyPI wheel | 41,003 B，sha256 `ac8db384383be44404e3ecf8ce6f866b0fa8fc8f745d9a5755cbac4dda291206` |
| TestPyPI sdist | 78,166 B，sha256 `cc993bbe43f390489010fe4da7755579bd887b9d346f29d8d425502c8dd3cb32` |
| 配方覆盖 | `qwen25-w8a8` |
| 发布记录 | `docs/releases/ascend-quant-toolkit-0.4.2-pypi-release-20261013.md` |

### 1.2 ascend-quant-toolkit 0.5.0（PyPI 已发布）

| 项 | 值 |
|---|---|
| 版本 | 0.5.0（`src/ascend_quant_toolkit/__init__.py`） |
| 本地分支 | `feat/toolkit-0.5.0-qwen35-w4a8` @ `b93c073a60a1e6b28dc29c68b56358b8c8c27e88` |
| PyPI wheel | `ascend_quant_toolkit-0.5.0-py3-none-any.whl`，48,923 B，sha256 `ad3b320f8cbcabd4527eba0b7180808be7190cc0b4b1bb18636232cea3f6657f`，2026-11-12T10:00:44Z |
| PyPI sdist | `ascend_quant_toolkit-0.5.0.tar.gz`，66,144 B，sha256 `ea065055cabfa3e87c89584068a1953311c547ee092fea9f255fa0df0a86381f`，2026-11-12T10:00:45Z |
| GitHub Release | **未找到**；0.5.0 目前只有 PyPI 分发，没有 GitHub Release |
| 本地 dist wheel | 48,217 B，sha256 `9b55d3d12c408ae0496fa5e69c33552823e6afdabe25b80d7e8c43eeb66d295b`；这是本地构建，**不是** PyPI 字节 |
| 配方覆盖 | `qwen25-w8a8`、`qwen35-35b-a3b-w4a8-gdnw8a8` |
| 新增制品记录 | `docs/artifacts/qwen35-w4a8-gdnw8a8-tp1-20261023.md` |
| 新增能力 | mixed-precision recipe 描述、`verify-artifact`、混合精度 artifact 拒绝单 scheme contract |

> 0.5.0 的 PyPI 字节已核验；但 PyPI wheel 的 sha256 与本地 `dist/` wheel 不同，
> 本索引不把 PyPI wheel 强行对应到 `b93c073`。`b93c073` 是 0.5.0 分支的源码基线，
> 制品/配方映射以 §2、§3 和 artifact record 为准。

## 2. 配方—checkpoint—版本对应

| Toolkit / 线 | 配方 | 配方 sha256 | Checkpoint | 制品类型 | 运行时/基准记录 |
|---|---|---|---|---|---|
| 0.4.2 | `qwen25-w8a8` | `9c9385daea4b81019ac210f73867b883b5f833a7dda9627b6e3e543bc099dbd0`（tag `toolkit-v0.4.2` 文件核验） | 未随 0.4.2 发布 Qwen3.5 checkpoint | — | 仅发布/安装验证；无本仓库内吞吐/PPL 记录 |
| 0.5.0 | `qwen25-w8a8` | `9c9385daea4b81019ac210f73867b883b5f833a7dda9627b6e3e543bc099dbd0` | 无新增 | — | 同一运行时适配 |
| 0.5.0 | `qwen35-35b-a3b-w4a8-gdnw8a8` | `0bcef0f56ddebea8f34841127e6ac5b369ffee20853e5a7d148c344387fd9384` | `/data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot` | 混合 W4A8/W8A8，6 分片，24,047,130,272 B（权重） | benchmark `reports/tech-outcomes-20261117/ascend-quant/` |
| gcy 线（非 toolkit 版本） | `qwen3_5_35b_a3b_w4a8_experts.yaml` | `3dd1102e250e2f2be98dc54dff9c5d58884cc0d307e3777f5d3943171c6fbbfb` | `/data/gcy/Qwen3.5-35B-A3B-W4A8` | 专家 int4 + GDN bf16，6 分片，25,099,795,178 B | #547 P0/P1/P2；benchmark `data/results/gcy-*` |
| 官方 W8A8（对照） | `qwen3_5_35b_a3b_w8a8.yaml` | `b18286e8213d64380238071ad87d7b1236f68dacb0813d1f995ec5928be907ba` | `/data/jxd/Qwen3.5-35B-A3B-W8A8` | W8A8，39,810,890,505 B | 18.85 GiB 对照；MMLU/PPL |

Checkpoint 配方身份（0.5.0 GDN 制品）：

- 基座：Qwen3.5-35B-A3B @ `712cf74392b05026a6db2bf213d343747d1f6d45`
- Producer config_id：`Qwen3_5_moe_w8a8_norot`
- Producer source sha256：`d666ef936d89bf41ea804d0b678e90ba86bbde74807bca1d3b9c4597b020d1db`
- 校准集：`calibImages`；Rotation：none

## 3. Checkpoint 指纹（0.5.0 GDN 制品）

| 项 | 值 |
|---|---|
| 权重分片 | 6 个 `quant_model_weights-00001..00006-of-00006.safetensors` |
| 分片合计 | 24,047,130,272 B |
| 报告口径制品大小 | **22.38 GiB**（含小文件；源字节 24,096,662,520 B，按 2^30 精确换算 22.44 GiB，源文档存在约 0.06 GiB 口径差） |
| `quant_model_description.json` | 124,851 tensor entries；`W4A8_DYNAMIC` 122,880、`W8A8_DYNAMIC` 360–361、`FLOAT` 1,611、`W8A16` 0 |
| 验证 | `ascend-quant-toolkit verify-artifact --artifact /data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot --recipe qwen35-35b-a3b-w4a8-gdnw8a8` → `valid: true`（2026-10-28） |
| 验证要点 | 声明/观测精度集合一致；W8A16=0；`linear_attn.in_proj_qkv`/`in_proj_z` 成对；FLOAT scope 无违规；6 分片字节一致 |
| 限制 | `prepare-runtime` / `export-contract` 按设计拒绝该混合精度制品；artifact contract 1.1.0 是单 scheme |

分片 sha256 见 `docs/artifacts/qwen35-w4a8-gdnw8a8-tp1-20261023.md`。

## 4. 已测指标（详细映射见 benchmark 报告）

| 指标 | 值 | 设备总量 | KV 预算 | 原始记录 |
|---|---|---:|---|---|
| BF16 → 混合量化权重 | 66.99 GiB → 22.38 GiB | 1×910B3（TP1 服务） | n/a（权重）；容量组见下 | artifact record；`du -sb` |
| 单卡部署 | 22.38 GiB 权重 + 33.49 GiB KV = 55.87 GiB，落在 0.95×60.96 GiB 预算内 | 1×910B3 | **33.49 GiB**（35,956,596,497 B） | `gen_w4a8gdn_tp1_curve.py` KV_NOTE |
| 正式同 KV 对照 C16 | W8A8 205.58 vs W4A8/GDN 203.39 output tok/s | 1×910B3 | **18.85 GiB**（20,242,376,704 B） | `data/results/perf/{w8a8,w4a8gdn}-18.85g/c16/summary.json` |
| 19.22 GiB 同 KV 附录 C1–C16 | W4A8/GDN 90.02/144.19/208.55/320.17/212.66；W8A8 89.63/151.14/201.33/307.03/206.13 | 1×910B3 | **19.22 GiB**（20,637,317,857 B）；两臂 820,136 tokens | `data/results/perf/{w8a8,w4a8gdn}-19.22g/` |
| 容量组 C1–C16 | 90.19 / 142.95 / 203.79 / 312.70 / 404.81 output tok/s | 1×910B3 | **33.49 GiB** | `data/results/perf/w4a8gdn-33.49g/` |
| gcy P0-1 同 KV C16 | W8A8 194.80 vs gcy W4A8 200.81 output tok/s | 1×910B3 | **18.85 GiB** | `swe_results/ab-*`（gcy evidence） |
| gcy 容量组 C16 | gcy W4A8 394.34 output tok/s | 1×910B3 | **32.56 GiB**（ab-w4a8max） | `swe_results/ab-w4a8max-c16-900s/` |
| PPL（双卡） | bf16 7.1694；W8A8 7.1750；experts int4 7.8551；GDN-W8A8 7.9070 | 2×910B3（TP2） | n/a（PPL，max-model-len 4096） | `data/results/ppl/` |
| PPL（单卡） | W8A8 7.1785；GDN-W8A8 7.4009 | 1×910B3（TP1） | n/a | `data/results/ppl/ppl_*_tp1_*` |
| MMLU-Pro 12,032 | BF16 79.862%；W8A8 79.496%；gcy W4A8 79.746%；GDN-W8A8 79.172% | BF16 TP2；量化 TP1 | n/a（32K context，gpu util 0.90） | `data/results/mmlu/` |
| HLE EP2 Gold / Revision | 145/668=21.71%；302/1143=26.42% | 2×910B3（TP2+EP2） | n/a | `data/results/hle/` |
| FrontierScience Olym / Research | 60/100=60.00%；1/60=1.67% | 2×910B3（TP2） | n/a | `data/results/frontierscience/` |
| 长上下文 8K/32K/128K | NIAH 单针两臂 100%；多针 W8A8 90/90、W4A8 100/90；长生成 32K 80% vs 100%、128K 60% vs 60% | 1×910B3（TP1） | **8 GiB**（8,589,934,592 B） | `data/results/longctx/` |
| 长上下文 260K | NIAH 30/30=100% vs 29/30=96.7%；长生成 80% vs 100% | 1×910B3（TP1） | **8 GiB** | `data/results/longctx-260k/` |
| BF16 锚点 PPL | 4K/32K/128K：W4A8 vs BF16 +2.09%/+2.50%/+2.83%；vs W8A8 +1.95%/+2.14%/+4.38% | BF16 TP2；量化 TP1 | **8 GiB** | `data/results/longctx/p2-2-bf16-anchor/` |
| 长文本性能 C1 | TTFT ~0.66/2.7/18/57 s；TPOT +0.6%~+2.2%；total tok/s −1% | 1×910B3（TP1） | 无显式 pin（C1） | `data/results/longperf/` |
| 满 262,144 capacity | 两臂 HTTP 200，prompt 262,143 + 1 token；~57.3 s | 1×910B3（TP1） | **8 GiB**；262,144 总上下文 | `data/results/longctx-262k/capacity_smoke_*.json` |
| 261K NIAH | W8A8 单针 24/25、多针 3/5；W4A8 单针 25/25、多针 5/5 | 1×910B3（TP1） | **8 GiB** | `data/results/longctx-262k/niah_*` |
| 260.8K 长生成 | W8A8 5/5=100%；W4A8 2/5=40%（3 条截断）；C3 触发，n=5 | 1×910B3（TP1） | **8 GiB** | `data/results/longctx-262k/longgen_*` |

完整数值、哈希和文件映射见 benchmark `reports/tech-outcomes-20261117/ascend-quant/README.md`
与 `evidence-map.csv`。

## 5. 版本配对与限制（必须如实引用）

1. **0.4.2 的 runtime counterpart**：0.4.2 release record 写“runtime counterpart
   (`vllm-ascend-quant-ext` 0.5.0) recorded separately”，但 0.5.0 toolkit README 把
   runtime extension 写成 **frozen at 0.4.1a4**，并注明 not part of the 0.5.0 release；
   `CHANGELOG.md` 又保留 `vllm-ascend-quant-ext 0.5.0 - 2026-10-12`。三者仍需
   jxd1111 确认唯一事实；本索引不合并 runtime 版本结论。
2. **0.5.0 发布渠道**：PyPI 0.5.0 字节已核验（§1.2）；**没有 GitHub Release**。
   PyPI wheel 与本地 `dist/` wheel 哈希不同，不能把 PyPI 字节强行归到 `b93c073`。
3. **0.4.2 配方哈希**：已在 tag `toolkit-v0.4.2` 上核验 `qwen25-w8a8.yaml` =
   `9c9385da...`；此前“tag 内未核验”的表述作废。
4. **0.4.2 runtime E2E 原件**：旧 repo 的
   `docs/evidence/w8a8-online-view-npu-e2e-20261012.md` 与
   `docs/evidence/w8a8-frozen-v1-npu-e2e-20260917.md` 不在当前 checkout；
   只能链旧 Release/固定提交，不复制成新证据，也不重跑。
5. **满 262,144 任务级**：capacity smoke 两臂均通过；261K NIAH C1/C2 未触发；
   260.8K 长生成 C3 触发（W4A8 2/5 vs W8A8 5/5，n=5，3 条截断；260K 方向相反）。
   不声称稳定退化，也不把“无退化”外推到 262,144 任务级。

## 6. 建议的对外引用层级

- **可直接引用（有原始记录/固定链接）**：
  - 0.4.2 PyPI 字节与哈希；0.5.0 PyPI 字节与哈希；0.4.2 Release id。
  - 0.4.2 tag 配方哈希；0.5.0 配方/checkpoint 分片哈希/verify-artifact 结果。
  - 18.85 GiB 正式对照、19.22 GiB 附录、33.49 GiB 容量组；PPL/MMLU/HLE/FS；
    8K/32K/128K + 260K 长上下文；满 262,144 capacity smoke。
- **必须带限定**：HLE `formal_acceptance=false`、自评分、单 trial；FS Research n=60；
  任务分数为 TP2；吞吐为 TP1；260.8K 长生成 C3 触发（n=5）。
- **不可引用/不可外推**：0.4.2 runtime E2E 的“新复制件”；跨 KV 的加速比较；
  把 260K 的“无退化”外推到完整 262,144 任务级。

## 7. 固定引用

| 内容 | 固定引用 |
|---|---|
| 0.4.2 Release | tag `toolkit-v0.4.2`；commit `50ba3f8fdad986fa647b0d55918d13b47a25ec42`；GitHub Release `411366927` |
| 0.4.2 PyPI | https://pypi.org/project/ascend-quant-toolkit/0.4.2/ |
| 0.5.0 PyPI | https://pypi.org/project/ascend-quant-toolkit/0.5.0/ |
| 0.5.0 toolkit 分支 | `b93c073a60a1e6b28dc29c68b56358b8c8c27e88` |
| artifact record | `docs/artifacts/qwen35-w4a8-gdnw8a8-tp1-20261023.md` |
| benchmark 报告 | `vLLM-HUST/vllm-hust-benchmark` `reports/tech-outcomes-20261117/ascend-quant/README.md`（PR 合并后填固定 commit） |
| benchmark Release | tag `tech-outcomes-20261117-ascend-quant`（上传后填附件 URL/SHA） |
| 本索引 | `vllm-ascend-quant-hust/docs/tech-outcomes-20261117/evidence-index-0.4.2-0.5.0.md` |
| 分工/适用范围 | `vllm-ascend-quant-hust/docs/tech-outcomes-20261117/contributions.md` |
