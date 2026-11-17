# 0.4.2 / 0.5.0 证据索引（合并稿）

状态：**提交稿（2026-11-17，已随 PR 提交）**
整理日期：2026-11-13；更新：2026-11-17（补 0.5.0 PyPI 发布字节）
用途：把 `ascend-quant-toolkit` 0.4.2 与 0.5.0 两个版本的发布证据、配方、checkpoint
和 benchmark 记录串到同一张索引中。只收录可追溯到原始记录的内容；缺失项在文末列出。

## 1. 版本与发布对象

### 1.1 ascend-quant-toolkit 0.4.2（已发布）

| 项 | 值 |
|---|---|
| 版本/日期 | 0.4.2 / 2026-10-13 |
| 源码修订 | `50ba3f8`（`main`）；peeled commit `50ba3f8fdad986fa647b0d55918d13b47a25ec42` |
| Tag | `toolkit-v0.4.2`，annotated tag object `be9393676a0808733302559f6e513672c08dfa1e` |
| GitHub Release | id `411366927`，2026-10-13T06:43:03Z；`https://github.com/jxd1111/Ascend-LLM-quant/releases/tag/toolkit-v0.4.2` |
| PyPI wheel | `ascend_quant_toolkit-0.4.2-py3-none-any.whl`，41,003 B，sha256 `57964380f85275e5e7d32d14bc6df94c020d6be70bfe47da0dcc430d946d2470` |
| PyPI sdist | `ascend_quant_toolkit-0.4.2.tar.gz`，78,146 B，sha256 `5ff91e5343a8a3faaf687cac690ab7b90d4afa96a209514a3224765899548781` |
| TestPyPI wheel | 41,003 B，sha256 `ac8db384383be44404e3ecf8ce6f866b0fa8fc8f745d9a5755cbac4dda291206` |
| TestPyPI sdist | 78,166 B，sha256 `cc993bbe43f390489010fe4da7755579bd887b9d346f29d8d425502c8dd3cb32` |
| 配方覆盖 | 发布记录中 `list-recipes` 只打印 `qwen25-w8a8` |
| 发布记录 | `docs/releases/ascend-quant-toolkit-0.4.2-pypi-release-20261013.md` |

### 1.2 ascend-quant-toolkit 0.5.0（已发布）

| 项 | 值 |
|---|---|
| 版本/日期 | 0.5.0 / 2026-10-28（`CHANGELOG.md`）；PyPI 上传 2026-11-12T10:00:44Z |
| `__version__` | `0.5.0`（`src/ascend_quant_toolkit/__init__.py`） |
| 发布 tag | `toolkit-v0.5.0` → commit `2eb0c5f18785b2eee25b525ad0f957f8a7ad3768`（当前 `origin/main`） |
| 本地分支 | `feat/toolkit-0.5.0-qwen35-w4a8` @ `b93c073a60a1e6b28dc29c68b56358b8c8c27e88`（发布前分支） |
| PyPI wheel | `ascend_quant_toolkit-0.5.0-py3-none-any.whl`，48,923 B，sha256 `ad3b320f8cbcabd4527eba0b7180808be7190cc0b4b1bb18636232cea3f6657f`，2026-11-12T10:00:44.436902Z |
| PyPI sdist | `ascend_quant_toolkit-0.5.0.tar.gz`，66,144 B，sha256 `ea065055cabfa3e87c89584068a1953311c547ee092fea9f255fa0df0a86381f`，2026-11-12T10:00:45.683173Z |
| PyPI 项目页 | `https://pypi.org/project/ascend-quant-toolkit/0.5.0/` |
| 本地 dist wheel | `dist/ascend_quant_toolkit-0.5.0-py3-none-any.whl`，48,217 B，sha256 `9b55d3d12c408ae0496fa5e69c33552823e6afdabe25b80d7e8c43eeb66d295b`；**与 PyPI 字节不同，不作为发布字节** |
| 配方覆盖 | `qwen25-w8a8`、`qwen35-35b-a3b-w4a8-gdnw8a8` |
| 新增制品记录 | `docs/artifacts/qwen35-w4a8-gdnw8a8-tp1-20261023.md` |
| 新增能力 | mixed-precision recipe 描述、`verify-artifact`、混合精度 artifact 拒绝单 scheme contract |

## 2. 配方—checkpoint—版本对应

| Toolkit | 配方 | 配方 sha256 | Checkpoint | 制品类型 | 运行时/基准记录 |
|---|---|---|---|---|---|
| 0.4.2 | `qwen25-w8a8` | 本地当前 YAML `9c9385da...`（0.4.2 tag 内未单独核验） | 未随 0.4.2 发布 Qwen3.5 checkpoint | — | 仅发布/安装验证；无本仓库内吞吐/PPL 记录 |
| 0.5.0 | `qwen25-w8a8` | `9c9385daea4b81019ac210f73867b883b5f833a7dda9627b6e3e543bc099dbd0` | 无新增 | — | 同一运行时适配 |
| 0.5.0 | `qwen35-35b-a3b-w4a8-gdnw8a8` | `0bcef0f56ddebea8f34841127e6ac5b369ffee20853e5a7d148c344387fd9384` | `/data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot` | 混合 W4A8/W8A8，6 分片，24,047,130,272 B（权重） | `vllm-hust-benchmark/reports/qwen35-w4a8-gdnw8a8-tp1-20261023/` |

Checkpoint 配方身份：

- 基座：Qwen3.5-35B-A3B @ `712cf74392b05026a6db2bf213d343747d1f6d45`
- Producer config_id：`Qwen3_5_moe_w8a8_norot`（保留自源 YAML）
- Producer source sha256：`d666ef936d89bf41ea804d0b678e90ba86bbde74807bca1d3b9c4597b020d1db`
- 校准集：`calibImages`
- Rotation：none

## 3. Checkpoint 指纹（0.5.0 配方产物）

| 项 | 值 |
|---|---|
| 权重分片 | 6 个 `quant_model_weights-00001..00006-of-00006.safetensors` |
| 分片合计 | 24,047,130,272 B |
| 报告口径制品大小 | **22.38 GiB**（含小文件；源字节 24,096,662,520 B，按 2^30 精确换算 22.44 GiB，源文档存在约 0.06 GiB 口径差） |
| `quant_model_description.json` | 124,851 tensor entries；`W4A8_DYNAMIC` 122,880、`W8A8_DYNAMIC` 360–361、`FLOAT` 1,611、`W8A16` 0 |
| 验证 | `ascend-quant-toolkit verify-artifact --artifact /data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot --recipe qwen35-35b-a3b-w4a8-gdnw8a8` → `valid: true`（2026-10-28） |
| 验证要点 | 声明/观测精度集合一致；W8A16=0；`linear_attn.in_proj_qkv`/`in_proj_z` 成对；FLOAT scope 无违规；6 分片字节一致 |
| 限制 | `prepare-runtime` / `export-contract` 按设计拒绝该混合精度制品；artifact contract 1.1.0 是单 scheme |

分片 sha256 见 `docs/artifacts/qwen35-w4a8-gdnw8a8-tp1-20261023.md` 第 28–35 行（本索引不重复展开）。

## 4. 已测指标（详细映射见 benchmark 报告）

| 指标 | 值 | 设备总量 | 原始记录 |
|---|---|---:|---|
| BF16 → 混合量化权重 | 66.99 GiB → 22.38 GiB | 1×910B3（TP1 服务） | artifact record；`du -sb`；curve generator KV_NOTE |
| 单卡部署 | 22.38 GiB 权重 + 33.49 GiB KV = 55.87 GiB，落在 0.95×60.96 GiB 预算内 | 1×910B3 | `gen_w4a8gdn_tp1_curve.py` KV_NOTE；report 启动命令 |
| 吞吐 C1–C16（容量组，KV 33.49 GiB） | 90.19 / 142.95 / 203.79 / 312.70 / 404.81 output tok/s | 1×910B3 | `swe-prefix-reuse/results/w4a8ours-tp1-pinmax-c*-900s/summary.json` |
| 同 KV（18.85 GiB）C16 对照 | W8A8 205.58 vs W4A8 203.39 output tok/s | 1×910B3 | `results/tp1-c16-900s/` 与 `results/w4a8ours-tp1-c16-900s/`（config 均为 `20242376704` B） |
| PPL（双卡） | bf16 7.1694；W8A8 7.1750；experts int4 7.8551；本制品 7.9070 | 2×910B3（TP2） | `logs/ppl_{bf16_20261022,w8a8,w4a8gcy,w8a8nr}.json` |
| PPL（单卡） | W8A8 7.1785；本制品 7.4009（复现 7.4005/7.4030） | 1×910B3（TP1） | `logs/ppl_w8a8_tp1_20261113T101838Z.json`（sha256 `bf4a10d0...`）、`logs/ppl_w8a8nr_tp1_20261113T101838Z.json`（sha256 `2c1f8e47...`）；历史 TP1 见 `ppl_dense_base_*` |
| HLE EP2 Gold | 145/668 = 21.71% | 2×910B3（TP2+EP2） | `hle-verified-public-20261106-ep2/report.json` |
| HLE EP2 Revision | 302/1143 = 26.42% | 2×910B3（TP2+EP2） | 同上 |
| FrontierScience Olympiad | 60/100 = 60.00% | 2×910B3（TP2） | `runs/fs-w8a8nr-tp2-20261106/results/frontierscience.csv` |
| FrontierScience Research | 1/60 = 1.67%，mean score 2.366 | 2×910B3（TP2） | 同上 |
| 长上下文 8K/32K/128K | NIAH 单针两臂 100%；多针 W8A8 90/90、W4A8 100/90；长生成 32K 80% vs 100%，128K 60% vs 60% | 1×910B3（TP1） | `long-context/run_20261113T111012Z/out/{niah,longgen}_{w8a8,w4a8}.jsonl`；`stability.json` |
| 长上下文 260K 扩展 | NIAH 30/30=100% vs 29/30=96.7%；长生成 80% vs 100% | 1×910B3（TP1） | `long-context-256k/run_20261116T042838Z/out/{niah,longgen}_{w8a8,w4a8}.jsonl`；`stability.json` |
| BF16 锚点长上下文 PPL | 4K/32K/128K：W4A8 vs BF16 +2.09%/+2.50%/+2.83%；vs W8A8 +1.95%/+2.14%/+4.38% | BF16 2×910B3 TP2；量化 1×910B3 TP1 | `p2-2-bf16-anchor/run_20261116T071251Z/out/longctx_ppl.csv` / `longctx.json` |
| 长文本性能 C1 8K/32K/128K/260K | TTFT ~0.66/2.7/18/57 s；W4A8 vs W8A8 TPOT +0.6%~+2.2%，total tok/s −1% | 1×910B3（TP1） | `long-context-perf/run_20261116T075647Z/out/longperf_summary.csv` |
| 长文本并发 8K/32K C1–C8、128K C1–C4、260K C1–C2 | W4A8 total tok/s −0.4%~−9.2%；32K C8 最差 | 1×910B3（TP1） | `long-context-perf-conc/run_20261116T113623Z/out/longperf_conc_summary.csv` |
| 完整 262,144 上下文 | **未测** | — | 260K 扩展未到满上下文，不外推 |

## 5. 版本配对与冲突（必须由负责人确认）

1. **0.4.2 的 runtime counterpart**：0.4.2 release record 写“runtime counterpart
   (`vllm-ascend-quant-ext` 0.5.0) recorded separately”，但当前 0.5.0 toolkit README
   把 runtime extension 写成 **frozen at 0.4.1a4**，并注明 not part of the 0.5.0 release。
   `CHANGELOG.md` 又保留 `vllm-ascend-quant-ext 0.5.0 - 2026-10-12` 条目。
   三者需要确认唯一事实，索引中暂不合并 runtime 版本结论。
2. **0.5.0 的发布字节**：PyPI 0.5.0 已于 2026-11-12 发布，wheel 48,923 B /
   sha256 `ad3b320f...`，sdist 66,144 B / sha256 `ea065055...`；tag
   `toolkit-v0.5.0` 指向 `2eb0c5f...`。本地 `dist/` wheel 是 48,217 B /
   `9b55d3d1...`，与 PyPI 字节不同；索引以 PyPI 字节为准，本地 wheel 只标为本地构建件。
3. **0.4.2 的配方 YAML 哈希**：0.4.2 发布记录没有给出 `qwen25-w8a8.yaml` 的
   tag 内哈希；本索引中的 `9c9385da...` 是当前 checkout 的哈希，不能冒充 0.4.2
   tag 的字节。若需要，应在 `50ba3f8` 上重新计算。
4. **0.4.2 运行期 E2E 原件**：旧 repo 的
   `docs/evidence/w8a8-online-view-npu-e2e-20261012.md` 与
   `docs/evidence/w8a8-frozen-v1-npu-e2e-20260917.md` 不在当前 checkout；
   只能链接旧 Release/固定提交，不能复制成新证据。

## 6. 建议的对外引用层级

- **可引用（有原始记录）**：0.4.2 PyPI 字节与哈希；0.5.0 PyPI wheel/sdist 字节与哈希；
  0.5.0 配方 sha256、checkpoint 分片哈希、verify-artifact 结果；吞吐、PPL、
  HLE/FS 的具体数值（带设备总量与限定）。
- **不可引用（缺原件）**：0.4.2 runtime E2E 的“新复制件”；
  完整 262,144 上下文（未测；260K 扩展已测）；19.22 GiB 同 KV 组（未测）；
  33.49 GiB 容量组与 18.85 GiB W8A8 之间的跨 KV 加速比较。
- **已可用的同 KV 对照**：18.85 GiB pin 下，W8A8 C16 205.58 与 W4A8/GDN C16 203.39
  （1 卡、同负载、同协议；差 1.1%，不构成加速结论）。
- **必须带限定**：HLE `formal_acceptance=false`、自评分、单 trial；FS Research n=60；
  任务分数为 TP2 两卡；吞吐曲线为 TP1 单卡。

## 7. 固定引用

| 内容 | 固定引用 |
|---|---|
| 0.4.2 release record | tag `toolkit-v0.4.2`；commit `50ba3f8fdad986fa647b0d55918d13b47a25ec42`；GitHub Release `411366927` |
| 0.5.0 toolkit 分支（发布前） | `b93c073a60a1e6b28dc29c68b56358b8c8c27e88` |
| 0.5.0 发布 tag | `toolkit-v0.5.0` → `2eb0c5f18785b2eee25b525ad0f957f8a7ad3768`；`https://github.com/vLLM-HUST/vllm-ascend-quant-hust/tree/toolkit-v0.5.0` |
| 0.5.0 PyPI 发布 | `https://pypi.org/project/ascend-quant-toolkit/0.5.0/`；wheel 48,923 B / sha256 `ad3b320f8cbcabd4527eba0b7180808be7190cc0b4b1bb18636232cea3f6657f`；sdist 66,144 B / sha256 `ea065055cabfa3e87c89584068a1953311c547ee092fea9f255fa0df0a86381f`；上传 2026-11-12T10:00:44Z/45Z |
| 0.5.0 artifact record | `docs/artifacts/qwen35-w4a8-gdnw8a8-tp1-20261023.md` |
| benchmark report | `reports/qwen35-w4a8-gdnw8a8-tp1-20261023/README.md`；benchmark 分支 `83956a6749dcea75e69e170600075092a6442d6b`；`main` `51563c0e6b65f5563475b539dad54f236f7e0470` |
| HLE EP2 package | `hle-verified-public-20261106-ep2/SHA256SUMS`；report sha256 `44027b80df828c906c745056440f0a1d4d34457e4ca04cfa9182aa319c903895` |
| FrontierScience | run `fs-w8a8nr-tp2-20261106`；evidence bundle `0144d0fa271b0bf52e840abd20d43ab6a8d491627feaa55e52238ae8d14ee8f9` |
