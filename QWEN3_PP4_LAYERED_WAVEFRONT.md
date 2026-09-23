# Qwen3-30B-A3B，PP=4：layered prefill 的 stage wavefront 与中继跳过

2026-09-23。PP=4、TP=1，在 layered prefill 上逐项打开两个改动，chunked 作为同负载对照：

- `max_concurrent_layered_prefills`：让多条 prefill 同时进入层流水，各自占住不同的 PP stage
- `skip_relay_stages`：当前层组里无事可做的 stage 不再跑一遍 runner 准备链路

48 层分 8 组，每组 6 层。每个 PP stage 12 层放 2 组，组不跨 stage，8 组的归属 stage 依次是 `[0,0,1,1,2,2,3,3]`。`fuse_mixed_batch` 在三行 layered 上都是打开的：mixed 步里 decode token 和 prefill token 进同一次 eager forward。

一组层只在一个 stage 上跑。改动前调度器同一时刻只让一条 prefill 进入层流水，于是 PP=4 下每个 step 只有一个 stage 在做 prefill，另外 3 个空转；而这个 step 仍然走满整个 ring，每个 stage 都要跑一遍准备链路和一次 0 层 forward，再把整段激活发给下一跳（32k prompt 下每跳 268 MB）。

四行采自同一趟，都是单次。

## PP=4，in=32768，out=512，G=8

`out=512，n=24，concurrency=8`，24/24。

| 配置 | TTFT avg (ms) | TTFT P90 (ms) | TPOT avg (ms) | ITL median (ms) | E2EL avg (ms) | E2EL P90 (ms) | 输出吞吐 (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| chunked MBT=4096 | 5614.0 | 11198.0 | 76.2 | 61.8 | 44576.7 | 51080.6 | 91.69 |
| layered G=8 | 16414.4 | 29568.8 | 106.0 | 61.7 | 70600.0 | 84957.8 | 57.97 |
| layered G=8，`max_concurrent_layered_prefills=4` | 12338.7 | 18099.8 | 74.2 | 61.9 | 50230.4 | 51487.7 | 81.52 |
| 同上 + `skip_relay_stages` | 10243.9 | 13826.2 | 70.1 | 61.8 | 46043.4 | 47639.2 | 88.93 |

## 相对未改动的 layered

表格第二行是未改动的 layered，也就是两个开关都不给时的现状。下面的百分比都以它为基准，负号表示更低（延迟更低、时间更短）。

| 指标 | 未改动 layered | + wavefront | 变化 | + wavefront + skip_relay | 变化 |
|---|---:|---:|---:|---:|---:|
| 输出吞吐 (tok/s) | 57.97 | 81.52 | +40.6% | 88.93 | +53.4% |
| 总耗时 (s) | 212.0 | 150.7 | -28.9% | 138.2 | -34.8% |
| TTFT avg (ms) | 16414.4 | 12338.7 | -24.8% | 10243.9 | -37.6% |
| TTFT P90 (ms) | 29568.8 | 18099.8 | -38.8% | 13826.2 | -53.2% |
| TPOT avg (ms) | 106.0 | 74.2 | -30.0% | 70.1 | -33.9% |
| ITL median (ms) | 61.7 | 61.9 | +0.3% | 61.8 | +0.2% |
| E2EL avg (ms) | 70600.0 | 50230.4 | -28.9% | 46043.4 | -34.8% |
| E2EL P90 (ms) | 84957.8 | 51487.7 | -39.4% | 47639.2 | -43.9% |
| E2EL max (ms) | 96070.9 | 51501.7 | -46.4% | 47652.8 | -50.4% |
| prefill 阶段 (s) | 117.1 | 55.6 | -52.5% | 43.2 | -63.1% |
| decode 阶段 (s) | 94.8 | 95.1 | +0.3% | 95.0 | +0.2% |
| 每组 step (ms) | 571 | 857 | +50.1% | 714 | +25.0% |

收益全部来自 prefill 阶段：`117.1 -> 43.2 s`。decode 阶段三行都在 94.8–95.1 s，ITL median 三行都在 61.7–61.9 ms —— 两个改动都没有碰 decode 路径。

尾部改善大于平均改善：E2EL P90 降 43.9%、max 降 50.4%，都高于 avg 的 34.8%。未改动的 layered 下 prompt 严格串行（日志里 24 条 prompt 对应 24 段连续的层组序列），排在后面的请求要等前面的整条走完 8 组，尾部因此被拉长；wavefront 让 prompt 交错（同样 24 条 prompt 变成 180 段），这部分排队消失。

两个开关的贡献可以分开看：wavefront 单独把吞吐从 57.97 拉到 81.52（+40.6%），在此之上 `skip_relay_stages` 再拉到 88.93（+9.1%）。

## 相对 chunked

以 chunked 为基准。负号表示 layered 更低（延迟更低、时间更短），正号表示更高。吞吐一行的负号是更慢。

| 指标 | chunked | layered（现状） | 变化 | + wavefront | 变化 | + wavefront + skip_relay | 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 输出吞吐 (tok/s) | 91.69 | 57.97 | -36.8% | 81.52 | -11.1% | 88.93 | -3.0% |
| 总耗时 (s) | 134.0 | 212.0 | +58.2% | 150.7 | +12.5% | 138.2 | +3.1% |
| TTFT avg (ms) | 5614.0 | 16414.4 | +192.4% | 12338.7 | +119.8% | 10243.9 | +82.5% |
| TTFT P90 (ms) | 11198.0 | 29568.8 | +164.1% | 18099.8 | +61.6% | 13826.2 | +23.5% |
| TPOT avg (ms) | 76.2 | 106.0 | +39.1% | 74.2 | -2.6% | 70.1 | -8.0% |
| ITL median (ms) | 61.8 | 61.7 | -0.2% | 61.9 | +0.2% | 61.8 | 0.0% |
| E2EL avg (ms) | 44576.7 | 70600.0 | +58.4% | 50230.4 | +12.7% | 46043.4 | +3.3% |
| E2EL P90 (ms) | 51080.6 | 84957.8 | +66.3% | 51487.7 | +0.8% | 47639.2 | -6.7% |
| E2EL max (ms) | 54753.4 | 96070.9 | +75.5% | 51501.7 | -5.9% | 47652.8 | -13.0% |
| prefill 阶段 (s) | 38.9 | 117.1 | +201.0% | 55.6 | +42.9% | 43.2 | +11.1% |
| decode 阶段 (s) | 95.1 | 94.8 | -0.3% | 95.1 | 0.0% | 95.0 | -0.1% |

两个改动把 layered 从"全面差于 chunked"拉到"互有胜负"：

- **仍然差的**：TTFT avg 高 82.5%。一条 prefill 要走完 8 个层组才出第一个 token，这是层切分的固有代价，本文两个改动都不触及。prefill 阶段也还高 11.1%。
- **已经追平的**：吞吐差 3.0%，E2EL avg 高 3.3%，总耗时高 3.1%，都在单次测量能分辨的边缘。
- **反超的**：TPOT avg 低 8.0%，E2EL P90 低 6.7%，E2EL max 低 13.0%。

decode 阶段和 ITL median 四行全部一致（94.8–95.1 s、61.7–61.9 ms），两种 prefill 策略对 decode 没有影响，上表所有差异都发生在 prefill 阶段。

E2EL P90 和 max 的反超不是"更快"，而是分布更紧：

| 配置 | E2EL min (ms) | median (ms) | max (ms) | max - min |
|---|---:|---:|---:|---:|
| chunked | 35429.3 | 43791.0 | 54753.4 | 19324.1 |
| + wavefront + skip_relay | 45186.3 | 45305.9 | 47652.8 | 2466.5 |

chunked 让一部分请求很早完成（最快 35.4 s），代价是另一部分拖到 54.8 s；wavefront 下 24 条请求几乎同时完成，跨度只有 2.5 s。所以 max 更低、min 更高，P90 和 max 的优势来自这个均匀性，不是单条请求变快了。

ITL max 则明显差于 chunked（5756.4 ms vs 937.3 ms）：`fuse_mixed_batch` 下一条 decode 搭上某条 prefill 后要跟着走完整段才采样，这一整段时间里它不出 token。`fuse_max_riders` 和 `release_idle_decodes` 两个开关就是为这个问题准备的，本文没有打开。

## 每组 step 为什么先变慢再变快

只开 wavefront 时每组 step 从 571 ms 涨到 857 ms：4 条 prefill 挤在一条 ring 上，每个 step 都要等三个无事可做的 stage 各跑一遍准备链路和一次整段激活传递。但同时有 prefill 的 stage 从 1 个变成多个，总体收益盖过了单步变慢，所以吞吐仍从 57.97 涨到 81.52。

`skip_relay_stages` 去掉那三个 stage 的空转后，每组 step 回到 714 ms（比 857 ms 少 16.7%），prefill 阶段再少 12.4 s。

## 配置

- 机型：8× Ascend 910B3。PP=4 用卡 0–3
- 模型：Qwen3-30B-A3B，48 层。V2，`VLLM_USE_V2_MODEL_RUNNER=1`
- 图模式：`FULL_DECODE_ONLY`。均匀 decode 进图，prefill 和 mixed 步保持 eager
- prefix cache 关，async scheduling 关（Phase-1 layered 不支持）
- `--max-num-seqs 8`，`--gpu-memory-utilization 0.85`，`--max-model-len 33792`
- chunked：`max_num_batched_tokens=4096`，32768/4096 = 8 个 token 分片，和 layered 的 8 组对称
- layered：`allowed_num_groups=[8]`，`group_token_target=512`，关掉 chunked prefill
- 三行 layered 都打开 `fuse_mixed_batch`。`fuse_max_riders` 和 `release_idle_decodes` 三行都是默认关
- 表内数据用 aisbench 采集，`synth normal`，`temperature=0`、`ignore_eos`

## 两个改动分别做什么

### stage wavefront

`max_concurrent_layered_prefills` 默认 1，取值会被 `pipeline_parallel_size` 截断。大于 1 时，调度器优先挑一条"下一组落在空闲 stage 上"的 prefill；没有这样的候选，才考虑放新的 prefill 进来，而且要等 stage 0 排空 —— 新 prefill 总是从第 0 组、也就是 stage 0 进入，同时放进来的几条会一直锁在同一个 stage 上齐步走，反而没有收益。

`fuse_mixed_batch` 下，从第 0 组搭上某条 prefill 的 decode 要跟着这条 prefill 走完整段，只在最后一组采样。两段重叠的 span 不能招募同一个 decode：它的 token 进度在采样前是冻结的，两段都满足不了，于是永远不离开 `running`，引擎随后卡死。depth=1 时 span 不会重叠，所以这个判断只在 wavefront 打开时生效。

### 跳过无事可做的中继 stage

两部分，都只作用在 fuse 路径和纯 P 路径上：

- 组 owner 之前的 stage 发出的 fuse 激活切成 0 行。这份数据没人读：owner 会恢复自己存下的 frontier，更早的 stage 各自重建 transport placeholder。空张量让收发两端都跳过实际传输，但元数据交换保持配对。这一项没有开关，始终生效。
- `skip_relay_stages` 默认关。打开后，既不拥有当前组、也不拥有下一组的 stage 直接跳过准备链路和那次 0 层 forward：在 owner 之前就发 0 行占位，在 owner 之后就把收到的张量原样转发，`sample_tokens` 从预存的空输出应答。采样步、最后一组、`dp_size > 1`、挂了 KV connector 这四种情况保留完整路径。`relay_empty_steps` 和 `relay_forward_steps` 两个计数器用来区分"跳过了但没有收益"和"根本没触发"。

## 正确性

PP=4、TP=1、卡 0–3，`max_model_len=2048`、`max_tokens=64`、`max_num_seqs=8` 的烟测，四个 profile（chunked、layered depth=1、depth=4、depth=4 + `skip_relay_stages`）逐项通过：

- `pd_mix_wavefront_interleaved`：depth=4 的两个 profile 都为真，wavefront 确实形成
- `relay_empty_steps=15`、`relay_forward_steps=15`：两条中继路径都实际触发过
- depth=4 开关 `skip_relay_stages` 前后，输出逐 token 相同

depth=1 与 depth=4 的输出不完全相同。depth 会改变 rider 的分组方式，MoE 和 attention 里的浮点归约顺序跟着变，所以烟测把这一项按"只报告、不判失败"处理。