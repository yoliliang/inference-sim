# 仿真代码自检（2026-09-26，提交 6048d9d7）

范围：fork 相对上游 2ebef6a 的全部改动（共享队列、pooled 批次形成、SRF 逐出、Mooncake 准入、精简统计、sweep/analyze/objective/compare 工具链）。方法：逐文件读代码，关键结论逐条复核；未跑新实验。

严重程度：BUG = 数字错或可能卡住请求；假设 = 不是错，但是用户未被告知的建模选择；小 = 健壮性。

## A. 确认的 bug

### A1. BUG：Mooncake 的解码负荷项失效，"now" 与 "predict" 是同一个策略
- `sim/admission_mooncake.go` 的 `decodeBatch` 造出的合成解码请求只设了 `ProgressIndex` 和 `NumNewTokens`，从未给 `OutputTokens`。真实步时长模型（trained-physics，`sim/latency/trained_physics_model.go`，以及 roofline，`sim/latency/latency.go`）判定一条请求是 decode 的条件是 `len(OutputTokens) > 0`；`ProgressIndex >= InputLen` 且没有输出 token 的请求两个分支都不进，贡献为零。
- 后果：预测 TBT（每步产出一个 token 的间隔）等于一个没有解码工作的批次的步时长，与批大小 k 和上下文长度无关，永远小于阈值 theta；两个 profile（`B1_mooncake`、`B1_mooncake_now`）只在 k 的取值上不同，而 k 不影响结果，所以两者决策相同。之前报告的"now 与 predict 字节相同、解码负荷从不触发"其实是这个 bug 的表现，不是参数的性质。
- 单元测试没抓到，因为测试用的是线性替身步时长函数，它只看 `ProgressIndex`。
- 已有的 Mooncake 结果（`2026-09-24_compare_mooncake_load`、`2026-09-24_compare_mooncake_scale_load1.2`）因此只是"按 prefill 负荷预测的早拒绝"，不是论文的完整规则。
- 修法：合成解码请求加 `OutputTokens = backing(1)`；补一个用真实 roofline 模型的测试，断言 k = 300 的预测 TBT 大于 k = 1 的。修后需重跑 Mooncake 的两组对比。

### A2. BUG（潜在，当前工作负载触不到）：pooled 批次形成先分配 KV 再取件，超大 prompt 会让整个集群空转
- `sim/batch_formation_pooled.go` 第 133 行 `AllocateKVBlocks` 在 `SharedQueue.Take` 之前执行，而"prompt 是否放得下"的可服务性检查在 `Take` 里（`AdoptSharedRequest`）。若队头 prompt 大于一台空实例的全部空闲块，分配失败、head-of-line 停止、批次为空、`emptyStepAt = now`，不再重排步；每台实例都一样，于是队列非空但所有实例空闲，之后每次到达或完成只唤醒一台做一次空步。push 模式下同样的请求在 `EnqueueRequest` 就被丢弃，不会挡住别人。
- 当前 `types3.yaml` 最大 prompt 2,048 token，最小缓存 2,500 块 = 40,000 token，触不到。换工作负载或把 `--blocks` 调小就会触发。
- 副作用：`Take` 拒绝时块已经写入前缀缓存的哈希表再释放，缓存会"宣称"有从未算过的内容。
- 修法：把可服务性检查提前到 `SharedQueueArrivalEvent` 入队时，不可服务的请求当场记为 dropped。

### A3. 小：共享队列模式会执行空步
- `Step` 的提前返回条件加了 `!sharedWork()`，一台在时刻 t 被唤醒、而同一时刻兄弟实例已把队列取空的实例，会对空批次执行 `scheduleBatch` 和 `executeBatchStep`：步数加一、记一次零占用的 KV 样本、时钟前进。push 模式下从不执行空步。影响每实例步数和 stdout 的 KV 平均，不影响窗口块。

### A4. 小：视界时刻仍在到达管线里的请求不计入任何桶
- 共享模式在 `Take` 时才建指标行，视界行只覆盖队列里的请求；到达事件（到达 + 准入延迟 + alpha 延迟）还未触发的请求没有行。n = 1 验证时看到的 2 行差异就是它。另外 `injected_requests` 是各桶之和（`sim/metrics.go`），所以 INV-1 是恒等式，不是对到达计数的检验。

### A5. 小：`SharedQueue.Candidates()` 返回内部切片，`Remove` 原地平移
- 现在的 `FCFSPool.Order` 复制了一份所以安全；将来任何直接返回 `shared` 切片的集合策略会在每次 `Take` 后跳过一个请求。应让 `Candidates` 返回副本或在接口注释里写明。

### A6. 确定性与浮点顺序：未发现问题
- map 遍历只出现在顺序无关处（按类型独立、事后排序、整数和）。精简统计的 `sum_sojourn_s` 按完成顺序累加，与按 id 顺序的路径只在末位有差。

## B. 被当作"物理"写进仿真器的策略性约定

1. **共享队列模式是 sweep.py 的默认**（`BASE_FLAGS`），push 基线用 `--shared-queue-push=false` 退出。默认集合策略 `fcfs-pool` 是我的占位选择，用户未定义。
2. **pooled 批次形成**：本机被逐出的请求排在共享队列里更早到达的请求之前（相对全局 FCFS 是一次优先级倒置）；发生逐出的那一步不接纳新请求；队头放不下就停止，即使后面更小的请求放得下；分块 = min(剩余 prompt, 预算)；取件零成本；alpha 延迟（API 处理延迟）在进共享队列之前用 0 号实例的模型计一次；唤醒最低编号的空闲实例（轻载时把负荷和前缀缓存集中在低编号实例上）；忙实例只在步边界看队列；`--scheduler` 只给本机队列排序，共享模式下只影响被逐出的请求。
3. **SRF**：用 `ProgressIndex` 当"持有的 KV / 重算成本"，它包含前缀缓存命中的 token，且忽略受害者重 prefill 时会命中自己残留块（约 4.8 percent 命中率）；平局取尾部；受害者若在批次中间，Phase 1 断开，之后的批次成员这一步不被调度（vLLM 亦如此，fcfs 下受害者总在尾部所以无人被跳过，SRF 下常不在尾部）。
4. **Mooncake**：除 A1 外，池化映射（最佳实例 = 最小预测 TTFT，解码负荷 = 实例均值）、chunk 下限 budget/4、空实例上下文 = prompt + 64、内存受限 TTFT 项的释放率 KvTokensInUse / t_d、t_d = 9 s、目标 2 s / 60 ms、theta = 1 都是 fork 的选择；输入是 50 ms 陈旧快照；若与共享队列同用，它看不到共享队列（快照的队列深度是每实例的）。

## C. 统计口径

1. **延迟分位数右删失**：窗口内 TTFT、E2E、等待的分位数只用已完成的窗口到达（`window_stats.go`），未完成者即使 TTFT 已知也不计。过载时被删失的正是最慢的请求，报告的 p99 实际是完成子集的 p99，约等于真实群体的 p(99(1 - f))，f 为未完成比例（1.2 R_cap 时约 1.7 percent）。可改为把删失值视为无穷大的群体分位数（f < 1 - p 时有效）。
2. **目标函数依赖视界并"奖励"积压**：未完成请求付 h (T - a)、不得收益，即使已经产出部分 token。过载下这是下界，其松紧取决于尾长 D 和策略留在视界的积压量：留下更多未完成工作的策略被少罚。过载比较应附尾长敏感性（D = 100 与 300 s）或期望剩余成本项。
3. **`output_tok_per_s`、`preempt_per_s` 是群体量不是时间平均**：完成的窗口到达的输出 token 除以窗口长度，不是窗口期间产出的 token；过载时随未完成比例下降而 GPU 仍饱和。`preempt_per_arrival`、`wasted_tok_per_arrival` 的分母 `arrived` 含被拒请求，对准入策略偏小（Mooncake 拒 23 percent 时明显）。
4. **TPOT 定义**：`compare.py` 用 (E2E 均值 - TTFT 均值) / (输出均值 - 1)，是总量之比，即按 token 加权、由 type3 主导，不是 Kim 等人的每请求 TPOT 均值；分子含 `PostDecodeFixedOverhead`。BLIS 自带的 `itl_mean` 是每 token 的替代。
5. **相对变化**：`_rel` 用全部基线种子的均值绝对值做分母，成对区间也除以这个固定分母，忽略了基线自身的变异；作展示可以，不是比值的置信区间。
6. **KV 利用率**：按采样时刻（非到达群体）在 [warm, end) 内平均，每 (实例, 样本) 等权；利用率 = 运行中请求引用的块 / 总块，前缀缓存里未被引用的块算空闲（vLLM 约定）。所以论文面板里的"memory usage"是在用比例，不是驻留内容。
7. 窗口成员资格、t 区间、成对差分：正确。stdout 聚合 TTFT 含在途请求，窗口块不含，两种定义并存。
8. 小：精简模式下 `WindowStats` 不幂等（会把未完成行加进 `Lean.win`），目前只调用一次。

## D. push 与共享队列两种模式的记账差异

- 行创建：push 在路由时，共享在 `Take` 时（或视界时为队列驻留者补行）。在共享队列里等待的请求在被取走前没有行、没有 `HandledBy`。`still_queued` = 共享队列驻留 + 各实例本机队列（本机队列只含被逐出者）。
- 状态 csv：`queue_depth` 是本机队列（共享模式下只有被逐出者），积压在 `cluster_shared_queue` 列且每实例行重复；`analyze.py` 画的是 `queue_depth`，共享模式下面板显示空队列。
- `TotalInputTokens`：push 在入队时计，共享在 `Take` 时计加视界驻留者；两者都不计管线中的请求。
- **等待与 TTFT 在被逐出请求重新接纳时被覆盖**（上游 #1122 约定，`TTFTSet` 重置；ITL 也重置）：E2E 仍正确，但"等待"是到最后一次接纳的时间，TTFT 是最后一次首 token 的时间。`analyze.py` 的面板标题"queue wait before first scheduling"对被逐出者不成立。SRF 改变了谁被逐出，所以 SRF 与 fcfs 的 TTFT / 等待差异有一部分来自这个约定。
- `wasted_tokens` = 逐出时的 `ProgressIndex`：含从未计算的前缀缓存 token，也忽略受害者残留块使重 prefill 更便宜；是重算量的上界。

## E. 性能隐患（n <= 64 时都很小）
pooled 模式每步每实例分配 O(Q) 的候选切片和一个本机集合（Q 为积压，1.2 R_cap 时数百）；`SharedQueue.Remove` 每次取件 O(Q) 平移；`wakeIdleInstance` 每次到达和完成 O(n)。若将来某个集合策略返回 stop = false（跳过而非停止），候选扫描会变成每步 O(Q) 次哈希计算。

## F. 建议的处理顺序
1. A1（改一行加一个测试，然后重跑两组 Mooncake 对比；现在 now 与 predict 的对比是空的）。
2. C1、C2（在任何过载结论之前，报告考虑删失的分位数和尾长敏感性）。
3. A2、A3（便宜，堵住潜在漏洞）。
4. D 中的标签修正（面板标题、状态 csv 画 `cluster_shared_queue`）。
