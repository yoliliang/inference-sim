# Scaling definition for the large-system experiments

Reference: Balseiro, Ma and Zhang (2025), "Dynamic pricing for reusable resources: the
power of two prices", Operations Research (arXiv 2308.13822v3, copy in this folder). Their
sections 3.2 (performance metric), 3.3 (asymptotic regime) and 6 (multiple classes and
resources) are the ones this note maps onto our simulation.

House style: no em dashes or en dashes.

## 1. Their regime, in our words

- One reusable resource with c identical units. A customer who is admitted holds one
  unit for a random usage duration with mean d, then returns it. No queue: a customer who
  finds no unit leaves (loss system). Service is non-preemptive.
- Control: an admission probability x, possibly depending on the number of free units
  (stock-dependent). Reward rate lambda g(x) with g concave.
- Objective: long-run average reward, lim inf (1/T) E [integral of lambda g(x_t) dt].
  This is a time average, not a discounted sum.
- Benchmark: the fluid relaxation FLU = max lambda g(x) subject to lambda x d <= c
  (Little's law in expectation). FLU upper-bounds every policy.
- Asymptotic regime (their section 3.3): scale c and lambda proportionally, keep the
  usage-duration distribution fixed, stay scarce (c / (lambda d) < 1). FLU grows linearly
  in c, so a policy with loss o(c) is asymptotically optimal. Static policies lose
  Theta(sqrt(c)); stock-dependent and two-price policies lose O(c^(1/(1+alpha))), down to
  O((log c)^2) when the reward is locally linear.
- Multiple classes (their 6.1): each class i has its own reward g_i, arrival rate lambda_i
  and mean duration d_i; the regime scales all lambda_i and c together with the d_i fixed;
  FLU(m) = max sum lambda_i g_i(x_i) subject to sum lambda_i x_i d_i <= c. Their policy
  dedicates floor(lambda_i x_i* d_i) units to class i and runs a two-price policy per class.
- Their conclusion names state-dependent usage durations (how long a customer stays
  depends on congestion) as an open extension. That is exactly our setting.

## 2. Our system as an instance of that model

| Their object | Ours |
|---|---|
| resource unit | one KV block (16 tokens) |
| c units | n instances x M blocks per instance (M = 15,909 at the auto pool, 2,500 in the memory-bound regime) |
| customer class i | request type i in {chat, RAG, long generation} with prompt P_i and answer length o_i |
| usage duration | the request's sojourn from admission to its last token; during decode it holds ceil((P_i + generated tokens) / 16) blocks, a growing occupancy, released at completion |
| d_i | the block-seconds a type-i request consumes; not exogenous: the decode step time grows with the batch on its instance, so d_i depends on congestion |
| admission probability x_i | the admission policy's accept decision per type; today always-admit, so x_i = 1 |
| loss system | not by default: rejected requests leave, admitted requests queue. Under always-admit the queue is unbounded and the system is unstable above capacity. An admission policy that rejects when the cluster is full turns it into a loss system, which is the regime where their theory applies |
| non-preemptive service | violated at overload: BLIS evicts the last-admitted request when KV is exhausted and recomputes it from scratch. Evictions are the cost of admitting too much, and an admission policy that keeps occupancy below M avoids them |
| reward g_i | our per-request reward pi_in P + pi_out o, minus the congestion cost h_i times sojourn; the WTP interpretation is replaced by a fixed price per token |
| long-run average reward | our value per second over the steady-state window (ours/objective.py) |

Two departures that our paper can name as contributions: (1) usage durations are
state-dependent through the batch, and (2) there are n parallel resources with a router,
so pooling across instances is part of the policy.

## 3. The sequence of systems

System n, for n in {1, 2, 4, 8, 16}:

- n instances, each with the same GPU, model, M blocks and per-step budgets as today.
- Total arrival rate lambda_n = n x lambda_bar, type shares fixed (0.5, 0.35, 0.15),
  length distributions fixed. So c_n = n M and lambda_n scale together, d_i fixed in
  distribution, exactly their regime.
- lambda_bar is chosen in the scarce regime: lambda_bar = 1.2 x (capacity of one
  instance under the B1 baseline), where capacity is R_cap / 4 from the capacity
  experiments. At 0.9 x capacity the fluid constraint is slack and admission has nothing
  to do; the scarce point is where policies differ.
- Everything reported per instance: arrival rate lambda_n / n, throughput / n, value per
  second / n, evictions per second / n. Under their regime these per-unit quantities
  converge as n grows, and FLU_n / n is a constant.
- The performance loss of a policy is FLU_n minus its long-run average value; plotting
  loss against n on log axes gives the empirical scaling exponent, as in their Figure 6.
  FLU_n for our system is a linear program over the type admission fractions with the
  block-seconds constraint sum_i lambda_i x_i E[blocks x seconds]_i <= n M, where the
  block-second consumption per type is measured from an uncongested run (n = 1, low rate).

## 4. What the simulation must record for this (all already in place)

- Per request: type, arrival, prompt and answer tokens, status in {completed, unfinished,
  rejected}, wait, TTFT, E2E, evictions, wasted tokens. Rejected requests appear as rows
  with status rejected, so rejection statistics per type are available the moment an
  admission policy starts rejecting; they read as 0 percent today, not as absent.
- Per instance over time: queue, batch, KV occupancy (state csv) and composition (snapshot
  csv), for the block-seconds measurement and for the visualizer.
- Per path: value per second = reward per second minus congestion cost per second; across
  paths: mean and t interval; between policies: paired differences on common seeds.

## 5. Cost of the scaling experiment

Simulation time grows roughly with n x lambda_n, that is with n squared per second of
simulated time. With --jobs on 8 cores: n = 16 at 1,200 s takes about 4 x 70 s per path.
Five seeds per n and five values of n is 25 paths, under 30 minutes. K = 5 is the working
value for scaling runs; K = 20 stays for the fixed-size policy comparisons.

## 6. Open points to settle before running

- Prices pi_in, pi_out and h_i per type (needed for FLU_n and for the objective).
- Whether the congestion cost charges the whole sojourn or only the queue wait.
- The admission policy family to test first: a static per-type accept probability (their
  static policy), an occupancy threshold per type (their two-price analogue: admit type i
  only when free blocks exceed a threshold), and always-admit as the baseline.
