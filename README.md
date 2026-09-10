# GLM-5.3-Flash on 4× RTX PRO 6000 Blackwell (SM120) — a multi-user production recipe

This is how we serve **GLM-5.3-Flash-NVFP4** (320B total / 18B active, vision, 1M context) to a
**multi-user agentic workload** — 24 concurrent requests, ~80–105k-token contexts, ~40k requests/day,
95% prefix-cache hit rate — on four RTX PRO 6000 Blackwell (96 GB, SM120, **no NVLink, two CPU
sockets**), with **SGLang**. It has run in production since 2026-08-26; every flag below has a
measured reason, and every idea we tried and rejected is listed with its number.

Most public recipes for this hardware optimize a single stream. This one optimizes **tokens per
second for many concurrent agents with huge contexts**, which is a different objective: our batch-1
number (~175 tok/s) is deliberately not the best you can get on this silicon; our batch-8 number
(~490 tok/s aggregate) and our 24-way concurrency are.

Everything here is MIT. The six SM120 patches are [0xSero's](https://github.com/0xSero/glm-5.3-flash-sglang-sm120)
and are fetched from his repository, not vendored.

## Recipe card

| | |
|---|---|
| hardware | 4× RTX PRO 6000 Blackwell 96 GB (SM120), PCIe Gen5 x16, **no NVLink**, 2 sockets (GPU0–1 / GPU2–3), 1 TB RAM, 420 GB disk |
| checkpoint | `LibertAIDAI/GLM-5.3-Flash-NVFP4` rev `9e0d74e3` (182 GiB): routed experts NVFP4 g16, attention/embeddings/lm_head BF16, vision tower BF16 |
| engine | `lmsysorg/sglang:glm-5.3-flash` (dev build, upstream ~`f13cb6f6a7`, 2026-08-26), flashinfer 0.6.17, deep_gemm 0.1.5.post3, torch 2.13+cu130, + 6 SM120 patches |
| parallelism | TP4, **EP1** (EP4 measured 3–11% slower at every batch size) |
| attention | `dsa` with `flashinfer_sparse_mla` prefill+decode, KV **fp8_e4m3**, page 64; KDA (34 linear layers) in Triton |
| MoE | **marlin** (W4A16) — +4…+16% per batch size over `flashinfer_cutlass` (W4A4); costs ~18% of the KV pool |
| speculation | NEXTN (EAGLE) topk 1, adaptive with **one candidate per batch bucket**: 5 steps at batch 1, 3 steps at batch ≥2 |
| graphs | CUDA graphs for decode at batch `1 2 4 8 12 16 24` |
| memory | `--mem-fraction-static 0.80`; KV pool 1.85 M tokens (fp8); KDA state pool 146 slots (5 per running request) |
| prefill | chunk 8192 (16k OOMs: see below), prefill CUDA graph disabled by the engine (incompatible with KDA) |
| concurrency | `--max-running-requests 24` (recipe default of 8 produced a 7–10-deep queue) |
| ops | atomic restart with fallback, 5-min telemetry series, quality gates, hourly `empty_cache` on idle |

The exact command: [`serve/serve.sh`](serve/serve.sh). The adaptive-speculation config: [`serve/adaptive_spec.json`](serve/adaptive_spec.json).

## Measured (production logs, per batch size, 24 h windows)

Decode throughput is the mean of SGLang's `Decode batch` log lines grouped by `#running-req`
([`bench/per_batch.sh`](bench/per_batch.sh)). This stratification is the only honest way to judge a
change on a multi-user box: aggregate numbers hid a −6…−24% regression at batch 2–6 while batch 1
looked fine.

| batch | recipe default (adaptive [1,3,5,7], graphs ≤8, cutlass, EP4) | + EP1 | + graphs ≤24, per-bucket spec | + marlin (current) |
|---|---|---|---|---|
| 1 | 143 tok/s | 149 | 153 | **173–197** |
| 2 | 235 | 242 | 249 | **263–285** |
| 4 | 335 | 360 | 359 | **374–388** |
| 6 | 378 | 420 | 385 | **394–431** |
| 8 | 322 | 350 | 438 | **472–508** |
| 10–16 | 155–234 (eager) | 155–234 (eager) | 453–480 | **485–542** |

Accept length of the MTP head: ~3.3 tokens/step at batch 1, ~2.8 at batch ≥2 (code in greedy
accepts more than prose). Prefill in full 8192-token chunks: **7,000–8,400 tok/s** at 50k depth,
~4,400 at 200k, ~2,500 above 300k (the DSA indexer scores every chunk against all keys).

Latency, 24 h of real traffic (40k requests, contexts p50 80k / p99 300k): time to first token
p50 0.8 s, p90 6 s, p99 40 s; inter-token p50 10 ms, p99 200 ms; queue wait p99 20–30 s; aborts
0.3%. The tail is entirely cold prefills of 100k+ tokens blocking admission (see "What still hurts").

## Why each flag (the part other recipes leave out)

- **`--ep-size 1`, not 4.** With EP4 each MoE layer needs an all-to-all across PCIe and two sockets;
  with TP4 every expert is sharded and the only communication is the allreduce we measured at
  17–22 µs per collective (~105 per step ≈ 2 ms). EP1 won at every batch size and had a higher
  MTP accept rate.
- **`--moe-runner-backend marlin`.** The image's `flashinfer_cutlass` W4A4 path falls back to slow
  CUTLASS tactics on SM120 (the TMA warp-specialized ones are sm_100a-only). Marlin dequantizes FP4
  to BF16 in registers, no activation quantization, tuned for small M — exactly decode. Trade-off:
  repacked weights take more VRAM (KV pool 2.26 M → 1.85 M tokens, KDA slots 170 → 146). Cache hit
  rate did not move (92–95%), so it stayed. `cutlass`, `flashinfer_trtllm`, `flashinfer_cutedsl`
  do not run NVFP4 MoE on SM120 in this build.
- **Adaptive speculation with one candidate per bucket** (`serve/adaptive_spec.json`). The default
  adaptive config `[1,3,5,7]` flipped between 3 and 5 steps **21,074 times in 20 hours** because at
  our acceptance (~2.3–2.8 accepted drafts) 3 and 5 are equivalent at batch 1 — it cannot decide,
  and each candidate costs a full set of CUDA graphs. Static 5/6 stopped the flipping but lost
  6–24% at batch 2–6: verifying 6 draft tokens per request activates far more experts than the
  extra accepted tokens pay for once more than one user is decoding. Pinning 5 at batch 1 and 3 at
  batch ≥2 keeps both regimes at their best with two graph states (+1.4 GB). Do not touch
  `--speculative-accept-threshold-*`: it is lossy.
- **`--cuda-graph-bs-decode 1 2 4 8 12 16 24`.** With graphs only up to 8, every step with 9–24
  requests ran eager (verify, draft and draft-extend): 155–234 tok/s aggregate. With graphs to 24:
  450–540. 24 is the cap (`--max-running-requests`); larger sizes are never used. Memory per size is
  ~0.4 GB per graph family; fewer speculation states pay for more sizes.
- **`--mem-fraction-static 0.80`, not 0.85.** The DSA indexer materializes an fp32 logits buffer of
  `chunk × context × 4 B` during prefill (16,384 × 61k × 4 B = 3.77 GiB was the exact number in our
  OOM), and its budget is computed once at the first prefill. 0.85 OOMed after days; 0.80 gives
  ~12 GB of dynamic headroom at boot.
- **`--chunked-prefill-size 8192`, not 16384.** Same buffer. Prefill cost is per token and grows
  with depth, not per chunk; a bigger chunk buys nothing and blocks decode for longer.
- **`--mamba-track-interval 64`.** The radix cache stores KDA state checkpoints; a prefix hit is
  only valid up to the nearest checkpoint. Default 256 re-prefilled up to 255 tokens per turn;
  64 makes it ≤63 (~17 MB per copy, negligible). Must be ≥ draft tokens and a multiple of page size.
- **`--sleep-on-idle` + `SGLANG_EMPTY_CACHE_INTERVAL=3600`.** The "VRAM leak" (~160 MiB/h after a
  3 GB warm-up) is torch allocator cache, not live memory: with hourly `empty_cache()` on idle, GPU0
  stayed flat for days. Note the `IdleSleeper` is only created on TP rank 0; ranks 1–3 still creep
  (~10–35 MiB/h) — a 6-line patch to `Scheduler.maybe_sleep_on_idle` would cover them.
- **`--max-running-requests 24`.** Each running request holds 5 KDA state slots (3 base + 2 overlap
  ping-pong); the engine caps concurrency at `slots // 5`. 24 fits comfortably; 32 would need
  more slots and, more importantly, more KV.
- **`--kv-cache-dtype fp8_e4m3` with `flashinfer_sparse_mla`.** On SM120 the fp8-packed `ds_mla`
  layout is what the native sparse-MLA kernel reads; `tilelang` needs bf16 KV (half the pool) and
  the stock kernel overflows SM120's 99 KiB shared memory. `flashmla_sparse`, `fa3`, `trtllm` are
  SM90/SM100 only.
- **Chat template.** The requant's `chat_template.jinja` silently disabled vision (it injected a
  "you are unable to process this image" reminder). Use the original `zai-org/GLM-5.3-Flash`
  template. Also: the template accepts `reasoning_effort: low|high` and **defaults to max** — see
  `docs/reasoning-effort.md`; a client that sends nothing pays 800–1,000 reasoning tokens even on a
  trivial prompt. `enable_thinking:false` is the worst option (the plan leaks into the content).

## What does NOT work on SM120 (so you don't spend the day we spent)

- `--ep-size 4` (slower), `--enable-pcie-oneshot-allreduce` (does not exist upstream),
  `--enable-flashinfer-allreduce-fusion` / `--enable-symm-mem` / `--enable-nccl-nvls` (SM90/SM100/NVSwitch),
  `--disable-custom-all-reduce` (no-op: the engine already disables it for TP>2 without NVLink).
- NCCL env tuning: the default already uses P2P/CUMEM on all four ring hops, cross-socket included.
  `NCCL_P2P_LEVEL=PHB` makes two hops fall back to SHM (worse); `NCCL_MIN_NCHANNELS=8`, `NCCL_PROTO=LL`
  do nothing at 8–48 KB; `NCCL_ALGO=Tree` is worse. Measured with [`bench/nccl_allreduce_bench.py`](bench/nccl_allreduce_bench.py).
- `--enable-mixed-chunk` (asserts with speculative decoding in this build), `--num-continuous-decode-steps`
  (dead flag, zero references), `--schedule-policy lpm` / `--schedule-conservativeness` (no effect in
  this regime), HiCache for hybrid KDA models (nodes with a Mamba component are pruned instead of
  offloaded, issue #33713), DCP (the SM120 sparse-MLA kernel returns no LSE), DP attention (kills
  the prefix cache without a cache-aware router).
- KDA backends `cutedsl`, `nvidia_kda`, `ptx_kda`, `flashinfer` (SM100/SM103; some pass the gate and
  fail later). `helion` and `flashkda` would run but are not installed and gain ≤5%.
- Chunk 16k (OOM, see above). `mem-fraction 0.85` (OOM after days).
- DFLASH/DSPARK: the build lacks the GLM capture adapter (entered the branch 2026-08-27); the only
  public DFlash2 draft is CC BY-NC-ND. A better drafter is nonetheless the biggest batch-1 lever
  left (+40–60% estimated).

## What still hurts, and the knob for it

SGLang's scheduler is **prefill-first**: while a chunked prefill is in flight, no new request is
admitted and the running requests' decode does not run. A cold 100k-token turn is 13 chunks of
~1.2 s; everyone else sees a frozen stream for 15 s. `--enable-mixed-chunk` is forbidden with
speculation in this build; the knob is `--prefill-decode-interval N` (N decode rounds between
chunks: N=8 ≈ 15 tok/s for neighbours during a foreign prefill, cold TTFT +15%; N=32 ≈ 40 tok/s,
+50%), optionally with `--chunked-prefill-size 4096`. Judge it with inter-token p99 and queue p99
from the telemetry below. The real cure is smaller prompts: when our clients' prompt p50 dropped
from 105k to 80k, TTFT p99 went from 100 s to 40 s and aborts from 1.8% to 0.3%.

## Operations

- **Restart atomically, with fallback** — [`serve/restart_with_fallback.sh`](serve/restart_with_fallback.sh).
  Kill and relaunch in one detached process; wait for `/health` 200 *or* "server is fired up" in the
  log; roll back to a known-good script on OOM/CUDA-error signatures, process death or timeout.
  "Traceback" is not a failure signature (a healthy boot prints ~150 from the FlashInfer autotune).
  Restart in a quiet hour: the prefix cache is empty afterwards and every cron comes back cold.
- **Validate before restarting** — parse the new argv without launching:
  `python -c "from sglang.srt.server_args import prepare_server_args as p; a=p(ARGV)"`; it also loads
  the adaptive JSON. Boot takes ~4 min (5–10 with marlin's weight repack); a 20-minute fallback
  budget is right.
- **Telemetry every 5 minutes** — [`observability/series.py`](observability/series.py) samples
  `/metrics` (latency histograms, queue, KV/KDA pools, accept length, aborts), `nvidia-smi` and the
  per-batch decode lines since the previous sample into a JSONL; [`agrega_series.py`](observability/agrega_series.py)
  turns the last 24 h into percentiles and per-batch tables. `/metrics` already had TTFT p99 and
  queue p99 — nobody was reading them.
- **Quality gates** — [`bench/gates.py`](bench/gates.py): models endpoint, long pt-BR generation
  (8,000 tokens: this model plans a 2,500-word essay inside its reasoning), opencode-shaped request
  with thinking, tool call, three vision probes, prefix cache, 8 concurrent streams. Run after every
  restart; keep the baseline per config.
- **A/B discipline** — one variable per restart, judged the next day by `per_batch.sh` on both logs.
- **Dashboard** — [`observability/dashboard/`](observability/dashboard/) (stdlib only, 127.0.0.1,
  no secret reaches the browser): live traffic, GPUs, stability, security checklist, daily history,
  and the 24 h cards. Works with just SSH via `CB_INST_IP/CB_INST_API/CB_INST_SSH`.

## Memory model (why the numbers are what they are)

Per GPU at boot: weights 45.8 GB (marlin: ~49) · KV 15.3 GB (2.26 M tokens fp8; 12.5 GB / 1.85 M
with marlin) + 1.4 GB for the MTP layer · KDA state pool 13.1 GB (170 slots × ~77 MB: SSM 5.8,
intermediate MTP state 5.1–6.8, conv 0.4) · CUDA graphs 2–4 GB · dynamic region ~12 GB. Details and
the formulas in [`docs/memory-model.md`](docs/memory-model.md). The KV pool is 97% full of cached
prefixes at all times (300–500 M tokens evicted/day at 92–95% hit rate): do not cap
`--max-total-tokens`.

## Would custom kernels help?

We researched it (10 agents, two microbenchmarks; [`docs/research-kernels.md`](docs/research-kernels.md)).
At batch 1, ~half the 20 ms step is MTP draft/verify, allreduce and gaps — kernels recover
~10–15% via fusions. At batch 16 the MoE grouped GEMM runs at ~32% of bandwidth (the only 2× slack)
and the kernel that fixes it exists (B12X, Apache-2.0, SM120-only) — in vLLM, not in SGLang.
The DSA indexer kernel in the image measured 646 TFLOPS fp8 (efficient). Order of value per week
of effort: engine flags → a better drafter → an engine A/B with B12X → writing kernels last.

## Layout

```
serve/          serve.sh · adaptive_spec.json · restart_with_fallback.sh · env.example
patches/        the six SM120 patches: what each does + fetch_patches.sh (from 0xSero's repo)
bench/          per_batch.sh · gates.py · prose_speed.py · bench_ab.py · canario.py
                auditoria_prefixo.py · nccl_allreduce_bench.py · dsa_indexer_bench.py
observability/  series.py · agrega_series.py · series_loop.sh · dashboard/
docs/           memory-model.md · sm120-does-not-work.md · incidents.md · tuning-log.md
                reasoning-effort.md · research-kernels.md
vast/lib.sh     helpers for a Vast.ai instance (optional)
```

## Credits

0xSero for the SM120 patches and the first working recipe; LibertAI for the NVFP4 checkpoint;
the SGLang team; the Local Inference Lab (rtx6kpro) for the vLLM+B12X reference numbers on the same
silicon. Built and operated by [CulturaBuilder](https://culturabuilder.com).
