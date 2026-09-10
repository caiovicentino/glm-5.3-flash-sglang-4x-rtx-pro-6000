# Memory model on one RTX PRO 6000 (96 GB), TP4

Numbers from the boot log (`Mamba Cache is allocated…`, `KV Cache is allocated…`) and
`/get_server_info` → `memory_usage`, marlin build (2026-09-04).

| region | size | how it is set |
|---|---|---|
| weights (1/4 of the model per GPU) | ~45.8 GB (cutlass) / ~49 GB (marlin, repacked) | fixed |
| KV pool (fp8_e4m3, page 64) | 15.3 GB = 2.26 M tokens (cutlass) · 12.5 GB = 1.85 M (marlin) | `mem-fraction-static` minus weights minus KDA pool |
| KV of the MTP (NextN) layer | 1.4 GB | same |
| KDA state pool | 13.1 GB = 170 slots (cutlass) · 146 slots (marlin) | `--mamba-full-memory-ratio` 0.9 (default): `rest × r/(1+r)` |
| CUDA graphs | ~4 GB with 4 speculation states and 8 sizes; ~2 GB with 1 state and 7 sizes; +1.44 GB per extra state | speculation states × captured batch sizes × 3 families (target_verify, draft_decode, draft_extend) |
| dynamic region | ~12 GB at boot | `1 − mem-fraction` (0.20) — activations, DSA indexer logits, ViT, allocator cache |

## KDA state slots

Each slot is one sequence's linear-attention state: 34 layers × (64/4 heads × 128 × 128 × fp32)
= 34 MiB of SSM state + conv, plus the MTP *intermediate* state (sized by the maximum draft
tokens across speculation candidates: 6.84 GB/GPU with candidates [1,3,5,7], 5.13 GB with max 5
steps). ~77 MB per slot all-in.

A running request occupies **5 slots** under the default `extra_buffer` strategy with the
overlap scheduler (3 base + 2 ping-pong); `extra_buffer_lazy` makes it 4. The engine caps
`max_running_requests` at `slots // 5` (log line "capped to N by the mamba state cache").
Slots not in use hold **checkpoints for the radix cache**: a prefix hit is only valid up to the
nearest KDA checkpoint (taken every `--mamba-track-interval` tokens, 64 here), so shrinking the
pool with `--max-mamba-cache-size` is not free; `--mamba-max-states-per-path` is the lossless way
to trim checkpoints if you ever need the memory.

## The KV pool is full — of cache

`sglang:token_usage` excludes evictable (cached) tokens. In production the pool reads "3–16% used"
while `kv_evictable_tokens` is ~97% of it. 300–500 M tokens are evicted per day to keep a
92–95% hit rate with 80–105k-token sessions (the pool holds ~18–21 such sessions warm). Capping
`--max-total-tokens` to "free" memory would evict live sessions.

## The DSA indexer buffer (the 16k-chunk OOM)

During prefill the indexer materializes fp32 logits of shape `[chunk, context]`:
16,384 × 61,000 × 4 B = 3.77 GiB — the exact size in our OOM. The engine splits it under a budget
`min(free × SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION(0.2), VRAM × (1 − mem_fraction) × 0.2, …)`
computed **once at the first prefill**; when the allocator cache later eats the headroom, the
allocation fails. Keep chunk 8192 and mem-fraction 0.80.

## The "leak"

VRAM free drops ~3 GB in the first 80 min (warm-up) and then 35–160 MiB/h. The engine's invariant
checker verifies the KV and KDA pools on every idle pass and would abort with "memory leak
detected" — it never did. What grows is torch's allocator cache. With `--sleep-on-idle` and
`SGLANG_EMPTY_CACHE_INTERVAL=3600`, rank 0 (the only rank with an `IdleSleeper`) stayed flat for
days; ranks 1–3 creep slowly. `/flush_cache` is not a soft restart: it resets the metrics counters.
