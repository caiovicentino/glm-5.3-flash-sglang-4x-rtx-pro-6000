# Tuning log (chronological, one variable per restart)

| date | change | result (per batch size, 24 h) |
|---|---|---|
| 08-26 | switch to GLM-5.3-Flash-NVFP4 on SGLang, 0xSero's recipe | boots; five outages while retuning (see incidents) |
| 08-26 | `--max-running-requests 8 → 24` | queue of 7–10 disappears |
| 08-27 | `--cuda-graph-max-bs-decode 32` | OOM at capture (4 spec states × sizes); back to 8 |
| 08-27 | chunk 16384 | OOM (indexer fp32 logits 3.77 GiB); back to 8192 |
| 08-31 | mem-fraction 0.85 | OOM after 4d22h; → 0.80 |
| 09-02 | `--ep-size 4 → 1` | +4% b1, +3% b2, +7% b4, +11% b6; better accept |
| 09-02 | "combo": static 5/6 spec, graphs to 24, KDA interval 64, sleep-on-idle+empty_cache | b1 =, **b2–6 −6…−24%**, bursts b10–16 155–234 → 450–480; +2 GB free at boot; GPU0 memory flat |
| 09-03 | adaptive spec with one candidate per bucket (b1: 5, b≥2: 3) | b1 +3%, b2 +3%, b3–6 parity, b8 +25%, b10–16 2–3× vs pre-combo |
| 09-04 | MoE `flashinfer_cutlass → marlin` | +9% b1, +6% b2, +4% b3–4, +2–8% b6–8; KV pool −18%, cache hit unchanged |
| 09-04 | NCCL env variants (bench only) | no change; P2P already on |
| pending | `--prefill-decode-interval 8–16` + chunk 4096 | target: inter-token p99 200–400 ms → <100 ms; cold TTFT +15–25% |

Measurement rules: two warm runs on an idle server for single-stream numbers (`prose_speed.py`);
`per_batch.sh` over a full day of production log for everything else; the same awk on both logs;
report the spread, not a single read.
