# Would kernels written for this box help? (research summary, 2026-09-04)

Ten research agents plus two of our own microbenchmarks. Verified claims cite the source in the
original write-up; numbers below are the conclusions.

**Where a batch-1 step (~20 ms) goes** (estimates): MoE GEMM 6–7 ms (marlin; the 6-token verify
touches ~48 experts/layer), dense attention + KDA 3–4 ms (KDA 0.5–0.7, DSA 1.5–2.5), allreduce
~2 ms (measured: 17–22 µs × ~105 collectives), MTP draft + extra verify 7–8 ms, gaps/CPU 1–2 ms.
Kernel-recoverable: 1.5–2.5 ms of fusions (KDA chain conv+recurrence+norm; MoE glue
align/topk/silu/quant) — 8–14%, weeks of CUDA. Megakernels (Hazy, Mirage MPK) need NVLink or a
single GPU and do not cover quantized MoE + hybrid attention + MTP over PCIe TP4.

**Where a batch-16 step (~93 ms) goes**: MoE grouped GEMM 55–65 ms against a ~28–30 ms bandwidth
floor (marlin at ~32% of bandwidth; a published GLM-5 measurement on H20 says 24%). This is the
only real 2× slack, and the kernel that closes it exists: **B12X** (Apache-2.0, SM120-only CuTe
DSL, persistent fused MoE). It lives in vLLM (`flashinfer_b12x`, since 2026-05); SGLang's PR
#29190 has been stalled since August. Isolated, B12X vs marlin is +0.6% at 1 prompt and +10% at 8;
the whole vLLM "Jovian" stack on the same silicon and model reports C1 248 tok/s (10.1 ms/step),
C8 903 tok/s, prefill 14.5k tok/s — the stack (one-shot allreduce, fused KDA/attention, piecewise
graphs), not one kernel.

**Prefill**: DSA is ~20% of a 100k cold prefill and nearly half at 400k (the indexer is the
quadratic term), but the image's `deep_gemm.fp8_mqa_logits` SM120 kernel measured **646 TFLOPS
fp8** with GLM shapes — it is not slow. The prefill collapse above 300k needs a profile, not a
kernel. The KDA prefill kernel in flashinfer main (#4633, SM120, 2.3–4.1× vs FlashKDA) has no
SGLang wrapper yet (5–8% of prefill).

**Communication**: 17–22 µs per collective idle, ~2 ms/step at batch 1, <10% at batch 16. No env
helps; cross-socket custom kernels measured worse in public data. TP2×PP2 is blocked with
speculation.

**KDA**: 3–4% of the batch-1 step; at batch 16 ~4 ms of verify snapshots, addressed by
`--enable-linear-replayssm-spec` (in the image) rather than a kernel.

**Order of value per week of effort**: engine flags already in the image
(`--prefill-decode-interval`, chunk 4096, `--enable-linear-replayssm-spec`, int8 KDA checkpoints)
→ a better drafter (+40–60% at batch 1, ≤1.2× at batch ≥2; blocked by licence/adapter) → an
engine A/B with B12X on a rented box → fused kernels via agent loops (KDA chain, MoE glue,
≈10–15% at batch 1) → writing a competitive NVFP4 grouped GEMM last (4–8 weeks to match marlin;
B12X took one expert 5.5 months).
