# What does not work (or does nothing) on SM120 with this build

Verified in the code of `lmsysorg/sglang:glm-5.3-flash` (2026-08-26) and by measurement.

## Flags that exist but are SM90/SM100/NVSwitch-only
`--enable-flashinfer-allreduce-fusion` (PR #32330 tried SM120 and regressed in TP4),
`--enable-symm-mem` (cc 9/10 tables), `--enable-nccl-nvls` (NVSwitch), `trtllm`/`fa3`/
`flashmla_sparse` DSA backends, KDA backends `cutedsl` (NVVM fails for sm_120a), `nvidia_kda`
(silently falls back to Triton), `ptx_kda` (sm_103a), `flashinfer` recurrent KDA (sm100a/sm103a).

## Flags that do not exist or are dead
`--enable-pcie-oneshot-allreduce` (fork-only, driver override on the host, −15.7% cross-socket),
`--num-continuous-decode-steps` (zero references outside server_args), `SGLANG_ENABLE_SPEC_V2`
(Spec V1 was removed in June 2026), `SGLANG_OPT_FUSED_KDA_VERIFY` (AMD only / later commit).

## Things that are no-ops here
`--disable-custom-all-reduce` (the engine already disables custom allreduce for TP>2 without
NVLink), `SGLANG_OPT_USE_TILELANG_INDEXER`, `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` (read only by the
DeepSeek-V4 path), every NCCL env we tried (`NCCL_P2P_LEVEL`, `NCCL_MIN_NCHANNELS`, `NCCL_PROTO=LL`,
`NCCL_ALGO=Tree` — the default already uses P2P/CUMEM on all hops).

## Things that are forbidden or harmful with this configuration
`--enable-mixed-chunk` with speculative decoding (assert), `--chunked-prefill-size 16384` (indexer
buffer OOM), `--mem-fraction-static 0.85` (OOM after days), `--cuda-graph-max-bs-decode 32` with
four speculation states (OOM at capture), `--ep-size 4` (slower), `--max-running-requests 8`
(queue), `--speculative-accept-threshold-*` (lossy), `enable_thinking:false` in clients (reasoning
leaks into the content), HiCache for hybrid KDA (issue #33713), DCP (no LSE from the SM120 kernel),
DP attention (prefix cache), `tilelang` DSA (bf16 KV + 99 KiB smem cliff).

## Things that would run but were not worth a restart
`helion` and `flashkda` KDA backends (not installed; ≤5%), `--enable-torch-compile` (no SM120
evidence; +1.3% on the closest published case), chunk 4096 alone (only with
`--prefill-decode-interval`).
