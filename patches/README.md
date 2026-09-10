# The six SM120 patches (from 0xSero)

The stock image `lmsysorg/sglang:glm-5.3-flash` does not run GLM-5.3-Flash on SM120 as is.
Six files from https://github.com/0xSero/glm-5.3-flash-sglang-sm120 are copied over the image's
tree (they are full-file replacements, +315/-17 lines in total). They are his work and are not
vendored here; `fetch_patches.sh` downloads them from his repository at the commit we run.

| file in the SGLang tree | delta | what it does |
|---|---|---|
| `kernels/ops/attention/flash_mla_sm120.py` | +217/-4 | makes the SM120 sparse-MLA kernel accept the GLM-5.3 shape (16 heads/GPU, topk 2048, fp8 KV); without it there is no native DSA kernel |
| `srt/layers/quantization/modelopt_quant.py` | +18/-0 | fixes the slicing of NVFP4 expert scales |
| `srt/layers/quantization/utils.py` | +27/-5 | SM120 quantization utilities |
| `srt/models/glm5_next.py` | +32/-2 | model file with SM120 fixes |
| `srt/models/deepseek_nextn.py` | +17/-4 | the NextN (MTP) layer for this checkpoint |
| `srt/layers/attention/dsa_backend.py` | +4/-2 | accepts `flashinfer_sparse_mla` as prefill and decode backend |

The first four are what makes the model boot at all. Apply BEFORE downloading the model and
verify `python -c "import sglang.srt.models.glm5_next"` succeeds — if it fails you have destroyed
nothing yet.

Two things we did NOT need from his recipe, verified in this build's code:
- `SGLANG_OPT_USE_TILELANG_INDEXER=1` and `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` are only read by the
  DeepSeek-V4 path (`layers/attention/dsv4/`). The GLM path (`layers/attention/dsa/`) uses
  `deep_gemm.fp8_paged_mqa_logits` — the image's deep_gemm has SM120 kernels for it. Harmless, inert.
- `--max-running-requests 8` (his value) is a single-user setting; we run 24 (see README).
