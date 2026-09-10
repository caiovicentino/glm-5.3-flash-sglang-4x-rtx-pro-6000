#!/bin/bash
# GLM-5.3-Flash-NVFP4 on 4x RTX PRO 6000 Blackwell (SM120) with SGLang — production serve script.
# Image: lmsysorg/sglang:glm-5.3-flash (dev build, commit ~f13cb6f6a7, 2026-08-26), flashinfer 0.6.17,
# deep_gemm 0.1.5.post3, torch 2.13+cu130. Six SM120 patches from 0xSero applied (see ../patches/).
# Every non-default flag below has a measured reason; see README.md ("Why each flag").
#
# 2026-09-04 (current): MoE in MARLIN (W4A16) — +4..+16% per batch size over flashinfer_cutlass,
#   at the cost of ~18% smaller KV pool (weights repacked take more VRAM).
# 2026-09-03: adaptive speculation with ONE candidate per batch bucket (batch 1 -> 5 steps,
#   batch >=2 -> 3 steps). Static 5/6 lost 6-24% at batch 2-6; the default adaptive config
#   flipped 3<->5 ~1,000 times/hour.
# 2026-09-02: CUDA graphs up to batch 24 (bursts of 9-24 ran eager: 155-234 tok/s -> 450-540),
#   KDA checkpoint every 64 tokens, sleep-on-idle + hourly empty_cache.

# ── a NCCL do vLLM esta em /etc/environment e QUEBRA o sglang (deep_ep aborta) ──
unset LD_PRELOAD
unset VLLM_NCCL_SO_PATH VLLM_ENABLE_PCIE_ALLREDUCE VLLM_PCIE_ALLREDUCE_BACKEND
unset VLLM_PCIE_ONESHOT_MAX_BYTES VLLM_USE_B12X_MOE VLLM_USE_B12X_FP8_GEMM
unset VLLM_USE_B12X_SPARSE_INDEXER NCCL_MIN_NCHANNELS NCCL_NET_GDR_LEVEL
unset NCCL_ALLOC_P2P_NET_LL_BUFFERS

# ── ENV do Dockerfile do Sero, verbatim ──
# Fragmentacao: em 31/08, apos 4d22h de uptime, o alocador padrao tinha picado
# a VRAM a ponto de um pedido de 296 MiB falhar com o KV a 2% de uso.
# expandable_segments usa memoria virtual e reune blocos livres.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_CUDA_ARCH_LIST=12.0a
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_FP8_WO_A_GEMM=0
# ── COMBO 2026-09-02: empty_cache no laco ocioso (so age com --sleep-on-idle) ──
export SGLANG_EMPTY_CACHE_INTERVAL=3600

# ── a chave de API atual, sem ecoar ──
# API key: read from env or a 600-mode file; never hardcode it in this script.
KEY="${API_KEY:-$(cat /root/.api_key 2>/dev/null)}"
[ -n "$KEY" ] || { echo "API_KEY not set (export API_KEY=... or write /root/.api_key)"; exit 1; }

# Served model name (what clients put in "model"). Keep it stable across restarts.
NOME="${SERVED_MODEL_NAME:-glm-5.3-flash}"

exec /opt/sglang/bin/python -m sglang.launch_server \
  --model-path "${MODEL_PATH:-/root/model-glm53}" \
  --served-model-name "$NOME" \
  --tp-size 4 --ep-size 1 \
  --context-length 1048576 \
  --quantization modelopt_fp4 \
  --attention-backend dsa \
  --dsa-prefill-backend flashinfer_sparse_mla \
  --dsa-decode-backend flashinfer_sparse_mla \
  --linear-attn-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend marlin \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --max-running-requests 24 \
  --mem-fraction-static 0.80 \
  --cuda-graph-max-bs-decode 24 \
  --cuda-graph-bs-decode 1 2 4 8 12 16 24 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --speculative-adaptive \
  --speculative-adaptive-config "${ADAPTIVE_SPEC:-/root/adaptive_spec.json}" \
  --mamba-track-interval 64 \
  --sleep-on-idle \
  --enable-multimodal \
  --enable-metrics \
  --media-url-max-file-size-mb 1024 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --api-key "$KEY" \
  --host 0.0.0.0 --port 8000
