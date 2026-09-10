#!/usr/bin/env bash
# Download the six SM120 patch files from 0xSero's repository and copy them over the SGLang tree
# extracted from the image (SGL=/sgl-workspace/sglang/python/sglang on the lmsysorg image).
# Pin REF to the commit you validated; diff before copying — the total is small and reviewable.
set -euo pipefail
REF="${REF:-main}"
SGL="${SGL:-/sgl-workspace/sglang/python/sglang}"
BASE="https://raw.githubusercontent.com/0xSero/glm-5.3-flash-sglang-sm120/$REF/patches"
declare -A MAP=(
  ["sglang-flash_mla_sm120-glm53.py"]="kernels/ops/attention/flash_mla_sm120.py"
  ["sglang-modelopt-quant-sm120.py"]="srt/layers/quantization/modelopt_quant.py"
  ["sglang-quant-utils-sm120.py"]="srt/layers/quantization/utils.py"
  ["sglang-glm5_next-debug.py"]="srt/models/glm5_next.py"
  ["sglang-deepseek_nextn-glm53.py"]="srt/models/deepseek_nextn.py"
  ["sglang-dsa_backend-glm53.py"]="srt/layers/attention/dsa_backend.py"
)
mkdir -p /root/sero-patches
for src in "${!MAP[@]}"; do
  dst="$SGL/${MAP[$src]}"
  curl -fsSL "$BASE/$src" -o "/root/sero-patches/$src"
  echo "== $src -> $dst"; diff -u "$dst" "/root/sero-patches/$src" | head -5 || true
  cp -f "$dst" "$dst.orig" 2>/dev/null || true
  cp -f "/root/sero-patches/$src" "$dst"
done
echo "patches applied; now: python -c 'import sglang.srt.models.glm5_next' (must succeed)"
