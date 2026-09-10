#!/usr/bin/env bash
# Per-batch-size decode throughput from the SGLang log ("Decode batch" lines): samples, mean accept length,
# mean tok/s. THIS is what judges a config change the next day — aggregate numbers hid a -6..-24%
# regression at batch 2-6 while batch 1 looked fine. Usage: per_batch.sh /root/glm53.log [min_samples]
LOG="${1:-/root/glm53.log}"; MIN="${2:-30}"
awk '/server is fired up/{f=1} f' "$LOG" | grep -a "Decode batch" \
 | grep -oE "#running-req: [0-9]+,.*accept len: [0-9.]+.*gen throughput \(token/s\): [0-9.]+" \
 | awk -F'[:,]' -v min="$MIN" '{bs=$2+0; for(i=1;i<=NF;i++){if($i~/accept len/)a=$(i+1)+0; if($i~/gen throughput/)g=$(i+1)+0}; n[bs]++; A[bs]+=a; G[bs]+=g}
   END{for(b in n) if(n[b]>=min) printf "bs%-3s n=%-6d accept=%.2f  %.0f tok/s\n", b, n[b], A[b]/n[b], G[b]/n[b]}' | sort -t s -k2 -n
