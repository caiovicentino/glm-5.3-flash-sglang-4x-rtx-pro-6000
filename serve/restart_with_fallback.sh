#!/bin/bash
# Atomic restart WITH FALLBACK. Kill + relaunch live in ONE detached process (an SSH session
# dropping between the kill and the relaunch once left production dead for 9 minutes).
# Brings up $NOVO; if /health does not return 200 AND the log has no "server is fired up"
# within LIMITE seconds, or the process dies, or the log shows an OOM/CUDA error signature,
# rolls back to $ANTIGO and makes it persistent.
#   trigger:  ANTIGO=/root/serve.known-good.sh LIMITE=1500 setsid nohup /root/restart_with_fallback.sh >/dev/null 2>&1 < /dev/null &
# Notes learned the hard way:
#   - "Traceback" is NOT a failure signature: a healthy boot prints ~150 of them (FlashInfer autotune).
#   - /health can return 503 for minutes after a restart while a storm of cold prefills drains
#     (the prefix cache is empty after a restart); hence the "server is fired up" alternative.
#   - do not trigger this inside an `a && b &` chain: the whole and-list is backgrounded as a
#     subshell that waits for the script and keeps the SSH session open.
exec >> /root/reinicio.log 2>&1
NOVO=${NOVO:-/root/serve_cmd_glm53.sh}      # the config being brought up
ANTIGO=${ANTIGO:?set ANTIGO=/path/to/known-good-serve.sh}   # fallback config
LIMITE=${LIMITE:-1200}
matar() {
  pkill -f "[s]glang.launch_server" 2>/dev/null
  for i in $(seq 1 12); do pgrep -f "[s]glang" >/dev/null || break; sleep 5; done
  pgrep -f "[s]glang" >/dev/null && { pkill -9 -f "[s]glang"; sleep 5; }
  for i in $(seq 1 20); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$U" -lt 2000 ] && break; sleep 3
  done
}
lancar() { mv -f /root/glm53.log /root/glm53.log.prev 2>/dev/null; setsid "$1" > /root/glm53.log 2>&1 < /dev/null & echo "  pid $!"; }
saudavel() { [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:8000/health)" = "200" ] || grep -aq "server is fired up" /root/glm53.log; }
falhou() { grep -aqE "CUDA out of memory|OutOfMemoryError|CUDA error|Segmentation fault|Killed" /root/glm53.log; }
esperar() {
  local t=0
  while [ $t -lt $LIMITE ]; do
    saudavel && return 0
    pgrep -f "[s]glang.launch_server" >/dev/null || { echo "  processo morreu em ${t}s"; return 1; }
    falhou && { echo "  assinatura de falha no log em ${t}s: $(grep -aoE 'CUDA out of memory|OutOfMemoryError|CUDA error|Segmentation fault|Killed' /root/glm53.log | head -1)"; return 1; }
    sleep 10; t=$((t+10))
  done
  echo "  expirou ${LIMITE}s sem /health 200"; return 1
}
echo "[$(date -u +%FT%TZ)] restart started (novo=$(sha256sum $NOVO | cut -c1-8) antigo=$(sha256sum $ANTIGO | cut -c1-8))"
matar
echo "[$(date -u +%FT%TZ)] launching new config"; lancar "$NOVO"
if esperar; then echo "[$(date -u +%FT%TZ)] ✅ NEW CONFIG HEALTHY"; exit 0; fi
echo "[$(date -u +%FT%TZ)] ⚠️ NEW CONFIG FAILED — rolling back"
cp -f /root/glm53.log /root/glm53.log.failed-$(date -u +%Y%m%dT%H%M%SZ)
matar
cp -f "$ANTIGO" "$NOVO"
echo "[$(date -u +%FT%TZ)] launching fallback config"; lancar "$ANTIGO"
if esperar; then echo "[$(date -u +%FT%TZ)] ✅ FALLBACK HEALTHY (fallback config persisted as $NOVO)"
else echo "[$(date -u +%FT%TZ)] 🔴 FALLBACK ALSO FAILED — manual intervention needed"; fi
