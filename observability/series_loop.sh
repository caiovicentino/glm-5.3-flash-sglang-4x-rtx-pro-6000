#!/bin/bash
# Supervisor do coletor de série temporal (series.py). Espelha o attestation_loop.sh,
# mas é um processo SEPARADO: um erro aqui nunca toca a atestação.
while true; do
  if [ ! -f /root/attestation/series.py ]; then
    echo "[$(date -u +%FT%TZ)] FALTA /root/attestation/series.py" >> /root/attestation/series.err
    sleep 300; continue
  fi
  python3 /root/attestation/series.py >> /root/attestation/series.err 2>&1
  echo "[$(date -u +%FT%TZ)] coletor de serie saiu — relancando em 30s" >> /root/attestation/series.err
  sleep 30
done
