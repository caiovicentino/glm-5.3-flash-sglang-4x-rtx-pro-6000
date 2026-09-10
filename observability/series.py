#!/usr/bin/env python3
"""Série temporal da máquina de inferência, a cada 5 min, em series.jsonl.

Separado do collect.py DE PROPÓSITO: a atestação é registro de conformidade e não
pode depender deste arquivo. Este guarda o que o /metrics do SGLang expõe e ninguém
lia: histogramas de latência (primeiro token, entre tokens, fila, ponta a ponta,
prompt não cacheado), fila e em voo, KV/KDA, aceitação do MTP, abortos, VRAM e
potência por GPU, e o throughput por lote das linhas "Decode batch" do log desde a
amostra anterior. Só números; nenhum conteúdo de prompt passa por aqui.

Uso: series.py            (laço infinito, 300 s)
     series.py --once     (uma amostra, para teste)
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

OUT = "/root/attestation/series.jsonl"
STATE = "/root/attestation/series.state"
METRICS = "http://127.0.0.1:8000/metrics"
LOG = "/root/glm53.log"
INTERVALO = 300

GAUGES = ("num_running_reqs", "num_queue_reqs", "kv_used_tokens", "kv_evictable_tokens",
          "kv_available_tokens", "mamba_used_tokens", "mamba_evictable_tokens",
          "spec_accept_length", "spec_num_steps", "graph_memory_usage_gb", "token_usage",
          "gen_throughput", "num_retracted_reqs")
COUNTERS = ("num_requests_total", "num_aborted_requests_total", "prompt_tokens_total",
            "cached_tokens_total", "generation_tokens_total", "evicted_tokens_total")
HISTS = ("time_to_first_token_seconds", "inter_token_latency_seconds", "queue_time_seconds",
         "e2e_request_latency_seconds", "uncached_prompt_tokens_histogram",
         "prompt_tokens_histogram")

RX_LINHA = re.compile(r"^sglang:(\w+)\{([^}]*)\}\s+([0-9.eE+-]+)")
RX_LE = re.compile(r'le="([^"]+)"')
RX_DEC = re.compile(r"#running-req: (\d+),.*accept len: ([\d.]+).*gen throughput \(token/s\): ([\d.]+)")
RX_PRE = re.compile(r"#new-token: (\d+), #cached-token: (\d+).*#queue-req: (\d+).*input throughput \(token/s\): ([\d.]+)")


def agora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def metricas():
    """Gauges: max entre ranks (o SGLang repete o mesmo valor por tp_rank).
    Contadores e histogramas: soma entre rótulos (is_streaming true/false)."""
    g, c, h = {}, {}, {}
    with urllib.request.urlopen(METRICS, timeout=10) as r:
        texto = r.read().decode("utf-8", "replace")
    for ln in texto.splitlines():
        m = RX_LINHA.match(ln)
        if not m:
            continue
        nome, rot, v = m.group(1), m.group(2), float(m.group(3))
        if nome in GAUGES:
            g[nome] = max(g.get(nome, 0.0), v)
        elif nome in COUNTERS:
            c[nome] = c.get(nome, 0.0) + v
        else:
            for suf in ("_bucket", "_sum", "_count"):
                if nome.endswith(suf) and nome[: -len(suf)] in HISTS:
                    base = nome[: -len(suf)]
                    d = h.setdefault(base, {"b": {}, "s": 0.0, "n": 0.0})
                    if suf == "_bucket":
                        le = RX_LE.search(rot)
                        if le:
                            d["b"][le.group(1)] = d["b"].get(le.group(1), 0.0) + v
                    elif suf == "_sum":
                        d["s"] += v
                    else:
                        d["n"] += v
                    break
    return g, c, h


def gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
    res = []
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) == 4:
            res.append([int(float(p[0])), int(float(p[1])), int(float(p[2])), round(float(p[3]), 1)])
    return res


def pid_servidor():
    out = subprocess.run(["pgrep", "-f", "[s]glang.launch_server"], capture_output=True, text=True).stdout.split()
    return min(int(x) for x in out) if out else None


def ler_estado():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def log_desde_ultima(estado):
    """Lê só o que o log ganhou desde a amostra anterior. Na primeira vez, ou depois
    de rotação (inode/tamanho mudou), começa do FIM: histórico não entra."""
    try:
        st = os.stat(LOG)
    except OSError:
        return {}, estado
    ino, tam = st.st_ino, st.st_size
    off = estado.get("off", 0) if estado.get("ino") == ino and estado.get("off", 0) <= tam else tam
    bs, pf = {}, {"n": 0, "new": 0, "cached": 0, "n_full": 0, "tps_full": 0.0, "fila_max": 0}
    with open(LOG, "rb") as f:
        f.seek(off)
        bloco = f.read()
    for ln in bloco.decode("utf-8", "replace").splitlines():
        if "Decode batch" in ln:
            m = RX_DEC.search(ln)
            if m:
                k = m.group(1)
                a = bs.setdefault(k, [0, 0.0, 0.0])
                a[0] += 1
                a[1] += float(m.group(3))
                a[2] += float(m.group(2))
        elif "Prefill batch" in ln:
            m = RX_PRE.search(ln)
            if m:
                novo, cach, fila, tps = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
                pf["n"] += 1
                pf["new"] += novo
                pf["cached"] += cach
                pf["fila_max"] = max(pf["fila_max"], fila)
                if novo >= 8000:
                    pf["n_full"] += 1
                    pf["tps_full"] += tps
    novo_estado = {"ino": ino, "off": tam}
    return {"bs": bs, "pf": pf}, novo_estado


def amostra():
    estado = ler_estado()
    g, c, h = metricas()
    logs, novo_estado = log_desde_ultima(estado)
    rec = {"ts": agora(), "pid": pid_servidor(), "g": g, "c": c, "h": h, "gpu": gpus(), **logs}
    with open(STATE, "w") as f:
        json.dump(novo_estado, f)
    return rec


def gravar(rec):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    uma = "--once" in sys.argv
    while True:
        try:
            gravar(amostra())
        except Exception as err:  # o coletor nunca derruba nada
            try:
                gravar({"ts": agora(), "error": str(err)[:200]})
            except OSError:
                pass
        if uma:
            return
        time.sleep(INTERVALO)


if __name__ == "__main__":
    main()
