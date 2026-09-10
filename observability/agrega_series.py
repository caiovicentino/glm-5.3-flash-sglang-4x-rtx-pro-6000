#!/usr/bin/env python3
"""Agrega o series.jsonl NA MÁQUINA e devolve o resumo das últimas 24 h em JSON.

Mesmo desenho do agrega_uso.py: o painel nunca baixa o arquivo bruto. Os
histogramas do SGLang são cumulativos e zeram a cada reinício, então os percentis
saem da DIFERENÇA entre a última e a primeira amostra de cada sessão (pid) dentro
da janela, somada entre sessões. Um percentil aqui é "≤ limite do bucket".

Saída: {"janela_h", "amostras", "lat": {...}, "vram": {...}, "lote": {...},
        "tot": {...}, "carga": {...}, "prefill": {...}}
"""
import calendar
import json
import os
import time

ARQ = "/root/attestation/series.jsonl"
JANELA_H = float(os.environ.get("JANELA_H", "24"))
ALARME_MIB = 3072


def t_epoch(ts):
    return calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))


def le_val(k):
    return float("inf") if k == "+Inf" else float(k)


def quantil(buckets, p):
    if not buckets:
        return None
    ks = sorted(buckets, key=le_val)
    tot = buckets[ks[-1]]
    if tot <= 0:
        return None
    for k in ks:
        if buckets[k] >= p * tot:
            return None if k == "+Inf" else float(k)
    return None


def delta_hist(a, b):
    """b - a, bucket a bucket (mesma sessão, b posterior)."""
    out = {"b": {}, "s": b["s"] - a["s"], "n": b["n"] - a["n"]}
    for k, v in b["b"].items():
        out["b"][k] = v - a["b"].get(k, 0.0)
    return out


def soma_hist(acc, d):
    if acc is None:
        return {"b": dict(d["b"]), "s": d["s"], "n": d["n"]}
    for k, v in d["b"].items():
        acc["b"][k] = acc["b"].get(k, 0.0) + v
    acc["s"] += d["s"]
    acc["n"] += d["n"]
    return acc


def main():
    recs = []
    try:
        with open(ARQ) as f:
            for l in f:
                l = l.strip()
                if not l.startswith("{"):
                    continue
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                if "g" in r and "ts" in r:
                    try:
                        r["t"] = t_epoch(r["ts"])
                    except ValueError:
                        continue
                    recs.append(r)
    except OSError as e:
        print(json.dumps({"erro": str(e)}))
        return 1
    if not recs:
        print(json.dumps({"erro": "sem amostras"}))
        return 1
    recs.sort(key=lambda r: r["t"])
    agora = recs[-1]["t"]
    jan = [r for r in recs if r["t"] >= agora - JANELA_H * 3600]

    # ---- sessões (pid) dentro da janela: deltas de contadores e histogramas ----
    sess = {}
    for r in jan:
        sess.setdefault(r.get("pid"), []).append(r)
    hist_acc, cont = {}, {}
    for pid, rs in sess.items():
        if len(rs) < 2:
            continue
        a, b = rs[0], rs[-1]
        for nome in b.get("h", {}):
            if nome in a.get("h", {}):
                hist_acc[nome] = soma_hist(hist_acc.get(nome), delta_hist(a["h"][nome], b["h"][nome]))
        for k in b.get("c", {}):
            cont[k] = cont.get(k, 0.0) + max(0.0, b["c"][k] - a.get("c", {}).get(k, 0.0))

    def lat(nome, esc=1.0):
        h = hist_acc.get(nome)
        if not h or h["n"] <= 0:
            return None
        q = lambda p: (None if quantil(h["b"], p) is None else round(quantil(h["b"], p) * esc, 3))
        return {"n": int(h["n"]), "media": round(h["s"] / h["n"] * esc, 3),
                "p50": q(.5), "p90": q(.9), "p99": q(.99)}

    lat_out = {"ttft": lat("time_to_first_token_seconds"),
               "itl_ms": lat("inter_token_latency_seconds", 1000.0),
               "fila": lat("queue_time_seconds"),
               "e2e": lat("e2e_request_latency_seconds"),
               "prompt_tok": lat("prompt_tokens_histogram"),
               "prefill_tok": lat("uncached_prompt_tokens_histogram")}

    # ---- totais do período ----
    req = cont.get("num_requests_total", 0.0)
    tot = {"requisicoes": int(req), "abortos": int(cont.get("num_aborted_requests_total", 0.0)),
           "prompt": int(cont.get("prompt_tokens_total", 0.0)),
           "cached": int(cont.get("cached_tokens_total", 0.0)),
           "gerados": int(cont.get("generation_tokens_total", 0.0)),
           "despejados": int(cont.get("evicted_tokens_total", 0.0))}
    tot["cache_pct"] = round(100.0 * tot["cached"] / tot["prompt"], 1) if tot["prompt"] else None
    tot["abortos_pct"] = round(100.0 * tot["abortos"] / req, 2) if req else None

    # ---- carga: em voo / fila por amostra ----
    carga_serie = [{"t": r["t"], "r": int(r["g"].get("num_running_reqs", 0)),
                    "q": int(r["g"].get("num_queue_reqs", 0))} for r in jan]
    carga = {"serie": carga_serie[-288:],
             "pico_voo": max((c["r"] for c in carga_serie), default=0),
             "pico_fila": max((c["q"] for c in carga_serie), default=0),
             "aceitacao": round(jan[-1]["g"].get("spec_accept_length", 0.0), 2),
             "passos": int(jan[-1]["g"].get("spec_num_steps", 0)),
             "kv_uso_pct": round(100.0 * jan[-1]["g"].get("token_usage", 0.0), 1)}

    # ---- VRAM livre por GPU ----
    vram_serie = []
    for r in jan:
        if r.get("gpu"):
            vram_serie.append({"t": r["t"], "livre": [g[1] - g[0] for g in r["gpu"]],
                               "w": [g[3] for g in r["gpu"]]})
    passo = max(1, len(vram_serie) // 144)
    vram = {"serie": vram_serie[::passo][-144:]}
    if vram_serie:
        vram["agora"] = vram_serie[-1]["livre"]
        # inclinação da GPU mais cheia nas últimas 6 h (regressão linear simples)
        rec6 = [v for v in vram_serie if v["t"] >= agora - 6 * 3600]
        if len(rec6) >= 3:
            xs = [v["t"] / 3600.0 for v in rec6]
            ys = [min(v["livre"]) for v in rec6]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            incl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
            vram["inclinacao_mib_h"] = round(incl, 1)
            minimo = min(vram["agora"])
            vram["min_agora"] = minimo
            vram["horas_ate_alarme"] = (round((minimo - ALARME_MIB) / -incl, 1)
                                        if incl < -1 and minimo > ALARME_MIB else None)

    # ---- throughput por lote (somas das linhas de decode) ----
    lote_acc = {}
    for r in jan:
        for k, (n, tps, acc) in r.get("bs", {}).items():
            a = lote_acc.setdefault(int(k), [0, 0.0, 0.0])
            a[0] += n
            a[1] += tps
            a[2] += acc
    lote = {str(k): {"n": v[0], "tps": round(v[1] / v[0], 1), "aceitacao": round(v[2] / v[0], 2)}
            for k, v in sorted(lote_acc.items()) if v[0] >= 20}

    # ---- prefill real (chunks cheios) ----
    pf = {"n": 0, "new": 0, "cached": 0, "n_full": 0, "tps_full": 0.0, "fila_max": 0}
    for r in jan:
        p = r.get("pf")
        if p:
            for k in ("n", "new", "cached", "n_full"):
                pf[k] += p.get(k, 0)
            pf["tps_full"] += p.get("tps_full", 0.0)
            pf["fila_max"] = max(pf["fila_max"], p.get("fila_max", 0))
    prefill = {"lotes": pf["n"], "chunks_cheios": pf["n_full"],
               "tps_chunk_cheio": round(pf["tps_full"] / pf["n_full"]) if pf["n_full"] else None,
               "cache_pct_log": round(100.0 * pf["cached"] / (pf["cached"] + pf["new"]), 1)
               if (pf["cached"] + pf["new"]) else None,
               "fila_max": pf["fila_max"]}

    print(json.dumps({"janela_h": JANELA_H, "amostras": len(jan), "desde": jan[0]["ts"],
                      "ate": jan[-1]["ts"], "sessoes": len(sess), "lat": lat_out, "tot": tot,
                      "carga": carga, "vram": vram, "lote": lote, "prefill": prefill},
                     ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
