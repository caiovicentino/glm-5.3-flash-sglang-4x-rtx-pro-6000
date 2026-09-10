#!/usr/bin/env python3
"""Agrega o usage.jsonl NA MÁQUINA e devolve só o resumo.

Por que existe: o painel lia `tail -3000` do arquivo bruto por SSH. Conforme o
coletor grava (288 amostras/dia), a janela deslizava e **os totais encolhiam** —
em 24 h o "total de prompt" caiu de 3,92 B para 3,63 B e o "coletando desde"
andou 22 horas para frente. Agregando aqui, o histórico é completo e o que
trafega é sempre alguns KB, independente do tamanho do arquivo.

Saída: JSON com {dias, serie, desde, registros}.
"""
import calendar
import json
import sys
import time

ARQ = "/root/attestation/usage.jsonl"


def chave_sessao(r):
    """Identifica a sessão do vLLM.

    Os contadores zeram a cada restart, então só somamos deltas DENTRO da mesma
    sessão. ⚠️ Nem todo registro tem o campo `session`; sem o fallback para
    `session_pid`, dois registros sem `session` comparariam None == None e o
    delta gigante de um restart entraria no gráfico como um dia de lixo.
    """
    return r.get("session") or r.get("session_pid")


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
                if "generation_tokens" in r and "ts" in r:
                    recs.append(r)
    except OSError as e:
        print(json.dumps({"erro": str(e)}))
        return 1
    if not recs:
        print(json.dumps({"erro": "sem registros"}))
        return 1

    recs.sort(key=lambda r: r["ts"])
    dias, serie, prev = {}, [], None
    for r in recs:
        try:
            t = calendar.timegm(time.strptime(r["ts"], "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            continue
        cur = {"s": chave_sessao(r), "t": t, "g": r["generation_tokens"],
               "p": r["prompt_tokens"], "q": r["requests_success"],
               "m": r.get("model_id") or "?"}
        if prev and prev["s"] == cur["s"]:
            dt, dg = cur["t"] - prev["t"], cur["g"] - prev["g"]
            dp, dq = cur["p"] - prev["p"], cur["q"] - prev["q"]
            if dt > 0 and dg >= 0 and dp >= 0:
                dia = time.strftime("%Y-%m-%d", time.gmtime(t))
                a = dias.setdefault(dia, {"g": 0, "p": 0, "q": 0, "_m": {}})
                a["g"] += dg
                a["p"] += dp
                a["q"] += max(0, dq)
                # O modelo mudou em 26/08 (DeepSeek -> GLM-5.3). Guardar qual
                # servia cada dia permite comparar os dois regimes no painel.
                a["_m"][cur["m"]] = a["_m"].get(cur["m"], 0) + 1
                serie.append({"t": t, "tps": round(dg / dt, 2)})
        prev = cur

    def curto(nome):
        """Nome do modelo em forma legivel no grafico."""
        n = (nome or "?").replace(" NVFP4 (weight-only)", "").replace("-0731", "")
        return n.split("(")[0].strip()[:18] or "?"

    todos = []
    for k, v in sorted(dias.items()):
        ms = v.pop("_m", {})
        dom = max(ms.items(), key=lambda x: x[1])[0] if ms else "?"
        todos.append({"dia": k, **v, "m": curto(dom)})
    # ⚠️ `totais` é a VIDA INTEIRA, não a soma do gráfico. O gráfico mostra 14
    # dias; se o card somasse essa lista, o total encolheria um dia por vez —
    # que é o mesmo bug do `tail -3000`, só que mais lento de perceber.
    tg = sum(d["g"] for d in todos)
    tp = sum(d["p"] for d in todos)
    tq = sum(d["q"] for d in todos)
    nd = len(todos) or 1

    # Dias com movimento: a media sobre dias mortos mente para baixo.
    ativos = [d for d in todos if d["g"] > 0]
    na = len(ativos) or 1

    def med(campo, base):
        return round(sum(d[campo] for d in base) / (len(base) or 1))

    # Recorte por modelo: o historico atravessa a troca de 26/08.
    porm = {}
    for d in todos:
        a = porm.setdefault(d["m"], {"g": 0, "p": 0, "q": 0, "dias": 0})
        a["g"] += d["g"]; a["p"] += d["p"]; a["q"] += d["q"]; a["dias"] += 1
    for k, a in porm.items():
        n = a["dias"] or 1
        a["media_g"] = round(a["g"] / n)
        a["media_p"] = round(a["p"] / n)
        a["ctx"] = round(a["p"] / a["q"]) if a["q"] else 0
        a["razao"] = round(a["p"] / a["g"], 1) if a["g"] else 0

    melhor = max(todos, key=lambda d: d["g"]) if todos else None
    pior = min(ativos, key=lambda d: d["g"]) if ativos else None
    ult7 = ativos[-7:]

    print(json.dumps({
        # HISTORICO COMPLETO. Antes eram 14 dias; o painel truncava o que a
        # maquina ja tinha calculado, e ninguem via a serie inteira.
        "dias": todos,
        "totais": {"g": tg, "p": tp, "q": tq, "dias": nd, "ativos": na},
        "resumo": {
            "media_g": med("g", ativos), "media_p": med("p", ativos),
            "media_q": med("q", ativos),
            "media_g_corridos": round(tg / nd),
            "razao": round(tp / tg, 1) if tg else 0,
            "ctx_medio": round(tp / tq) if tq else 0,
            "melhor": melhor, "pior": pior,
            "media7_g": med("g", ult7), "media7_p": med("p", ult7),
        },
        "por_modelo": porm,
        "serie": serie[-240:],
        "desde": recs[0]["ts"],
        "registros": len(recs),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
