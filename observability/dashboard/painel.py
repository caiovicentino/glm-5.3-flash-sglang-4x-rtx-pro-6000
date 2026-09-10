#!/usr/bin/env python3
"""
Painel local da máquina de inferência — tempo real, só na sua máquina.

    ./dashboard/painel.py            → abre em http://127.0.0.1:8099
    PORTA=9000 ./dashboard/painel.py

SEGURANÇA DO PRÓPRIO PAINEL (por que ele é assim):
- Escuta **só em 127.0.0.1**. Nunca 0.0.0.0 — senão o painel vira mais uma porta
  exposta, exatamente o problema que ele existe para vigiar.
- **Nenhum segredo chega ao navegador.** A chave da API e a da Vast ficam no
  processo Python; o HTML recebe só números e status.
- Sem CDN, sem fonte externa, sem telemetria: a página é autocontida.
- Só leitura. O painel nunca escreve na instância.

Dependências: nenhuma além da stdlib.
"""
import calendar
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTA = int(os.environ.get("PORTA", "8099"))

# Cada fonte tem custo diferente; caches separados evitam martelar SSH/API.
TTL = {"s24": 120, "metrics": 3, "gpu": 20, "est": 30, "vast": 120, "seg": 60,
       "atest": 60, "hist": 300}   # hist casa com o intervalo do coletor

_cache, _lock = {}, threading.RLock()   # RLock: coletar() segura e reentra
_serie = deque(maxlen=60)   # (t, generation_tokens) → tok/s ao vivo
_pico = {"conc": 0}   # maior concorrencia desde que o PAINEL subiu (zera ao reiniciar)

# Referencia medida com servidor vazio (docs/EVAL-2026-08-01): 2.717 tok/s
# agregados com 32 streams. E o teto conhecido, nao um limite teorico.
#
# ⚠️ 2026-08-26: este numero e do DeepSeek-V4-Flash no vLLM com max-num-seqs 64.
# A maquina agora roda GLM-5.3-Flash no SGLang com --max-running-requests 8, e
# NINGUEM mediu o agregado nesse regime. Enquanto nao medir, a barra "agregado"
# compara contra um teto de outro sistema. Nao tratar como valida.
REF_AGREGADO = 2717.0


def sh(cmd, timeout=25):
    try:
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                           timeout=timeout, cwd=RAIZ)
        return r.stdout.strip()
    except Exception:
        return ""


def instancia():
    """Descobre a instancia ativa (id, ip, portas, custo).

    Caminho normal: pergunta a API da Vast via lib.sh.

    Caminho SEM CHAVE DA VAST (para quem so tem SSH): definir CB_INST_IP,
    CB_INST_API e CB_INST_SSH no ambiente. A chave da Vast e de acesso TOTAL —
    destroi instancia e gasta credito — entao nao se compartilha com quem so
    precisa ver o painel. Com o override, o card de saldo mostra "—" e todo o
    resto funciona, porque tudo depois daqui usa SSH.
    """
    ip = os.environ.get("CB_INST_IP")
    if ip:
        return {"id": os.environ.get("CB_INST_ID", "—"), "ip": ip,
                "api": os.environ.get("CB_INST_API", "8000"),
                "ssh": os.environ.get("CB_INST_SSH", "22"),
                "dph": float(os.environ.get("CB_INST_DPH") or 0),
                "status": "running"}

    out = sh('source vast/lib.sh && instancia_ativa >/dev/null 2>&1 && '
             'echo "$INST_ID|$INST_IP|$INST_API_PORT|$INST_SSH_PORT|$INST_DPH|$INST_STATUS|$INST_PREPAGO|$INST_DESCONTO"')
    p = out.split("|")
    if len(p) < 6 or not p[0]:
        return None
    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0
    # prepago/desconto: instancia pre-paga (reserva). A Vast
    # guarda isso em `credit_balance` DA INSTANCIA (nao no credito da conta) e
    # cobra a taxa com desconto de reserva (`credit_discount`, 2,5% hoje). Sem ler
    # esses dois campos o painel seguia mostrando "2,2 dias" com a maquina paga.
    return {"id": p[0], "ip": p[1], "api": p[2], "ssh": p[3],
            "dph": float(p[4] or 0), "status": p[5],
            "prepago": _f(p[6]) if len(p) > 6 else 0.0,
            "desconto": _f(p[7]) if len(p) > 7 else 0.0}


def chave():
    # API key of the inference server: env var first, then a 600-mode file. Never in the page.
    return os.environ.get("API_KEY") or sh('cat ~/.api_key 2>/dev/null')


def metricas(inst):
    """Contadores do servidor. Agnóstico de engine.

    Em 2026-08-26 a máquina passou de vLLM para SGLang e esta função era
    `vllm:`-only: o painel exibiu números fósseis do DeepSeek por horas sem
    sinalizar nada. Ela agora aceita as DUAS famílias — o rollback existe.

    ⚠️ SGLang só serve /metrics com `--enable-metrics`. Sem a flag: 404.
    """
    try:
        with urllib.request.urlopen(f"http://{inst['ip']}:{inst['api']}/metrics", timeout=8) as r:
            corpo = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    # Agregacao por TIPO, nao cega. O SGLang reporta gauges como
    # `max_total_num_tokens` UMA VEZ POR tp_rank com o MESMO valor — somar da 4x
    # (medido: 15.143.168 em vez de 3.785.792). Contadores `_total` vem rotulados
    # por is_streaming e ai somar e o certo.
    bruto, engine = {}, None
    for l in corpo.splitlines():
        if l.startswith("#"):
            continue
        m = re.match(r"((?:vllm|sglang):[a-z_0-9]+)(?:\{[^}]*\})?\s+([0-9.e+-]+)", l)
        if m and not m.group(1).endswith("_bucket"):
            bruto.setdefault(m.group(1), []).append(float(m.group(2)))
            engine = engine or m.group(1).split(":")[0]
    if not engine:
        return None
    v = {k: (sum(xs) if k.endswith("_total") else max(xs))
         for k, xs in bruto.items()}
    g = lambda k: v.get(f"{engine}:{k}", 0.0)

    if engine == "sglang":
        req = g("num_requests_total")
        # ⚠️ NAO use sglang:cache_hit_rate — ele reporta 0.0 SEMPRE, mesmo com o
        # cache acertando 81%. O painel mostrou "0%" por dias por confiar nele.
        #
        # Calcular dos contadores. Semantica confirmada em 31/08 comparando as
        # metricas contra a soma de #new-token/#cached-token do log na MESMA
        # janela: prompt_tokens_total INCLUI os cacheados.
        #     log  81,2%  ·  cached/prompt  79,1% ✅  ·  cached/(cached+prompt) 44,2% ❌
        cach, prom = g("cached_tokens_total"), g("prompt_tokens_total")
        taxa_calc = (cach / prom) if prom else 0.0
        return {
            "engine": "sglang",
            "geradas": g("generation_tokens_total"), "prompt": g("prompt_tokens_total"),
            "req": req, "rodando": g("num_running_reqs"), "fila": g("num_queue_reqs"),
            "kv": g("token_usage") * 100, "preempcoes": g("num_retracted_reqs"),
            "cache": taxa_calc * 100,
            "mtp": g("spec_accept_rate") * (100 if g("spec_accept_rate") <= 1 else 1),
            "tok_passo": g("spec_accept_length"),
            "ctx": (g("prompt_tokens_total") / req) if req else 0,
            "pool_metrica": g("max_total_num_tokens") or None,
        }

    q, h = g("prefix_cache_queries_total"), g("prefix_cache_hits_total")
    d, a = g("spec_decode_num_drafts_total"), g("spec_decode_num_accepted_tokens_total")
    dt = g("spec_decode_num_draft_tokens_total")   # tokens, não passos
    req = g("request_success_total")
    return {
        "engine": "vllm",
        "geradas": g("generation_tokens_total"), "prompt": g("prompt_tokens_total"),
        "req": req, "rodando": g("num_requests_running"), "fila": g("num_requests_waiting"),
        "kv": g("kv_cache_usage_perc") * 100, "preempcoes": g("num_preemptions_total"),
        "cache": (h / q * 100) if q else 0,
        "mtp": (a / dt * 100) if dt else 0,
        "tok_passo": (1 + a / d) if d else 0,
        "ctx": (g("prompt_tokens_total") / req) if req else 0,
        "pool_metrica": None,
    }


def gpus(inst):
    out = sh(f'source vast/lib.sh && INST_IP={inst["ip"]} INST_SSH_PORT={inst["ssh"]} '
             f'gpu_ssh "nvidia-smi --query-gpu=index,utilization.gpu,power.draw,'
             f'memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"', 30)
    r = []
    for l in out.splitlines():
        p = [x.strip() for x in l.split(",")]
        if len(p) == 6 and p[0].isdigit():
            r.append({"i": int(p[0]), "util": float(p[1]), "w": float(p[2]),
                      "mem": float(p[3]) / 1024, "memtot": float(p[4]) / 1024, "t": float(p[5])})
    return r


def estabilidade(inst):
    """Uptime, restarts e os DOIS tetos de concorrencia — lidos da maquina.

    Agnostico de engine desde 2026-08-26. A logica de deteccao vive em
    /root/estado.sh NA MAQUINA, nao aqui: montar shell inline atravessando
    python → ssh → bash quebrava no escape aninhado, e quebrava CALADO
    (devolvia None e o painel mostrava travessao, sem erro nenhum).

    O script emite chaves normalizadas: eng, up, pool, seqs, restarts.
    """
    out = sh(f'source vast/lib.sh && INST_IP={inst["ip"]} INST_SSH_PORT={inst["ssh"]} '
             f'gpu_ssh /root/estado.sh', 40)
    d = {}
    for l in out.splitlines():
        l = l.strip()
        if "=" in l and not l.startswith("["):
            k, _, v = l.partition("=")
            d[k.strip()] = v.strip()
    def _i(k):
        try:
            return int(d.get(k, "").replace(",", ""))
        except (ValueError, AttributeError):
            return None
    return {"uptime": d.get("up") or "?", "restarts": d.get("restarts") or "?",
            "pool": _i("pool"), "max_seqs": _i("seqs"), "engine": d.get("eng")}


def saldo():
    out = sh('source vast/lib.sh && vast_api users/current/ | '
             'python3 -c "import sys,json;print(json.load(sys.stdin).get(\'credit\') or 0)"', 25)
    try:
        return float(out)
    except Exception:
        return None


def seguranca(inst, k):
    """Checagens ativas. Cada uma vira um item com ícone + rótulo — nunca cor sozinha."""
    base = f"http://{inst['ip']}:{inst['api']}"
    code = lambda u, h="": sh(f'curl -s -o /dev/null -w "%{{http_code}}" -m 8 {h} "{u}"', 15)
    sem = code(f"{base}/v1/models")
    errada = code(f"{base}/v1/models", '-H "Authorization: Bearer invalida"')
    met = code(f"{base}/metrics")
    tunel = sh('curl -s -o /dev/null -w "%{http_code}" -m 10 '
               'https://inference.culturabuilder.com/v1/models', 15)
    itens = [
        {"n": "Autenticação da API", "ok": sem == "401" and errada == "401",
         "d": f"401 sem chave e com chave inválida" if sem == "401" else f"HTTP {sem} sem chave"},
        {"n": "Prompts fora dos logs", "ok": True, "d": "verificado: 0 linhas com conteúdo"},
        {"n": "KV só em VRAM", "ok": True, "d": "sem cache de prefixo em disco"},
        {"n": "Telemetria do engine", "ok": True, "d": "desligada (do_not_track + env)"},
        {"n": "/metrics público", "ok": met != "200", "grau": "warning",
         "d": "aberto sem auth — fecha com o túnel" if met == "200" else "fechado"},
        {"n": "TLS no transporte", "ok": tunel == "401", "grau": "serious",
         "d": "túnel ativo" if tunel == "401" else f"pendente (hostname HTTP {tunel})"},
        {"n": "Chave rotacionada", "ok": False, "grau": "warning",
         "d": "pendente — está em 4 commits do histórico"},
    ]
    return itens


def atestacao(inst):
    out = sh(f'source vast/lib.sh && INST_IP={inst["ip"]} INST_SSH_PORT={inst["ssh"]} '
             f'gpu_ssh "pgrep -cf \'[c]ollect.py\'; tail -1 /root/attestation/usage.jsonl"', 30)
    ls = [x for x in out.splitlines() if x.strip()]
    viva = ls[0].strip() == "1" if ls else False
    modelo, alias, idade = "?", None, None
    for l in ls[1:]:
        try:
            d = json.loads(l)
            modelo = d.get("model_id") or "?"
            alias = d.get("served_alias")
            # timegm, não mktime: o timestamp é UTC e mktime assumiria hora local
            idade = time.time() - calendar.timegm(time.strptime(d["ts"], "%Y-%m-%dT%H:%M:%SZ"))
            break
        except Exception:
            pass
    return {"viva": viva, "modelo": modelo, "alias": alias,
            "idade_min": (idade / 60) if idade else None}


def historico(inst):
    """
    Série temporal REAL, vinda do coletor de atestação (roda 24/7, a cada 5 min).

    Os contadores do vLLM zeram a cada restart — por isso só somamos deltas DENTRO
    da mesma `session`. Atravessar a fronteira de sessão produziria um delta
    negativo gigante e um dia inteiro de lixo no gráfico.
    """
    # A agregacao roda NA MAQUINA (/root/agrega_uso.py). Antes esta funcao baixava
    # `tail -3000` do arquivo bruto: conforme o coletor grava 288 amostras/dia, a
    # janela deslizava e OS TOTAIS ENCOLHIAM -- em 24 h o "total de prompt" caiu de
    # 3,92 B para 3,63 B e o "coletando desde" andou 22 horas para frente. Agora o
    # historico e completo e trafegam ~8 KB, independente do tamanho do arquivo.
    # O `grep ^{` e obrigatorio: todo python3 daquela maquina imprime
    # "[hybrid_loader] import hook armed" em stdout antes da saida real.
    out = sh(f'source vast/lib.sh && INST_IP={inst["ip"]} INST_SSH_PORT={inst["ssh"]} '
             f'gpu_ssh "python3 /root/agrega_uso.py 2>/dev/null | grep \'^{{\'"', 60)
    ag = None
    for l in out.splitlines():
        l = l.strip()
        if l.startswith("{"):
            try:
                d = json.loads(l)
            except ValueError:
                continue
            if "dias" in d:
                ag = d
                break
    if ag:
        # Repassar TUDO que o agregador calcula. A versao anterior escolhia
        # quatro chaves na mao, entao `resumo` e `por_modelo` eram computados na
        # maquina e descartados aqui em silencio — o card ficava sem medias sem
        # que nada indicasse por que.
        return {k: ag[k] for k in
                ("dias", "serie", "desde", "totais", "resumo", "por_modelo",
                 "registros") if k in ag}
    recs = []
    if not recs:
        # NAO devolver {} aqui. O fresco() so preserva o valor anterior quando a
        # funcao retorna None; um dict vazio ele aceita como leitura BOA e
        # sobrescreve o historico bom por nada -- por TTL["hist"]=300s inteiros.
        # Era esse o bug: o grafico aparecia e sumia sozinho.
        #
        # `tail` sem nenhuma linha valida, com o arquivo existindo do outro lado
        # (o card de atestacao le o mesmo arquivo e mostra "ultimo registro ha N
        # min"), so acontece se o SSH falhou ou estourou o timeout. Instancia
        # realmente nova cai aqui tambem, e ai o cache nunca enche e a mensagem
        # "sem historico ainda" fica correta.
        print(f"[painel] historico: 0 registros — leitura FALHOU, mantendo cache "
              f"anterior ({time.strftime('%H:%M:%S')})", flush=True)
        return None
    dias, serie, prev = {}, [], None
    for r in recs:
        try:
            t = calendar.timegm(time.strptime(r["ts"], "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            continue
        cur = {"s": r.get("session"), "t": t, "g": r["generation_tokens"],
               "p": r["prompt_tokens"], "q": r["requests_success"]}
        if prev and prev["s"] == cur["s"]:
            dt, dg = cur["t"] - prev["t"], cur["g"] - prev["g"]
            dp, dq = cur["p"] - prev["p"], cur["q"] - prev["q"]
            if dt > 0 and dg >= 0 and dp >= 0:
                dia = time.strftime("%Y-%m-%d", time.gmtime(t))
                a = dias.setdefault(dia, {"g": 0, "p": 0, "q": 0})
                a["g"] += dg; a["p"] += dp; a["q"] += max(0, dq)
                serie.append({"t": t, "tps": dg / dt})
        prev = cur
    return {
        "dias": [{"dia": k, **v} for k, v in sorted(dias.items())][-14:],
        "serie": serie[-240:],
        "desde": recs[0]["ts"] if recs else None,
    }


def serie24(inst):
    """Últimas 24 h do coletor series.py (latência por percentil, fila, VRAM por GPU,
    throughput por lote, abortos). Agregado NA MÁQUINA por /root/agrega_series.py,
    pelo mesmo motivo do histórico: o que trafega são ~10 KB, nunca o arquivo bruto.
    Coletor e agregador vivem em attestation/ no repo e em /root/attestation na
    instância; o laço é /root/series_loop.sh, separado da atestação de propósito."""
    out = sh(f'source vast/lib.sh && INST_IP={inst["ip"]} INST_SSH_PORT={inst["ssh"]} '
             f'gpu_ssh "python3 /root/agrega_series.py 2>/dev/null | grep \'^{{\'"', 60)
    for l in out.splitlines():
        if l.startswith("{"):
            try:
                return json.loads(l)
            except ValueError:
                return None
    return None


def coletar():
    """Serializada: duas coletas ao mesmo tempo abririam SSH em dobro.

    O servidor e threading, entao dois GET /api simultaneos rodariam esta funcao
    em paralelo, cada um com sua propria rodada de SSH. Segurando o lock, o
    segundo espera e encontra o cache ja quente — sai na hora, sem tocar na
    instancia. Junto com a guarda de sobreposicao no JS, o numero de conexoes
    SSH deixa de crescer com o numero de abas abertas.
    """
    with _lock:
        return _coletar()


def _coletar():
    ag = time.time()
    with _lock:
        inst = _cache.get("inst")
        if not inst or ag - _cache.get("inst_t", 0) > 60:
            inst = instancia()
            _cache["inst"], _cache["inst_t"] = inst, ag
    if not inst:
        return {"erro": "nenhuma instância ativa"}

    d = dict(_cache.get("dados", {}))
    def fresco(nome, fn):
        if ag - _cache.get(nome + "_t", 0) > TTL[nome]:
            r = fn()
            if r is not None:
                _cache[nome], _cache[nome + "_t"] = r, ag
        return _cache.get(nome)

    m = fresco("metrics", lambda: metricas(inst))
    if m:
        _serie.append((ag, m["geradas"]))
    # janela curta (~5 amostras) para a taxa AGORA; a serie inteira alimenta o grafico
    tps = 0.0
    jan = list(_serie)[-6:]
    if len(jan) >= 2 and jan[-1][0] > jan[0][0]:
        tps = max(0.0, (jan[-1][1] - jan[0][1]) / (jan[-1][0] - jan[0][0]))
    if m:
        _pico["conc"] = max(_pico["conc"], int(m["rodando"]))

    k = _cache.get("chave") or chave()
    _cache["chave"] = k
    sal = fresco("vast", saldo)
    # Autonomia: se ha pre-pago na instancia, e ele que segura a maquina no ar;
    # o credito da conta so paga internet e extras. Taxa reservada = dph com desconto.
    dph_res = inst["dph"] * (1 - (inst.get("desconto") or 0)) if inst["dph"] else 0
    pre = inst.get("prepago") or 0
    if pre > 0 and dph_res:
        dias = pre / dph_res / 24
    else:
        dias = (sal / inst["dph"] / 24) if (sal and inst["dph"]) else None
    prepago = {"valor": pre, "dph_res": dph_res, "desconto": inst.get("desconto") or 0,
               "dias": (pre / dph_res / 24) if (pre > 0 and dph_res) else None}

    d.update({
        "ts": ag, "inst": {kk: vv for kk, vv in inst.items()},
        "m": m, "tps": tps,
        "gpu": fresco("gpu", lambda: gpus(inst)) or [],
        "est": fresco("est", lambda: estabilidade(inst)) or {},
        "seg": fresco("seg", lambda: seguranca(inst, k)) or [],
        "at": fresco("atest", lambda: atestacao(inst)) or {},
        "saldo": sal, "dias": dias, "prepago": prepago, "pico_conc": _pico["conc"], "ref": REF_AGREGADO,
        "hist": fresco("hist", lambda: historico(inst)) or {},
        # quando a ultima leitura BEM-SUCEDIDA aconteceu. Sem isto, dado velho
        # preservado pelo cache seria indistinguivel de dado ao vivo -- trocar um
        # grafico que some por um grafico que mente nao e conserto.
        "hist_t": _cache.get("hist_t", 0),
        "s24": fresco("s24", lambda: serie24(inst)) or {},
        "spark": [v for _, v in _serie],
    })
    _cache["dados"] = d
    return d


PAGINA = r"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Painel · máquina de inferência</title><style>
:root{color-scheme:dark;
 --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
 --grid:#2c2c2a; --ring:rgba(255,255,255,.10);
 --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
 --serie:#3987e5; --serie-track:rgba(57,135,229,.18);}
@media (prefers-color-scheme:light){:root:where(:not([data-theme="dark"])){color-scheme:light;
 --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
 --grid:#e1e0d9; --ring:rgba(11,11,11,.10); --serie:#2a78d6; --serie-track:rgba(42,120,214,.15);}}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
 font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;padding:20px}
header{display:flex;flex-direction:column;align-items:center;text-align:center;
 gap:12px;margin:6px 0 24px}
.marca{color:var(--ink);opacity:.94}
.logo{width:172px;height:auto;display:block;margin:0 auto}
h1{font-size:17px;font-weight:600;margin:0 0 3px;letter-spacing:-.01em}
h1 .alias{font-weight:500;color:var(--muted);font-size:13px;letter-spacing:0}
.sub{color:var(--muted);font-size:12px}
.grid{display:grid;gap:14px;align-items:start;
 grid-template-columns:repeat(auto-fit,minmax(270px,1fr));max-width:1500px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px}
.card h2{font-size:11px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;
 color:var(--muted);margin:0 0 14px}
.hero{font-size:52px;font-weight:600;line-height:1.05;letter-spacing:-.02em}
.hero .u{font-size:17px;font-weight:500;color:var(--ink2);letter-spacing:0}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:14px 10px}
.t .l{color:var(--muted);font-size:11px}
.t .v{font-size:22px;font-weight:600;letter-spacing:-.01em}
.t .v small{font-size:12px;font-weight:500;color:var(--ink2)}
.meter{margin:9px 0}
.meter .top{display:flex;justify-content:space-between;font-size:12px;color:var(--ink2);margin-bottom:4px}
.track{height:7px;border-radius:4px;background:var(--serie-track);overflow:hidden}
.fill{height:100%;border-radius:4px;transition:width .4s ease}
.sec{display:flex;gap:9px;align-items:flex-start;padding:8px 0;border-bottom:1px solid var(--grid)}
.sec:last-child{border-bottom:0}
.ico{font-size:13px;line-height:1.35;flex:0 0 15px}
.sec .n{font-weight:500}.sec .d{color:var(--muted);font-size:12px}
.rowv{display:flex;justify-content:space-between;padding:5px 0;font-size:13px}
.rowv span:first-child{color:var(--muted)}
.mono{font-variant-numeric:tabular-nums}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:1px}
svg{display:block;width:100%;height:44px;margin-top:8px}
.wide{grid-column:span 2}
@media(max-width:900px){.wide{grid-column:span 1}}
.chart{position:relative;width:100%}
.chart svg{height:auto}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);
 border:1px solid var(--ring);border-radius:7px;padding:6px 9px;font-size:12px;
 box-shadow:0 4px 14px rgba(0,0,0,.35);opacity:0;transition:opacity .1s;z-index:9}
.tip .k{color:var(--muted)}
.vazio{color:var(--muted);font-size:12px;padding:12px 0}
.foot{color:var(--muted);font-size:11px;margin-top:16px}
</style></head><body>
<header><div class="marca">__LOGO__</div>
<div><h1 id="modelo">carregando…</h1><div class="sub" id="cab"></div></div></header>
<div class="grid" id="g"></div>
<div class="foot" id="ft"></div>
<script>
const N=(n,d=0)=>n==null?"—":n.toLocaleString("pt-BR",{minimumFractionDigits:d,maximumFractionDigits:d});
const C=n=>n==null?"—":n>=1e9?(n/1e9).toFixed(2)+" B":n>=1e6?(n/1e6).toFixed(1)+" M":n>=1e3?(n/1e3).toFixed(1)+" k":N(n);
const esc=s=>String(s).replace(/[<>&]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));
// grau → cor + ícone. O ícone é obrigatório: cor nunca carrega sentido sozinha.
const G={good:["var(--good)","✓"],warning:["var(--warning)","▲"],
         serious:["var(--serious)","▲"],critical:["var(--critical)","✕"]};

function meter(rot,val,max,grau){
  // grau null = magnitude pura (hue de serie). Cor de status so para ESTADO real:
  // 100% de utilizacao e o estado desejado, nao um alerta.
  const p=Math.max(0,Math.min(100,val/max*100));
  const cor = grau ? G[grau][0] : "var(--serie)";
  return `<div class="meter"><div class="top"><span>${rot}</span>
    <span class="mono">${N(val,0)}${max===100?"%":""}</span></div>
    <div class="track"><div class="fill" style="width:${p}%;background:${cor}"></div></div></div>`;
}
function spark(v){
  if(!v||v.length<3)return"";
  const d=v.slice(1).map((x,i)=>x-v[i]).filter(x=>x>=0);
  if(d.length<2)return"";
  const mx=Math.max(...d,1),w=100,h=40;
  const pts=d.map((x,i)=>`${(i/(d.length-1)*w).toFixed(1)},${(h-x/mx*h).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-label="tokens por segundo">
    <polyline points="${pts}" fill="none" stroke="var(--serie)" stroke-width="2"
      vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>`;
}

const TIP=(()=>{const e=document.createElement("div");e.className="tip";
  document.body.appendChild(e);return e;})();
function mostraTip(ev,html){TIP.innerHTML=html;TIP.style.opacity=1;
  TIP.style.left=Math.min(ev.clientX+12,innerWidth-190)+"px";
  TIP.style.top=(ev.clientY-14)+"px";}
function escondeTip(){TIP.style.opacity=0;}

// Barras: magnitude por periodo discreto. Topo arredondado em 4px ancorado na
// linha de base, 2px de respiro entre barras.
function barras(el,dados){
  if(!dados||!dados.length){el.innerHTML='<div class="vazio">sem histórico ainda — o coletor grava a cada 5 min</div>';return;}
  // pt=26 (era 16) para caber o rotulo do topo do eixo Y. Sem ele o grafico
  // mostrava forma sem grandeza: dava para ver que 04/08 foi o maior dia, mas
  // nao QUANTO, nem de que unidade.
  const w=el.clientWidth||520,h=160,pb=24,pt=26;
  const mx=Math.max(...dados.map(d=>d.g),1);
  const bw=Math.max(6,(w-2*(dados.length-1))/dados.length-2);
  const passo=(w+2)/dados.length;
  let sv=`<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">`;
  // topo da escala: linha tracejada na altura da maior barra + valor com unidade
  sv+=`<line x1="0" y1="${pt}" x2="${w}" y2="${pt}" stroke="var(--grid)"
        stroke-width="1" stroke-dasharray="2 4"/>`;
  sv+=`<text x="0" y="${pt-7}" font-size="10" fill="var(--muted)">${C(mx)} tokens</text>`;
  sv+=`<line x1="0" y1="${h-pb}" x2="${w}" y2="${h-pb}" stroke="var(--baseline,#383835)" stroke-width="1"/>`;
  // Dia PARCIAL nao se compara com dia inteiro. O ultimo e hoje (ainda correndo)
  // e o primeiro comeca na hora em que o coletor subiu — desenhar os dois cheios
  // faz a barra menor parecer queda de uso quando e so o dia nao ter acabado.
  const hoje=new Date().toISOString().slice(0,10);
  const parcial=d=> d.dia===hoje || d.dia===(dados[0]||{}).dia;
  dados.forEach((d,i)=>{
    const bh=Math.max(2,(d.g/mx)*(h-pb-pt)),x=i*passo,y=h-pb-bh;
    const dia=d.dia.slice(8)+"/"+d.dia.slice(5,7);
    sv+=`<rect class="bar" data-i="${i}" x="${x}" y="${y}" width="${bw}" height="${bh}"
      rx="4" fill="var(--serie)"${parcial(d)?' opacity=".45"':""}/>`;
    if(dados.length<=8)
      sv+=`<text x="${x+bw/2}" y="${h-8}" text-anchor="middle" font-size="10"
        fill="var(--muted)">${dia}</text>`;
  });
  sv+="</svg>";el.innerHTML=sv;
  el.querySelectorAll(".bar").forEach(r=>{
    const d=dados[+r.dataset.i];
    r.addEventListener("mousemove",e=>mostraTip(e,
      `<b>${d.dia}</b>${parcial(d)?' <span class="k">· dia parcial</span>':""}<br>`+
      `<span class="k">gerados</span> ${C(d.g)}<br>`+
      `<span class="k">prompt</span> ${C(d.p)}<br><span class="k">requisições</span> ${C(d.q)}`));
    r.addEventListener("mouseleave",escondeTip);
  });
}

// Linha: serie continua no tempo. 2px, com crosshair no hover.
function linhas(el,serie){
  // VRAM livre por GPU: uma polilinha por GPU, mesma cor, tracejados diferentes
  // (legível em impressão e para daltônicos); a linha laranja é o alarme de 3 GB.
  if(!serie||serie.length<3){el.innerHTML='<div class="vazio">série curta — enche ao longo do dia</div>';return;}
  const w=el.clientWidth||520,h=160,pb=24,pt=26,ng=serie[0].livre.length;
  const mx=Math.max(3072,...serie.map(p=>Math.max(...p.livre)))*1.05;
  const t0=serie[0].t,t1=serie[serie.length-1].t,dt=Math.max(1,t1-t0);
  const X=p=>(p.t-t0)/dt*w, Y=v=>h-pb-(v/mx)*(h-pb-pt);
  const hr=q=>new Date(q*1000).toLocaleTimeString("pt-BR",{hour:"2-digit",minute:"2-digit"});
  const dia=q=>new Date(q*1000).toLocaleDateString("pt-BR",{day:"2-digit",month:"2-digit"});
  const rotX=q=>(dia(t0)!==dia(t1)?dia(q)+" ":"")+hr(q);
  const dash=["","6 3","2 3","8 3 2 3"];
  let sv=`<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">`;
  sv+=`<line x1="0" y1="${Y(3072).toFixed(1)}" x2="${w}" y2="${Y(3072).toFixed(1)}" stroke="var(--warning)" stroke-width="1" stroke-dasharray="2 4"/>`;
  sv+=`<line x1="0" y1="${h-pb}" x2="${w}" y2="${h-pb}" stroke="var(--baseline,#383835)" stroke-width="1"/>`;
  const u=serie[serie.length-1];
  for(let g=0;g<ng;g++){
    const pts=serie.map(p=>`${X(p).toFixed(1)},${Y(p.livre[g]).toFixed(1)}`).join(" ");
    sv+=`<polyline points="${pts}" fill="none" stroke="var(--serie)" stroke-width="${g?1.4:2}" stroke-dasharray="${dash[g%4]}" stroke-linejoin="round"/>`;
    sv+=`<text x="${w-2}" y="${(Y(u.livre[g])-3).toFixed(1)}" font-size="9" fill="var(--muted)" text-anchor="end">GPU${g}</text>`;
  }
  sv+=`<text x="0" y="${h-6}" font-size="10" fill="var(--muted)">${rotX(t0)}</text>`;
  sv+=`<text x="${w}" y="${h-6}" font-size="10" fill="var(--muted)" text-anchor="end">${rotX(t1)}</text>`;
  sv+=`<text x="0" y="${pt-5}" font-size="10" fill="var(--muted)">${N(mx/1024,1)} GB <tspan opacity=".7">(topo)</tspan> · tracejado laranja = alarme de 3 GB</text>`;
  sv+="</svg>";el.innerHTML=sv;
}
function linha(el,serie){
  if(!serie||serie.length<3){el.innerHTML='<div class="vazio">série curta — enche ao longo do dia</div>';return;}
  // mesma altura e mesmo respiro do topo que as barras — os dois cards ficam
  // lado a lado e alturas diferentes fazem o par parecer desalinhado
  const w=el.clientWidth||520,h=160,pb=24,pt=26;
  const mx=Math.max(...serie.map(p=>p.tps),1);
  const t0=serie[0].t,t1=serie[serie.length-1].t,dt=Math.max(1,t1-t0);
  const X=p=>(p.t-t0)/dt*w, Y=p=>h-pb-(p.tps/mx)*(h-pb-pt);
  const pts=serie.map(p=>`${X(p).toFixed(1)},${Y(p).toFixed(1)}`).join(" ");
  const hr=q=>{const d=new Date(q*1000);return d.toLocaleTimeString("pt-BR",{hour:"2-digit",minute:"2-digit"});};
  // A serie atravessa a meia-noite: "17:18 → 13:13" sozinho sugere que o tempo
  // anda para tras. Quando o inicio e o fim caem em dias diferentes, o rotulo
  // passa a carregar a data.
  const dia=q=>new Date(q*1000).toLocaleDateString("pt-BR",{day:"2-digit",month:"2-digit"});
  const cruzaDia = dia(t0)!==dia(t1);
  const rotX = q => (cruzaDia? dia(q)+" " : "") + hr(q);
  let sv=`<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">`;
  // topo da escala, para a linha ter grandeza e nao so forma
  sv+=`<line x1="0" y1="${pt}" x2="${w}" y2="${pt}" stroke="var(--grid)"
        stroke-width="1" stroke-dasharray="2 4"/>`;
  sv+=`<line x1="0" y1="${h-pb}" x2="${w}" y2="${h-pb}" stroke="var(--baseline,#383835)" stroke-width="1"/>`;
  sv+=`<polyline points="${pts}" fill="none" stroke="var(--serie)" stroke-width="2"
        stroke-linejoin="round" stroke-linecap="round"/>`;
  sv+=`<line id="cross" x1="0" y1="${pt}" x2="0" y2="${h-pb}" stroke="var(--muted)"
        stroke-width="1" opacity="0"/>`;
  sv+=`<text x="0" y="${h-6}" font-size="10" fill="var(--muted)">${rotX(t0)}</text>`;
  sv+=`<text x="${w}" y="${h-6}" font-size="10" fill="var(--muted)" text-anchor="end">${rotX(t1)}</text>`;
  sv+=`<text x="0" y="${pt-5}" font-size="10" fill="var(--muted)">${N(mx,0)} tok/s <tspan fill="var(--muted)" opacity=".7">(pico)</tspan></text>`;
  sv+="</svg>";el.innerHTML=sv;
  const svg=el.querySelector("svg"),cross=el.querySelector("#cross");
  svg.addEventListener("mousemove",e=>{
    const r=svg.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*w;
    let melhor=serie[0],dd=1e9;
    serie.forEach(p=>{const q=Math.abs(X(p)-x);if(q<dd){dd=q;melhor=p;}});
    cross.setAttribute("x1",X(melhor));cross.setAttribute("x2",X(melhor));
    cross.setAttribute("opacity","0.6");
    mostraTip(e,`<b>${hr(melhor.t)}</b><br><span class="k">vazão</span> ${N(melhor.tps,0)} tok/s`);
  });
  svg.addEventListener("mouseleave",()=>{cross.setAttribute("opacity","0");escondeTip();});
}

let _emVoo=false;
async function tick(){
  // Guarda de sobreposicao. O tick e de 3s mas uma coleta com cache frio leva
  // ~55s (varios SSH em serie). Sem esta guarda, ~18 ticks disparavam em cima
  // uns dos outros, cada um abrindo suas proprias conexoes SSH; a instancia nao
  // dava conta, parte falhava, e a falha apagava o historico. Era esse o
  // "aparece e some".
  if(_emVoo) return;
  _emVoo=true;
  let d; try{ const r=await fetch("/api"); if(!r.ok) return; d=await r.json(); }catch(e){ return; }
  finally{ _emVoo=false; }
  if(d.erro){ document.getElementById("cab").textContent="⚠️ "+d.erro; return; }
  const m=d.m||{}, dias=d.dias;
  const grauDias = dias==null?"warning":dias<2?"critical":dias<3?"serious":dias<5?"warning":"good";
  const at0=d.at||{};
  // o MODELO em destaque; o alias de serving vem atras, rotulado como alias —
  // era exatamente essa confusao que fazia a atestacao afirmar o modelo errado
  document.getElementById("modelo").textContent = at0.modelo||"—";
  document.getElementById("cab").innerHTML =
    `<span class="dot" style="background:${d.inst.status==="running"?"var(--good)":"var(--critical)"}"></span>`+
    `instância ${esc(d.inst.id)} · ${esc(d.inst.status)}`;

  const cards=[];
  // HERO: exatamente um por painel — o número que pode nos custar a máquina
  cards.push(`<div class="card"><h2>Autonomia do saldo</h2>
    <div class="hero" style="color:${G[grauDias][0]}">${dias==null?"—":dias.toFixed(1)}<span class="u"> dias</span></div>
    ${(d.prepago&&d.prepago.valor>0)?`
    <div class="rowv"><span>pré-pago na instância</span><span class="mono">$${N(d.prepago.valor,2)}</span></div>
    <div class="rowv"><span>termina em</span><span class="mono">${(()=>{const t=new Date(Date.now()+d.prepago.dias*86400e3);const dd=Math.floor(d.prepago.dias),hh=Math.round((d.prepago.dias-dd)*24);return t.toLocaleDateString("pt-BR",{day:"2-digit",month:"2-digit"})+" "+t.toLocaleTimeString("pt-BR",{hour:"2-digit",minute:"2-digit"})+" · "+dd+" d "+hh+" h";})()}</span></div>
    <div class="rowv"><span>taxa reservada</span><span class="mono">$${N(d.prepago.dph_res*24,0)}/dia <span class="sub">(−${N(d.prepago.desconto*100,1)}%)</span></span></div>
    <div class="rowv"><span>crédito da conta</span><span class="mono">$${N(d.saldo,2)} <span class="sub">(internet e extras)</span></span></div>`:`
    <div class="rowv"><span>saldo</span><span class="mono">$${N(d.saldo,2)}</span></div>
    <div class="rowv"><span>queima</span><span class="mono">$${N(d.inst.dph*24,0)}/dia</span></div>`}
    ${dias!=null&&dias<3?`<div class="sec" style="border:0;padding-top:10px">
      <span class="ico" style="color:${G[grauDias][0]}">${G[grauDias][1]}</span>
      <div><div class="n">${(d.prepago&&d.prepago.valor>0)?"Renovar o pré-pago":"Recarregar agora"}</div>
      <div class="d">sem crédito a Vast destrói a instância sem aviso</div></div></div>`:""}</div>`);

  const est=d.est||{}, maxs=est.max_seqs||32, pool=est.pool;
  const porSessao = m.rodando>0 ? d.tps/m.rodando : 0;
  // capacidade real = pool de KV dividido pelo contexto medio observado; o teto de
  // sequencias e o outro limite. Quem morde primeiro e o menor dos dois.
  const cabem = (pool&&m.ctx) ? pool/m.ctx : null;
  const ocup = maxs ? m.rodando/maxs*100 : 0;
  cards.push(`<div class="card"><h2>Concorrência e vazão</h2><div class="tiles">
    <div class="t"><div class="l">sessões agora</div><div class="v mono">${N(m.rodando)}</div></div>
    <div class="t"><div class="l">pico no painel</div><div class="v mono">${N(d.pico_conc)}</div></div>
    <div class="t"><div class="l">agregado</div><div class="v mono">${N(d.tps,0)}<small> tok/s</small></div></div>
    <div class="t"><div class="l">por sessão</div><div class="v mono">${N(porSessao,0)}<small> tok/s</small></div></div>
    <div class="t"><div class="l">na fila</div><div class="v mono" style="color:${
      m.fila>0?"var(--warning)":"var(--ink)"}">${N(m.fila)}</div></div>
    <div class="t"><div class="l">tokens/passo</div><div class="v mono">${N(m.tok_passo,2)}</div></div>
    </div>${spark(d.spark)}
    ${meter(`sessões · teto ${maxs}`,m.rodando,maxs,ocup>85?"warning":null)}
    ${meter(`agregado · referência ${C(d.ref)} t/s`,Math.min(d.tps,d.ref),d.ref,null)}
    ${meter("KV cache em uso",m.kv||0,100,(m.kv||0)>80?"critical":(m.kv||0)>60?"warning":"good")}
    <div class="rowv" style="border-top:1px solid var(--grid);margin-top:8px;padding-top:8px">
      <span>cabem no pool</span><span class="mono">${cabem?`~${N(cabem,0)} sessões`:"—"}</span></div>
    <div class="rowv"><span>pool de KV</span><span class="mono">${pool?C(pool)+" tokens":"—"}</span></div>
    </div>`);

  // Este card le os contadores VIVOS do /metrics, que zeram a cada restart do
  // vLLM. O "Acumulado por dia" vem do historico da atestacao, que soma por
  // sessao e ATRAVESSA restart. Os dois totais divergem de proposito (11,1 M
  // contra 14,6 M em 06/08) — sem o rotulo, parece contradicao no painel.
  cards.push(`<div class="card"><h2>Tráfego acumulado</h2>
    <div class="sub" style="margin:-10px 0 12px">desde o boot do ${m.engine==="sglang"?"SGLang":"vLLM"} · zera a cada restart</div>
    <div class="tiles">
    <div class="t"><div class="l">requisições</div><div class="v mono">${C(m.req)}</div></div>
    <div class="t"><div class="l">tokens gerados</div><div class="v mono">${C(m.geradas)}</div></div>
    <div class="t"><div class="l">tokens de prompt</div><div class="v mono">${C(m.prompt)}</div></div>
    <div class="t"><div class="l">contexto médio</div><div class="v mono">${C(m.ctx)}</div></div>
    <div class="t"><div class="l">cache de prefixo</div><div class="v mono">${N(m.cache,1)}<small>%</small></div></div>
    <div class="t"><div class="l">aceitação MTP</div><div class="v mono">${N(m.mtp,0)}<small>%</small></div></div>
    </div></div>`);

  // temperatura carrega a severidade; utilizacao e so magnitude
  const gp=(d.gpu||[]).map(g=>meter(`GPU${g.i} · ${N(g.w,0)} W · ${N(g.t,0)}°C`,g.util,100,
      g.t>85?"critical":g.t>78?"warning":null)).join("");
  cards.push(`<div class="card"><h2>GPUs</h2>${gp||'<div class="d">sem leitura</div>'}
    <div class="rowv"><span>no ar há</span><span class="mono">${esc(d.est.uptime||"—")}</span></div>
    <div class="rowv"><span>preempções</span><span class="mono" style="color:${
      m.preempcoes>0?"var(--critical)":"var(--ink)"}">${N(m.preempcoes)}</span></div></div>`);

  const at=d.at||{};
  const itens=(d.seg||[]).map(s=>{
    const g=s.ok?"good":(s.grau||"critical");
    return `<div class="sec"><span class="ico" style="color:${G[g][0]}">${G[g][1]}</span>
      <div><div class="n">${esc(s.n)}</div><div class="d">${esc(s.d)}</div></div></div>`;}).join("");
  const ga=at.viva?"good":"critical";
  cards.push(`<div class="card"><h2>Segurança</h2>${itens}
    <div class="sec"><span class="ico" style="color:${G[ga][0]}">${G[ga][1]}</span>
    <div><div class="n">Atestação coletando</div><div class="d">${
      at.viva?`modelo ${esc(at.modelo)} · último registro há ${N(at.idade_min,0)} min`
             :"coletor parado"}</div></div></div></div>`);

  const H=d.hist||{};
  // O grafico mostrava a forma de cada dia mas nunca o TOTAL do periodo, que e
  // O numero que se leva para a reuniao usa `totais`, que o agregador calcula
  // sobre o histórico INTEIRO. A soma das barras nao serve: o grafico mostra 14
  // dias, entao o total encolheria um dia por vez -- foi assim que o "total de
  // prompt" caiu de 3,92 B para 3,63 B em 24 h. O rodape diz explicitamente que
  // as barras sao uma janela do total, para o numero continuar explicavel.
  const DS=H.dias||[];
  const TT=H.totais||null;
  const somaG=TT?TT.g:DS.reduce((a,x)=>a+(x.g||0),0);
  const somaP=TT?TT.p:DS.reduce((a,x)=>a+(x.p||0),0);
  const somaQ=TT?TT.q:DS.reduce((a,x)=>a+(x.q||0),0);
  const nDias=TT?TT.dias:DS.length;
  const RS=H.resumo||null;      // medias, melhor/pior, tendencia de 7 dias
  const PM=H.por_modelo||null;  // recorte DeepSeek x GLM-5.3
  cards.push(`<div class="card wide"><h2>Acumulado por dia</h2>
    <div class="sub" style="margin:-10px 0 12px">tokens gerados por dia · passe o mouse para prompt e requisições</div>
    <div class="chart" id="ch-dia"></div>
    ${DS.length?`<div class="sub" style="border-top:1px solid var(--grid);margin-top:12px;padding-top:10px">
      total desde o início da coleta — atravessa restart do servidor, por isso é maior que o card de tráfego · o gráfico mostra o histórico COMPLETO (${DS.length} dias)</div>
    <div class="tiles" style="margin-top:8px">
      <div class="t"><div class="l">total gerado</div><div class="v mono">${C(somaG)}</div></div>
      <div class="t"><div class="l">total de prompt</div><div class="v mono">${C(somaP)}</div></div>
      <div class="t"><div class="l">requisições</div><div class="v mono">${C(somaQ)}</div></div>
      <div class="t"><div class="l">dias</div><div class="v mono">${nDias}</div></div>
    </div>

    ${RS?`<div class="sub" style="border-top:1px solid var(--grid);margin-top:14px;padding-top:10px">
      média por dia — sobre os ${TT.ativos} dias com movimento, não os ${TT.dias} corridos</div>
    <div class="tiles" style="margin-top:8px">
      <div class="t"><div class="l">gerados/dia</div><div class="v mono">${C(RS.media_g)}</div></div>
      <div class="t"><div class="l">prompt/dia</div><div class="v mono">${C(RS.media_p)}</div></div>
      <div class="t"><div class="l">requisições/dia</div><div class="v mono">${C(RS.media_q)}</div></div>
      <div class="t"><div class="l">contexto médio</div><div class="v mono">${C(RS.ctx_medio)}</div></div>
    </div>
    <div class="rowv" style="margin-top:10px"><span>razão prompt : geração</span>
      <span class="mono">${RS.razao}:1</span></div>
    ${RS.melhor?`<div class="rowv"><span>melhor dia</span>
      <span class="mono">${esc(RS.melhor.dia)} · ${C(RS.melhor.g)}</span></div>`:""}
    ${RS.pior?`<div class="rowv"><span>pior dia (com movimento)</span>
      <span class="mono">${esc(RS.pior.dia)} · ${C(RS.pior.g)}</span></div>`:""}
    ${RS.media7_g?`<div class="rowv"><span>últimos 7 dias</span>
      <span class="mono">${C(RS.media7_g)}/dia · ${RS.media7_g>RS.media_g?"+":""}${N((RS.media7_g/RS.media_g-1)*100,0)}% vs a média</span></div>`:""}`:""}

    ${PM&&Object.keys(PM).length>1?`<div class="sub" style="border-top:1px solid var(--grid);margin-top:14px;padding-top:10px">
      por modelo — o histórico atravessa a troca de 26/08, então os dois regimes aparecem separados</div>
    <table style="width:100%;margin-top:8px;border-collapse:collapse;font-size:12px">
      <tr style="color:var(--dim)"><td>modelo</td><td class="mono" style="text-align:right">dias</td>
        <td class="mono" style="text-align:right">gerados/dia</td>
        <td class="mono" style="text-align:right">ctx médio</td>
        <td class="mono" style="text-align:right">razão</td></tr>
      ${Object.entries(PM).sort((a,b)=>b[1].dias-a[1].dias).map(([k,v])=>
        `<tr><td>${esc(k)}</td><td class="mono" style="text-align:right">${v.dias}</td>
         <td class="mono" style="text-align:right">${C(v.media_g)}</td>
         <td class="mono" style="text-align:right">${C(v.ctx)}</td>
         <td class="mono" style="text-align:right">${v.razao}:1</td></tr>`).join("")}
    </table>`:""}`:""}
    ${H.desde?`<div class="rowv" style="border-top:1px solid var(--grid);margin-top:10px;padding-top:8px">
      <span>coletando desde</span><span class="mono">${esc(H.desde.replace("T"," ").replace("Z"," UTC"))}</span></div>`:""}
    ${(()=>{ // idade da ultima leitura BEM-SUCEDIDA do histórico
       if(!d.hist_t) return "";
       const idade=d.ts-d.hist_t;
       if(idade < 900) return "";   // 3x o TTL de 300s — dentro do normal
       const hh=new Date(d.hist_t*1000).toLocaleTimeString("pt-BR");
       return `<div class="rowv" style="color:${G.warning[0]}">
         <span style="color:inherit">⚠️ histórico desatualizado</span>
         <span class="mono">última leitura ${hh}</span></div>`;
     })()}</div>`);
  cards.push(`<div class="card wide"><h2>Vazão ao longo do tempo</h2>
    <div class="sub" style="margin:-10px 0 12px">tokens gerados por segundo, amostrado a cada tick do painel</div>
    <div class="chart" id="ch-vaz"></div></div>`);

  // ---- série de 24 h: coletor series.py na máquina, agregado por agrega_series.py ----
  const S=d.s24||{};
  if(S.lat){
    const L=S.lat, T=S.tot||{}, CG=S.carga||{}, PF=S.prefill||{}, V=S.vram||{}, LT=S.lote||{};
    const f=(x,dc=1)=>x==null?"—":N(x,dc);
    // janela real da série: sem isto, uma tabela de 24 h parece parada quando só os
    // lotes raros (16, 17) não ganharam amostra nova — a janela desliza, a linha não muda
    const hj=(z)=>{ if(!z) return "—"; const t=new Date(z); return t.toLocaleDateString("pt-BR",{day:"2-digit",month:"2-digit"})+" "+t.toLocaleTimeString("pt-BR",{hour:"2-digit",minute:"2-digit"}); };
    const janela=`janela ${hj(S.desde)} → ${hj(S.ate)} · ${S.amostras||0} amostras de 5 min · o resumo é refeito a cada 2 min`;
    const td=(x)=>`<td class="mono" style="text-align:right">${x}</td>`;
    const linhasLat=[["primeiro token (s)",L.ttft,1],["espera na fila (s)",L.fila,2],["entre tokens (ms)",L.itl_ms,1],
      ["ponta a ponta (s)",L.e2e,1],["tokens prefilados por req.",L.prefill_tok,0],["tokens do prompt",L.prompt_tok,0]];
    cards.push(`<div class="card wide"><h2>Latência e fila · ${S.janela_h} h</h2>
      <div class="sub" style="margin:-10px 0 12px">percentis dos histogramas do servidor · ${C(T.requisicoes||0)} requisições em ${S.sessoes} sessão(ões) · "p99 ≤ 80" quer dizer que 1% esperou mais que isso<br>${janela}</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <tr style="color:var(--dim)"><td>medida</td>${td("p50")}${td("p90")}${td("p99")}${td("média")}</tr>
        ${linhasLat.map(([n,o,dc])=>{o=o||{};return `<tr><td>${n}</td>${td(f(o.p50,dc))}${td(f(o.p90,dc))}${td(f(o.p99,dc))}${td(f(o.media,dc))}</tr>`;}).join("")}
      </table>
      <div class="tiles" style="margin-top:10px">
        <div class="t"><div class="l">abortos</div><div class="v mono">${T.abortos??"—"} <span class="sub">(${f(T.abortos_pct,2)}%)</span></div></div>
        <div class="t"><div class="l">cache no período</div><div class="v mono">${f(T.cache_pct,1)}%</div></div>
        <div class="t"><div class="l">pico em voo / fila</div><div class="v mono">${CG.pico_voo??"—"} / ${CG.pico_fila??"—"}</div></div>
        <div class="t"><div class="l">prefill real (chunks cheios)</div><div class="v mono">${PF.tps_chunk_cheio?C(PF.tps_chunk_cheio)+" tok/s":"—"}</div></div>
      </div>
      <div class="rowv" style="margin-top:8px"><span>tokens despejados do cache</span><span class="mono">${C(T.despejados||0)}</span></div>
      <div class="rowv"><span>aceitação do MTP agora · passos</span><span class="mono">${CG.aceitacao??"—"} · ${CG.passos??"—"}</span></div></div>`);
    cards.push(`<div class="card wide"><h2>Memória livre por GPU · ${S.janela_h} h</h2>
      <div class="sub" style="margin:-10px 0 12px">MiB livres a cada 5 min · o cache do alocador só volta quando o servidor fica ocioso, e só na GPU0</div>
      <div class="chart" id="ch-vram"></div>
      ${V.agora?`<div class="tiles" style="margin-top:8px">${V.agora.map((x,i)=>`<div class="t"><div class="l">GPU${i} agora</div><div class="v mono">${C(x)} MiB</div></div>`).join("")}</div>
      <div class="rowv" style="margin-top:8px"><span>inclinação (GPU mais cheia, 6 h)</span><span class="mono">${V.inclinacao_mib_h==null?"—":(V.inclinacao_mib_h>0?"+":"")+N(V.inclinacao_mib_h,0)+" MiB/h"}</span></div>
      <div class="rowv"><span>alarme de 3 GB</span><span class="mono">${V.horas_ate_alarme==null?"sem previsão de bater":"em ~"+N(V.horas_ate_alarme,1)+" h"}</span></div>`:""}</div>`);
    if(Object.keys(LT).length) cards.push(`<div class="card"><h2>Throughput por lote · ${S.janela_h} h</h2>
      <div class="sub" style="margin:-10px 0 12px">média das linhas de decode do log, por requisições em voo · é o que julga cada config no dia seguinte<br>${janela}</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <tr style="color:var(--dim)"><td>lote</td>${td("amostras")}${td("aceitação")}${td("tok/s")}</tr>
        ${Object.entries(LT).sort((a,b)=>+a[0]-+b[0]).map(([k,v])=>`<tr><td>${k}</td>${td(C(v.n))}${td(N(v.aceitacao,2))}${td(N(v.tps,0))}</tr>`).join("")}
      </table></div>`);
  } else {
    cards.push(`<div class="card"><h2>Série de 24 h</h2><div class="d">coletor ainda sem amostras suficientes — enche ao longo do dia</div></div>`);
  }
  document.getElementById("g").innerHTML=cards.join("");
  // desenhar só depois do innerHTML: as funções medem clientWidth
  barras(document.getElementById("ch-dia"),H.dias);
  linha(document.getElementById("ch-vaz"),H.serie);
  if(S.vram&&document.getElementById("ch-vram")) linhas(document.getElementById("ch-vram"),S.vram.serie);
  document.getElementById("ft").textContent =
    "atualizado "+new Date(d.ts*1000).toLocaleTimeString("pt-BR")+
    " · só 127.0.0.1 · nenhum segredo chega ao navegador";
}
tick(); setInterval(tick,3000);
</script></body></html>"""


_LOGO = ""
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.svg")) as _f:
        _LOGO = _f.read()
except OSError:
    pass
PAGINA = PAGINA.replace("__LOGO__", _LOGO)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # sem log de acesso: o painel não guarda rastro de uso

    def do_GET(self):
        if self.path.startswith("/api"):
            # Responde do ULTIMO snapshot, nunca coleta aqui. Em 04/09 cada handshake
            # SSH passou a levar 4,6 s, o /api bloqueava 1m52 esperando seis fontes em
            # serie e o navegador desistia antes: "carregando..." para sempre. A coleta
            # roda numa thread de fundo (ver __main__) respeitando os TTLs de cada fonte.
            dados = _cache.get("dados")
            if not dados:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Retry-After", "3")
                self.end_headers()
                self.wfile.write(b'{"aguardando": true}')
                return
            corpo = json.dumps(dados).encode()
            tipo = "application/json"
        else:
            corpo, tipo = PAGINA.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(corpo)


if __name__ == "__main__":
    # 127.0.0.1 explícito: o painel jamais deve virar mais uma porta exposta.
    # ThreadingHTTPServer, nao HTTPServer: o servidor de uma requisicao por vez
    # atendia uma coleta de ~55s enquanto as outras esperavam na fila de escuta
    # (padrao 5) e eram DERRUBADAS — apareciam como HTTP 000 em 7,8s. A coleta em
    # si continua serializada pelo lock de coletar(); o que muda e que a conexao
    # deixa de morrer enquanto espera.
    def _atualizador():
        # Coleta continua em segundo plano; cada fonte so e refeita quando seu TTL
        # vence (fresco()), entao o custo por volta e o das fontes vencidas.
        while True:
            try:
                coletar()
            except Exception as e:  # a thread nunca morre por uma fonte quebrada
                print("coleta falhou:", str(e)[:120])
            time.sleep(2)
    threading.Thread(target=_atualizador, daemon=True, name="coleta").start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORTA), H)
    srv.daemon_threads = True
    print(f"painel em http://127.0.0.1:{PORTA}  (ctrl-c para sair)")
    print("escutando só em loopback · segredos ficam no processo, não no navegador")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nencerrado")
