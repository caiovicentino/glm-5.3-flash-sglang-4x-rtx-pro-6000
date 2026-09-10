#!/usr/bin/env python3
"""Bateria A/B para comparar UMA mudança de configuração no servidor.

Escrita uma vez e rodada duas vezes (antes/depois). Se eu escrevesse dois
testes, estaria comparando dois testes, não duas configurações.

Cuidados que a bateria toma, e por quê:

  * TODO prompt leva um sal único por execução. Sem isso o cache de prefixo
    entrega 96-99% de acerto e a medição vira medição do cache, não do motor.
    Já aconteceu: uma rodada "fria" da arena deu 99,8% de acerto.
  * Os tokens vêm de `usage` via `stream_options.include_usage`, nunca de
    contagem de chunks do SSE. Contar chunks já subestimou em 4x (235 contra
    1.020 reais).
  * O decode é dividido pela janela DEPOIS do TTFT, e o prefill pela janela
    ATÉ o TTFT. Dividir pelo relógio de parede inteiro subestima o prefill em
    4x em rodadas rasas.
  * Código e prosa são medidos separados: a aceitação do DSpark é 69% em
    código e 25% em prosa, então uma média entre os dois esconde o efeito.

Uso:
    python3 evals/bench_ab.py <rotulo>      # grava evals/ab_<rotulo>.json
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROTULO = sys.argv[1] if len(sys.argv) > 1 else "sem-rotulo"
# Permite apontar a mesma bateria para outro modelo servido (ex.: teste do Qwen3.8),
# mantendo os quatro estágios idênticos para a comparação ser honesta.
MODELO = os.environ.get("BENCH_MODEL", "glm-5.3-flash")
SAL = f"{ROTULO}-{int(time.time())}"

CODIGO = ("Implemente em Python um cache LRU com TTL: classe completa, type hints, "
          "docstrings e uma bateria de testes com pytest cobrindo expiração, "
          "evicção por capacidade e concorrência.")
PROSA = ("Escreva um ensaio em português sobre por que sistemas distribuídos falham "
         "de formas que sistemas locais não falham, com exemplos concretos.")


def alvo():
    return os.environ.get("BASE", "http://127.0.0.1:8000")

def chave():
    return os.environ.get("API_KEY") or open(os.path.expanduser("~/.api_key")).read().strip()

BASE, KEY = alvo(), chave()
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def enchimento(n_tokens, semente):
    """Texto único e volumoso. 3,18 chars/token medido via /tokenize."""
    alvo_chars = int(n_tokens * 3.18)
    base = (f"Registro {semente}-{{i}}: o processo {semente} anotou latência de "
            "{i} ms, fila de {j} itens e taxa de acerto de 0,{i} no turno {i}. ")
    partes, i = [], 0
    while sum(map(len, partes)) < alvo_chars:
        partes.append(base.format(i=i, j=(i * 7) % 101))
        i += 1
    return "".join(partes)[:alvo_chars]


def chamada(prompt, max_tokens, rotulo):
    corpo = {"model": MODELO, "stream": True, "max_tokens": max_tokens,
             "temperature": 0.6, "stream_options": {"include_usage": True},
             "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time()
    ttft, uso, fim = None, None, None
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(corpo).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=1800) as r:
        for ln in r:
            s = ln.decode("utf-8", "replace").strip()
            if not s.startswith("data: ") or "[DONE]" in s:
                continue
            try:
                o = json.loads(s[6:])
            except Exception:
                continue
            if o.get("usage"):
                uso = o["usage"]
            for ch in (o.get("choices") or []):
                d = ch.get("delta") or {}
                if (d.get("content") or d.get("reasoning")
                        or d.get("reasoning_content")) and ttft is None:
                    ttft = time.time() - t0
                if ch.get("finish_reason"):
                    fim = ch["finish_reason"]
    dt = time.time() - t0
    entrada = (uso or {}).get("prompt_tokens", 0)
    saida = (uso or {}).get("completion_tokens", 0)
    return {"rotulo": rotulo, "ttft": ttft or 0.0, "total": dt,
            "entrada": entrada, "saida": saida, "finish": fim,
            # prefill pela janela ATÉ o ttft; decode pela janela DEPOIS dele
            "prefill_tps": entrada / max(ttft or 1e-9, 1e-9),
            "decode_tps": saida / max(dt - (ttft or 0), 1e-9)}


res = {"rotulo": ROTULO, "sal": SAL, "base": BASE, "quando": time.strftime("%FT%TZ", time.gmtime())}
print(f"── bateria A/B · rótulo={ROTULO} · modelo={MODELO} · alvo={BASE}")

# ---------------------------------------------------------------- aquecimento
print("  aquecendo (descartado)...", flush=True)
chamada(f"[{SAL}-warm] diga ok", 16, "warm")

# ------------------------------------------------ 1. stream único, por conteúdo
print("\n── 1. stream único (3 repetições, prompt único a cada uma) ──", flush=True)
res["single"] = {}
for nome, p in (("codigo", CODIGO), ("prosa", PROSA)):
    linhas = [chamada(f"[{SAL}-{nome}-{i}] {p}", 1200, nome) for i in range(3)]
    d = [x["decode_tps"] for x in linhas]
    t = [x["ttft"] for x in linhas]
    res["single"][nome] = {"decode_tps": d, "ttft": t,
                           "decode_med": statistics.mean(d), "ttft_med": statistics.mean(t)}
    print(f"  {nome:<7} decode {statistics.mean(d):>6.1f} tok/s "
          f"(min {min(d):.0f} max {max(d):.0f}) · TTFT {statistics.mean(t):.2f}s", flush=True)

# --------------------------------------------------- 2. prefill a 95k, 1 stream
print("\n── 2. prefill a ~95k, stream único (2 repetições) ──", flush=True)
linhas = []
for i in range(2):
    p = enchimento(95000, f"{SAL}-p{i}") + "\n\nResuma em uma frase o que você acabou de ler."
    linhas.append(chamada(p, 64, f"prefill95k-{i}"))
    print(f"  rep {i}: {linhas[-1]['entrada']:,} tok entrada · TTFT {linhas[-1]['ttft']:.2f}s "
          f"· prefill {linhas[-1]['prefill_tps']:,.0f} tok/s", flush=True)
res["prefill_95k"] = linhas

# ------------------------------------------------- 3. concorrência 8 x 95k único
print("\n── 3. concorrência: 8 streams a ~95k, prompts todos diferentes ──", flush=True)
prompts = [enchimento(95000, f"{SAL}-c{i}") + "\n\nEscreva três parágrafos analisando o padrão acima."
           for i in range(8)]
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    conc = list(ex.map(lambda a: chamada(a[1], 400, f"conc-{a[0]}"), enumerate(prompts)))
janela = time.time() - t0
ent = sum(x["entrada"] for x in conc)
sai = sum(x["saida"] for x in conc)
ttfts = sorted(x["ttft"] for x in conc)
res["concorrencia_8x95k"] = {
    "janela_s": janela, "entrada_total": ent, "saida_total": sai,
    "prefill_agregado": ent / max(ttfts[-1], 1e-9),
    "decode_agregado": sai / max(janela - ttfts[0], 1e-9),
    "ttft_min": ttfts[0], "ttft_mediana": statistics.median(ttfts), "ttft_max": ttfts[-1]}
c = res["concorrencia_8x95k"]
print(f"  entrada {ent:,} tok · saída {sai:,} tok em {janela:.1f}s")
print(f"  prefill agregado {c['prefill_agregado']:,.0f} tok/s · decode agregado {c['decode_agregado']:,.0f} tok/s")
print(f"  TTFT  min {c['ttft_min']:.1f}s · mediana {c['ttft_mediana']:.1f}s · max {c['ttft_max']:.1f}s")

# ------------------------------- 4. decode concorrente (faixa média de mensagem)
# Os estágios 1-3 são pesados em PREFILL (1 M de entrada contra 3.200 de saída) e
# medem mal a faixa de 86 KB a 6 MB, que é onde vive o allreduce de decode com
# concorrência: N_seqs x 6 posições x hidden 7168 x 2 bytes. Com 16 streams dá
# ~1,4 MB — exatamente o intervalo que VLLM_PCIE_DMA_MIN_BYTES controla.
print("\n── 4. decode concorrente: 16 streams, prompt curto, saída longa ──", flush=True)
curtos = [f"[{SAL}-d{i}] Escreva um texto longo e detalhado sobre o tema {i}: "
          "arquitetura de sistemas distribuídos, com exemplos." for i in range(16)]
t0 = time.time()
with ThreadPoolExecutor(max_workers=16) as ex:
    dec = list(ex.map(lambda a: chamada(a[1], 1200, f"dec-{a[0]}"), enumerate(curtos)))
janela_d = time.time() - t0
sai_d = sum(x["saida"] for x in dec)
tt_d = sorted(x["ttft"] for x in dec)
por_stream = statistics.mean(x["decode_tps"] for x in dec)
res["decode_16x"] = {"janela_s": janela_d, "saida_total": sai_d,
                     "decode_agregado": sai_d / max(janela_d - tt_d[0], 1e-9),
                     "decode_por_stream": por_stream,
                     "ttft_mediana": statistics.median(tt_d)}
d4 = res["decode_16x"]
print(f"  {sai_d:,} tok de saída em {janela_d:.1f}s")
print(f"  decode AGREGADO {d4['decode_agregado']:,.0f} tok/s · por stream {por_stream:.1f} tok/s"
      f" · TTFT mediana {d4['ttft_mediana']:.2f}s")

destino = os.path.join(RAIZ, "evals", f"ab_{ROTULO}.json")
with open(destino, "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"\n  gravado em {destino}")
