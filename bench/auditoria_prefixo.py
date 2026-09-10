"""Auditoria de estabilidade de prefixo: o que quebra o prefix cache na prática.

Com o teto de prefill saturado por hardware em ~9.300 tok/s (ver
docs/CAPACIDADE-MEDIDA-2026-08-08.md), servir mais exige FAZER MENOS trabalho —
e a única alavanca grande que sobra é a taxa de acerto do cache de prefixo,
hoje em 82,4%.

O cache do vLLM casa por BLOCO (256 tokens) a partir da posição ZERO. Qualquer
variação no começo do prompt invalida tudo que vem depois. Este script mede,
por requisição, quanto do prompt foi servido do cache em cenários que imitam o
que cada cliente nosso faz.

Como mede: `prompt_tokens_cached_total` / `prompt_tokens_total` do /metrics, em
delta apertado ao redor de cada requisição. O vLLM NÃO devolve isso no usage
(prompt_tokens_details vem nulo), então delta é o único caminho — e por isso há
guarda: se mais de uma requisição fechar na janela, a amostra é descartada,
porque seria tráfego de produção somado ao nosso.

Uso: python3 evals/auditoria_prefixo.py HOST:PORTA CHAVE
"""
import os, json
import sys
import time
import urllib.request

EP, K = sys.argv[1], sys.argv[2]
H = {"Authorization": f"Bearer {K}", "Content-Type": "application/json"}


def metricas():
    with urllib.request.urlopen(f"http://{EP}/metrics", timeout=15) as r:
        txt = r.read().decode("utf-8", "replace")
    g = {}
    for ln in txt.splitlines():
        if ln.startswith("vllm:"):
            try:
                g[ln.split("{")[0].split(" ")[0]] = float(ln.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    v = lambda k: g.get("vllm:" + k, 0.0)
    return (v("prompt_tokens_total"), v("prompt_tokens_cached_total"),
            v("e2e_request_latency_seconds_count"))


def envia(msgs, tools=None):
    """Devolve (prompt_tokens, cached_tokens, contaminada)."""
    a = metricas()
    corpo = {"model": os.environ.get("MODEL", "glm-5.3-flash"), "messages": msgs, "max_tokens": 4,
             "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        corpo["tools"] = tools
        corpo["tool_choice"] = "auto"
    req = urllib.request.Request(f"http://{EP}/v1/chat/completions",
                                 data=json.dumps(corpo).encode(), headers=H)
    urllib.request.urlopen(req, timeout=600).read()
    time.sleep(1.2)                      # os contadores fecham depois da resposta
    b = metricas()
    return b[0] - a[0], b[1] - a[1], (b[2] - a[2]) != 1


def linha(rot, msgs, tools=None):
    p, c, sujo = envia(msgs, tools)
    pct = 100 * c / p if p else 0
    marca = "  ⚠️ contaminada" if sujo else ""
    barra = "█" * round(pct / 5)
    print(f"  {rot:<34} {p:>7,.0f} tok · cache {pct:>5.1f}% {barra}{marca}".replace(",", "."))
    return pct


# Corpo grande o bastante para cobrir vários blocos de 256 tokens.
def bloco(n, semente=0):
    return (f"[registro {semente}] " + "A praça São Paulo apresentou variação de "
            "audiência no período, com desvio ante a média histórica e impacto em "
            "alcance, frequência e afinidade do painel. ") * n


SYS = "Você é um assistente de análise de audiência. Seja objetivo."
FERR = [{"type": "function", "function": {"name": n, "description": f"Ferramenta {n}.",
         "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}}
        for n in ("consultar_painel", "exportar_csv", "comparar_pracas")]

print(f"{'cenário':<36} {'prompt':>9}   acerto do cache")
print("-" * 74)

print("\n── controle ──")
base = [{"role": "system", "content": SYS},
        {"role": "user", "content": bloco(60)}]
linha("1ª vez (esperado: baixo)", base)
linha("repetido idêntico", base)

print("\n── conversa que só CRESCE (o caso bom) ──")
conv = [{"role": "system", "content": SYS}]
for t in range(1, 5):
    conv = conv + [{"role": "user", "content": bloco(40, t)},
                   {"role": "assistant", "content": f"Analisado o lote {t}."}]
    linha(f"turno {t} (histórico cresce)", conv[:-1] or conv)

print("\n── janela deslizante, como o chat.html ──")
# chat.html manda history.slice(-12): passando de 12, a mensagem MAIS ANTIGA cai
# e o prompt inteiro desloca — o prefixo quebra desde a posição zero.
longa = [{"role": "system", "content": SYS}]
for t in range(1, 9):
    longa += [{"role": "user", "content": bloco(30, 100 + t)},
              {"role": "assistant", "content": f"ok {t}"}]
linha("janela cheia, 1ª vez", longa)
linha("mesma janela, repetida", longa)
linha("janela DESLIZOU (dropa a 1ª)", longa[1:] + [{"role": "user", "content": bloco(30, 200)}])

print("\n── o que mais quebra ──")
linha("ferramentas na ordem A,B,C", base, FERR)
linha("mesmas ferramentas, ordem C,B,A", base, FERR[::-1])
ts = [{"role": "system", "content": SYS + f" Data e hora: {time.strftime('%H:%M:%S')}."},
      {"role": "user", "content": bloco(60)}]
linha("system com timestamp, 1ª", ts)
time.sleep(1.1)
ts2 = [{"role": "system", "content": SYS + f" Data e hora: {time.strftime('%H:%M:%S')}."},
       {"role": "user", "content": bloco(60)}]
linha("system com timestamp, 2ª", ts2)

print("\nLeitura: acerto alto = o servidor reaproveitou o prefixo e pagou pouco prefill.")
print("Acerto que DESABA entre duas linhas quase iguais aponta o que quebra o cache.")
