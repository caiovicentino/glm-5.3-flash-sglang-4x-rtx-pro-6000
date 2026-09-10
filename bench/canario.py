#!/usr/bin/env python3
# /// script
# dependencies = ["httpx"]
# ///
"""
Canário diário — detecta mudança que ninguém anunciou.

  ./canario.py IP:PORTA CHAVE            # compara com o baseline
  ./canario.py IP:PORTA CHAVE --gravar   # define o baseline atual

POR QUE

Com temperature=0 a saída deve ser byte a byte idêntica entre execuções. Se mudar
sem que ninguém tenha mexido no serve, alguma coisa mudou por baixo: kernel,
driver, pesos corrompidos no disco, GPU degradando, ou — o caso que já nos
aconteceu — uma flag diferente da que achávamos estar rodando.

Este eval teria pegado o overlay errado. Aquele bug custou 60% da velocidade e
NÃO alterava a qualidade da resposta, então nenhum eval de qualidade acusaria.
Mas alterava o tier de precisão dos pesos, e portanto os logits — o hash mudaria.

LIMITAÇÃO CONHECIDA (medida em 2026-07-31)

O vLLM NÃO é determinístico bit a bit nem com temperature=0: o resultado depende
de como as requisições foram agrupadas em batch, e esta máquina tem tráfego real
de usuários. Rodamos o eval de tool calling duas vezes seguidas e obtivemos 4/5 e
depois 5/5, com prompts idênticos.

Por isso o canário gera cada saída DUAS vezes e só acusa mudança quando as duas
concordam entre si e discordam do baseline. Se as duas discordarem entre si, o
prompt é marcado como instável e não conta — comparar hash único daria alarme
falso todo dia.

Custa centavos e roda em ~3 minutos. Bom para cron diário.
"""
import argparse
import asyncio
import hashlib
import json
import pathlib
import sys

import httpx

import os
MODEL = os.environ.get("MODEL", "glm-5.3-flash")
BASELINE = pathlib.Path(__file__).parent / "canario_baseline.json"

# Prompts curtos, determinísticos e de domínios diferentes — se só um mudar, é
# ruído de amostragem; se vários mudarem, mudou o modelo/kernel.
PROMPTS = [
    ("aritmetica", "Calcule 17 * 23 + 199. Responda apenas o número."),
    ("codigo", "Escreva uma função Python chamada dobro que recebe x e devolve x*2. "
               "Apenas o código, sem explicação."),
    ("ptbr", "Complete em uma frase: A privacidade de dados importa porque"),
    ("json", 'Devolva apenas este JSON preenchido: {"pais": "?", "capital": "?"} para o Brasil.'),
    ("listagem", "Liste 5 estados brasileiros em ordem alfabética, separados por vírgula."),
]


async def gerar(client, base, key, prompt):
    r = await client.post(
        f"{base}/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 200, "temperature": 0, "top_p": 1, "seed": 42,
              "chat_template_kwargs": {"enable_thinking": False}},
        timeout=600)
    r.raise_for_status()
    d = r.json()
    txt = d["choices"][0]["message"].get("content") or ""
    return txt, d["usage"]["completion_tokens"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("endpoint")
    ap.add_argument("key")
    ap.add_argument("--gravar", action="store_true")
    a = ap.parse_args()
    base = f"http://{a.endpoint}"

    atual, instaveis = {}, []
    async with httpx.AsyncClient() as client:
        for nome, p in PROMPTS:
            # duas gerações: separa mudança real de não-determinismo de batching
            t1, n1 = await gerar(client, base, a.key, p)
            t2, _ = await gerar(client, base, a.key, p)
            h1 = hashlib.sha256(t1.encode()).hexdigest()[:16]
            h2 = hashlib.sha256(t2.encode()).hexdigest()[:16]
            if h1 != h2:
                instaveis.append(nome)
            atual[nome] = {"hash": h1, "estavel": h1 == h2,
                           "tokens": n1, "inicio": t1.strip()[:60]}

    if a.gravar:
        BASELINE.write_text(json.dumps(atual, ensure_ascii=False, indent=2))
        print(f"baseline gravado em {BASELINE.name} ({len(atual)} prompts)")
        for k, v in atual.items():
            print(f"  {k:<10} {v['hash']}  {v['inicio']!r}")
        return

    if not BASELINE.exists():
        print("SEM BASELINE. Rode com --gravar numa configuração que você confia.")
        sys.exit(2)

    ref = json.loads(BASELINE.read_text())
    mudou = []
    for nome, _ in PROMPTS:
        r, v = ref.get(nome), atual[nome]
        if not r:
            print(f"  ?  {nome}: sem baseline"); continue
        if not v["estavel"]:
            print(f"  ~  {nome:<10} instável entre duas gerações — não conta")
            continue
        if r["hash"] == v["hash"]:
            print(f"  ✅ {nome:<10} idêntico ({v['hash']})")
        else:
            mudou.append(nome)
            print(f"  ❌ {nome:<10} MUDOU  {r['hash']} → {v['hash']}")
            print(f"        antes: {r['inicio']!r}")
            print(f"        agora: {v['inicio']!r}")

    print("\n" + "=" * 60)
    if instaveis:
        print(f"~ instáveis (batching, não é regressão): {', '.join(instaveis)}")
    if not mudou:
        print("✅ Nada mudou. Modelo, kernels e pesos consistentes com o baseline.")
        return
    print(f"⚠️  {len(mudou)} de {len(PROMPTS)} prompts mudaram: {', '.join(mudou)}")
    print("   Com temperature=0 isso NÃO deveria acontecer sozinho. Investigue:")
    print("   • o serve_cmd.sh é o mesmo? (flags, overlay, envs)")
    print("   • houve troca de máquina/driver?")
    print("   • rode /gpu-doctor — o md5 do tier é o do autor (2309011aac57)?")
    print("   Se a mudança for intencional (nova config aprovada), regrave com --gravar.")
    sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
