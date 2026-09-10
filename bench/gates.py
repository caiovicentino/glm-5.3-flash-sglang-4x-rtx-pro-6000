#!/usr/bin/env python3
"""Quality gates for GLM-5.3-Flash. Run AFTER the server is up; prints PASSA/FALHA per gate and a verdict.
Nothing here changes the server. Critical gates: models endpoint, long pt-BR generation, opencode-shaped
request with thinking, reading text in an image. Known non-regressions on this build: gate G reads
usage.prompt_tokens_details.cached_tokens, which the build does not populate (check the log instead)."""
import base64, io, json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import os
BASE = os.environ.get("BASE", "http://localhost:8000/v1")
KEY = os.environ.get("API_KEY") or open("/root/.api_key").read().strip()
MODELO = os.environ.get("MODEL", "glm-5.3-flash")
R = {}

def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as h:
        return json.loads(h.read())

def chat(msgs, **kw):
    b = {"model": MODELO, "messages": msgs, **kw}
    return post("/chat/completions", b)

def gate(nome, ok, detalhe=""):
    R[nome] = ok
    print(f"  [{'PASSA' if ok else 'FALHA'}] {nome}" + (f" — {detalhe}" if detalhe else ""), flush=True)

# ───────────────────────── A · sobe e responde ─────────────────────────
print("\n══ A · SERVIDOR ══", flush=True)
try:
    m = json.loads(urllib.request.urlopen(
        urllib.request.Request(BASE + "/models", headers={"Authorization": f"Bearer {KEY}"}),
        timeout=60).read())
    nomes = [x["id"] for x in m.get("data", [])]
    gate("A · /v1/models responde", MODELO in nomes, f"nomes={nomes}")
except Exception as e:
    gate("A · /v1/models responde", False, f"{type(e).__name__}: {e}"); sys.exit(1)

# ───────────────── B · geracao longa pt-BR (detecta KV corrompido) ─────────────────
print("\n══ B · GERACAO LONGA pt-BR (o portao critico) ══", flush=True)
t0 = time.time()
try:
    r = chat([{"role": "user", "content":
        "Escreva um texto tecnico DETALHADO em portugues do Brasil, com pelo menos 2500 palavras, "
        "explicando como funciona a inferencia de modelos de linguagem do tipo Mixture-of-Experts: "
        "roteamento de tokens, atencao esparsa, cache de prefixo, decodificacao especulativa e o "
        "gargalo de banda de memoria. Use subtitulos e paragrafos longos."}],
        max_tokens=8000, temperature=0.7)  # 4000 was too small: the model plans the whole essay inside its reasoning
    txt = r["choices"][0]["message"].get("content") or ""
    ntok = r.get("usage", {}).get("completion_tokens", 0)
    dt = time.time() - t0
    # heuristicas de saida corrompida
    palavras = txt.split()
    rep_max = 0
    if palavras:
        from collections import Counter
        rep_max = Counter(palavras).most_common(1)[0][1] / len(palavras)
    acentos = len(re.findall(r"[áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]", txt))
    lixo = len(re.findall(r"[^\w\s\.,;:!?()\-–—\"'/%\n\[\]áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]", txt))
    ok = (ntok > 800 and rep_max < 0.12 and acentos > 20 and lixo / max(len(txt), 1) < 0.02)
    gate("B · geracao longa coerente", ok,
         f"{ntok} tokens · {dt:.0f}s · rep_max={rep_max:.1%} · acentos={acentos} · lixo={lixo/max(len(txt),1):.2%}")
    print("  ── amostra (300 primeiros chars) ──", flush=True)
    print("  " + (txt[:300].replace("\n", "\n  ") if txt else "<VAZIO>"), flush=True)
except Exception as e:
    gate("B · geracao longa coerente", False, f"{type(e).__name__}: {str(e)[:120]}")

# ───────────────── D · a requisicao INTEIRA do opencode ─────────────────
print("\n══ D · FORMA EXATA DO OPENCODE ══", flush=True)
try:
    r = chat([{"role": "system", "content": "Voce e um assistente de programacao."},
              {"role": "user", "content": "Escreva uma funcao Python que inverte uma string. So o codigo."}],
             max_tokens=800, temperature=0.6,
             chat_template_kwargs={"enable_thinking": True})
    msg = r["choices"][0]["message"]
    c = msg.get("content") or ""
    rc = msg.get("reasoning_content") or ""
    gate("D · opencode (enable_thinking)", bool(c.strip()),
         f"content={len(c)} chars · reasoning={len(rc)} chars")
    if not c.strip():
        print("  ⚠️ content VAZIO — same symptom as the enable_thinking parser bug", flush=True)
except Exception as e:
    gate("D · opencode (enable_thinking)", False, f"{type(e).__name__}: {str(e)[:120]}")

# ───────────────── E · tool calling (parser glm47) ─────────────────
print("\n══ E · TOOL CALLING ══", flush=True)
try:
    tools = [{"type": "function", "function": {
        "name": "consultar_gpu", "description": "Consulta metricas de uma GPU",
        "parameters": {"type": "object", "properties": {
            "indice": {"type": "integer", "description": "indice da GPU"}},
            "required": ["indice"]}}}]
    r = chat([{"role": "user", "content": "Consulte as metricas da GPU numero 2."}],
             tools=tools, tool_choice="auto", max_tokens=500, temperature=0.3)
    tc = r["choices"][0]["message"].get("tool_calls") or []
    ok = False
    if tc:
        try:
            args = json.loads(tc[0]["function"]["arguments"])
            ok = tc[0]["function"]["name"] == "consultar_gpu" and args.get("indice") == 2
        except Exception: ok = False
    gate("E · tool call valida", ok, f"{len(tc)} chamada(s)" + (f" · {tc[0]['function']}" if tc else ""))
except Exception as e:
    gate("E · tool call valida", False, f"{type(e).__name__}: {str(e)[:120]}")

# ───────────────── H · VISAO (a razao de tudo isto) ─────────────────
print("\n══ H · VISAO ══", flush=True)
def png_b64(desenha, w=560, h=560):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (w, h), "white"); d = ImageDraw.Draw(im)
    try: f = ImageFont.load_default(size=64)
    except TypeError: f = ImageFont.load_default()
    desenha(d, f, w, h)
    b = io.BytesIO(); im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

def ver(img, pergunta):
    r = chat([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": img}},
        {"type": "text", "text": pergunta}]}], max_tokens=300, temperature=0.1)
    return (r["choices"][0]["message"].get("content") or "").strip()

try:
    i1 = png_b64(lambda d, f, w, h: d.rectangle([0, 0, w, h], fill="#c81e1e"))
    a1 = ver(i1, "De que cor e esta imagem? Responda so a cor.")
    ok1 = any(x in a1.lower() for x in ["vermelh", "red"])
    gate("H1 · cor solida", ok1, f"resposta: {a1[:70]!r}")

    CODE = "CB4271"
    i2 = png_b64(lambda d, f, w, h: d.text((40, h//2 - 40), CODE, fill="black", font=f))
    a2 = ver(i2, "Que texto aparece nesta imagem? Responda apenas o texto.")
    ok2 = CODE.lower().replace(" ", "") in a2.lower().replace(" ", "")
    gate("H2 · le texto na imagem", ok2, f"esperado {CODE!r} · obteve {a2[:70]!r}")

    def duas(d, f, w, h):
        d.rectangle([30, 30, w//2 - 20, h//2 - 20], fill="#1e5ac8")
        d.ellipse([w//2 + 20, h//2 + 20, w - 30, h - 30], fill="#1eb04a")
    i3 = png_b64(duas)
    a3 = ver(i3, "Descreva as formas e suas cores e posicoes.")
    al = a3.lower()
    ok3 = (any(x in al for x in ["quadrad", "retangul", "square"]) and
           any(x in al for x in ["circul", "circle", "redond"]))
    gate("H3 · formas e posicoes", ok3, f"resposta: {a3[:90]!r}")
except Exception as e:
    gate("H · visao", False, f"{type(e).__name__}: {str(e)[:150]}")

# ───────────────── G · cache de prefixo ─────────────────
print("\n══ G · CACHE DE PREFIXO ══", flush=True)
try:
    base = "Contexto fixo para teste de cache. " * 900
    m = [{"role": "user", "content": base + "\n\nResponda apenas: OK"}]
    r1 = chat(m, max_tokens=10, temperature=0)
    r2 = chat(m, max_tokens=10, temperature=0)
    def cached(r):
        u = r.get("usage", {})
        d = u.get("prompt_tokens_details") or {}
        return d.get("cached_tokens", u.get("cached_tokens", 0)) or 0
    c1, c2 = cached(r1), cached(r2)
    pt = r2.get("usage", {}).get("prompt_tokens", 1)
    gate("G · cache de prefixo ativo", c2 > c1 or c2 / max(pt, 1) > 0.5,
         f"1a={c1} · 2a={c2} de {pt} ({c2/max(pt,1):.0%})")
except Exception as e:
    gate("G · cache de prefixo ativo", False, f"{type(e).__name__}: {str(e)[:120]}")

# ───────────────── F · concorrencia ─────────────────
print("\n══ F · CONCORRENCIA ══", flush=True)
try:
    def um(i):
        t = time.time()
        r = chat([{"role": "user", "content": f"Conte de 1 ate 60. Sessao {i}."}],
                 max_tokens=400, temperature=0.5)
        return r.get("usage", {}).get("completion_tokens", 0), time.time() - t
    N = 8
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N) as ex:
        res = list(ex.map(um, range(N)))
    wall = time.time() - t0
    tot = sum(x[0] for x in res)
    gate("F · 8 streams concorrentes", tot > 0,
         f"{tot} tokens em {wall:.1f}s = {tot/wall:.0f} tok/s agregado")
except Exception as e:
    gate("F · 8 streams concorrentes", False, f"{type(e).__name__}: {str(e)[:120]}")

# ───────────────── veredito ─────────────────
print("\n" + "=" * 62, flush=True)
crit = ["A · /v1/models responde", "B · geracao longa coerente",
        "D · opencode (enable_thinking)", "H2 · le texto na imagem"]
falhas = [k for k, v in R.items() if not v]
print(f"  {sum(R.values())}/{len(R)} portoes passaram", flush=True)
if falhas: print("  falharam: " + " · ".join(falhas), flush=True)
cf = [k for k in crit if k in R and not R[k]]
print(f"\n  VEREDITO: {'❌ ROLLBACK — portao critico falhou: ' + ', '.join(cf) if cf else '✅ pode ficar'}", flush=True)
