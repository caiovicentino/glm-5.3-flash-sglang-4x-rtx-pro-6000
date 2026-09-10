#!/usr/bin/env python3
"""Batch-1 pt-BR prose speed: 8 x 600 tokens, thinking off, sequential. Run twice (2nd = warm) on an idle server."""
import json, time, urllib.request
import os
KEY = os.environ.get("API_KEY") or open("/root/.api_key").read().strip()
MODEL = os.environ.get("MODEL", "glm-5.3-flash")
def gera(txt, n):
    b = {"model":MODEL,"max_tokens":n,"temperature":0.7,
         "chat_template_kwargs":{"enable_thinking":False},
         "messages":[{"role":"user","content":txt}]}
    r = urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps(b).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    return json.loads(urllib.request.urlopen(r,timeout=900).read())["usage"]["completion_tokens"]
temas=["cache de prefixo","decodificacao especulativa","atencao linear","quantizacao NVFP4",
       "paralelismo de tensores","expert parallelism","CUDA graphs","fragmentacao de memoria"]
t0=time.time(); tot=0
for t in temas:
    tot+=gera(f"Escreva um texto tecnico longo e detalhado sobre {t} em inferencia de LLM.", 600)
print(f"  gerei {tot} tokens em {time.time()-t0:.0f}s")
