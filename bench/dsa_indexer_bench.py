#!/usr/bin/env python3
"""Time deep_gemm.fp8_mqa_logits (the DSA indexer prefill kernel) with GLM-5.3-Flash shapes.
On RTX PRO 6000 the SM120 kernel shipped in the image hit 646 TFLOPS fp8 (~60-65% of peak) — it is NOT
the reason prefill slows down at 300k+ context."""
import torch, time, deep_gemm, json, re
cfg=json.load(open("/root/model-glm53/config.json")); tc=cfg.get("text_config",cfg)
H=int(tc.get("index_n_heads",32)); D=int(tc.get("index_head_dim",128))
n=8192; dev="cuda:0"
for L in (50_000, 100_000):
    q=torch.randn(n,H,D,device=dev,dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k=torch.randn(L,D,device=dev,dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    ks_=torch.ones(L,device=dev,dtype=torch.float32)
    w=torch.rand(n,H,device=dev,dtype=torch.float32)
    ks=torch.zeros(n,device=dev,dtype=torch.int32); ke=torch.full((n,),L,device=dev,dtype=torch.int32)
    for _ in range(2): out=deep_gemm.fp8_mqa_logits(q,(k,ks_),w,ks,ke,clean_logits=True)
    torch.cuda.synchronize(); ts=[]
    for _ in range(5):
        t=time.perf_counter(); out=deep_gemm.fp8_mqa_logits(q,(k,ks_),w,ks,ke,clean_logits=True); torch.cuda.synchronize(); ts.append(time.perf_counter()-t)
    ms=min(ts)*1000; flop=2*n*L*H*D
    print(f"  fp8_mqa_logits q={n} L={L} H={H} D={D}: {ms:.1f} ms  ->  {flop/min(ts)/1e12:.0f} TFLOPS  (logits {out.shape}, {out.numel()*4/2**30:.2f} GiB)", flush=True)
    del out,q,k; torch.cuda.empty_cache()
