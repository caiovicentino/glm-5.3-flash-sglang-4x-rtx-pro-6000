#!/usr/bin/env python3
"""Allreduce microbenchmark in 4 processes, same torch/NCCL as the server. Run next to production
(<1 GB per GPU). Compare env variants, e.g.: NCCL_P2P_LEVEL=PHB, NCCL_MIN_NCHANNELS=8, NCCL_PROTO=LL.
Use NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P to see the transport per hop ("via P2P/CUMEM" vs "via SHM").
On our box (2 sockets, PCIe Gen5, no NVLink) the DEFAULT already uses P2P on all 4 hops: 8 KB = 22 us,
48 KB = 17 us (idle). No env variable helped; PHB and Tree made it worse."""
import os, time, statistics, torch, torch.distributed as dist, torch.multiprocessing as mp
SIZES_KB = (8, 48, 256)
def worker(rank, world):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=os.environ.get("BENCH_PORT", "29581"))
    torch.cuda.set_device(rank); dist.init_process_group("nccl", rank=rank, world_size=world)
    for kb in SIZES_KB:
        x = torch.ones(kb * 512, dtype=torch.bfloat16, device="cuda")
        for _ in range(100): dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        lats = []
        for b in range(20):
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(200): dist.all_reduce(x)
            torch.cuda.synchronize(); lats.append((time.perf_counter() - t) / 200 * 1e6)
        if rank == 0: print(f"RESULT {kb:5d} KB: min {min(lats):6.1f}  mediana {statistics.median(lats):6.1f}  max {max(lats):7.1f} us/op", flush=True)
    dist.barrier(); dist.destroy_process_group()
if __name__ == "__main__": mp.spawn(worker, args=(4,), nprocs=4)
