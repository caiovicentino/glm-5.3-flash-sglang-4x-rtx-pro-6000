# Incidents and what they taught (anonymized)

1. **Five outages in one day (model switch, 2026-08-26).** Causes: `--max-running-requests 8`
   queueing, `--cuda-graph-max-bs-decode 32` OOM at capture, chunk 16k OOM (indexer buffer),
   mem-fraction 0.85 OOM, and an SSH session that dropped between `pkill` and the relaunch —
   production dead for 9 minutes with nobody to bring it up. Fix: kill and relaunch in ONE detached
   script on the box (`serve/restart_with_fallback.sh`), triggered by the shortest possible SSH
   command. Also: a live process with 0 MiB of VRAM is *loading*, not dead — we declared death
   prematurely twice.
2. **Vision silently disabled by the chat template.** The requant shipped a `chat_template.jinja`
   that injected "you are unable to process this image"; the model politely refused images with no
   error anywhere. Weights were fine. Replace with the original `zai-org` template (has the
   `emit_image()` macros).
3. **Repetition loops traced to the client, not the server.** `enable_thinking:false` in a client
   config selected a parser path that does not separate reasoning from content. Four server-side
   searches found nothing; the fix was one line in the client. Client config is inference surface.
4. **The adaptive speculation flip-flop.** 21,074 step changes in 20 hours, invisible in every
   aggregate metric; found only by counting log lines. Cost: four graph states for two used, and
   noise. Fix: one candidate per batch bucket.
5. **Static 5/6 speculation looked like a win and was a regression.** Batch 1: 149 → 150 tok/s;
   batch 2–6: −6…−24%. Only visible per batch size. Rule: compare `per_batch.sh` on both logs, never
   aggregates, and wait a full day.
6. **A false rollback almost fired.** After a restart during traffic, the server was already
   serving but `/health` returned 503 for five minutes: every cron came back cold (the prefix cache
   is empty after a restart), 1.4 M tokens of prefill queued, and the health probe could not
   generate. The fallback watchdog would have killed a good server. Fix: accept "server is fired
   up" in the log as healthy too; restart in quiet hours.
7. **A hypothesis read from the NCCL source was wrong for our runtime.** "Default PXB means no P2P
   on Intel hosts" — a 3-minute allreduce microbenchmark showed P2P/CUMEM on all four hops. Runtime
   hypotheses get a microbenchmark before they get a restart.
8. **The dashboard's "loading…" forever.** SSH handshakes to the box went from 2 s to 4.6 s; the
   API endpoint collected six SSH-backed sources inside the request (1m52) and the browser gave up.
   Fix: serve the last snapshot, collect in a background thread, and multiplex SSH
   (`ControlMaster`, short `ControlPath` — the macOS `$TMPDIR` path exceeds the socket limit and
   ssh silently falls back).
9. **The quality gate that always failed.** The long pt-BR gate gave a thinking model 4,000 tokens
   for a 2,500-word essay; the model planned the whole essay inside its reasoning and hit the
   limit with empty content — on every config, including good ones. Gates need a baseline per
   config before their verdict is trusted.
