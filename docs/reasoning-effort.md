# `reasoning_effort` — the template defaults to max

GLM-5.3-Flash's chat template reads `reasoning_effort` from `chat_template_kwargs` and accepts
`low` and `high`; anything else becomes **`max`**, injected as `<|system|>Reasoning Effort: Max`.
A client that sends nothing pays for `max`.

Same simple pt-BR prompt, production server, temperature 1.0 / top_p 0.95:

| `chat_template_kwargs` | reasoning tokens | total tokens | wall time |
|---|---|---|---|
| none (= max) | 1,043 | 1,341 | 77 s (under load) |
| `{"reasoning_effort": "low"}` | 1 | 241 | 6 s |
| `{"reasoning_effort": "high"}` | 16 | 277 | 13 s |
| `{"reasoning_effort": "max"}` | 824 | 1,106 | 22 s |
| `{"enable_thinking": false}` | 0 | 1,583 | 26 s — the plan leaks into the content, in English |

The final answer had the same length at low/high/max. The response carries
`usage.reasoning_tokens`, so a harness can measure this per route. Map routes: fast/simple →
`low`, interactive default → `high`, plan/review → `max`. Never `enable_thinking:false` to "save".
Every request leaving the box sooner is free capacity: it frees one of the 24 slots and its KV.
