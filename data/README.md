# Published run data

Complete raw records for the three runs reported in the top-level README.
`runs/` is a scratch directory and stays gitignored; this is the published
dataset.

| directory | model | games | notes |
|---|---|---|---|
| `qwen25-pilot/` | qwen2.5:3b (ollama) | 120 | below the benchmark's floor |
| `qwen25-7b-pilot/` | qwen2.5:7b (ollama) | 120 | first tier with real separation |
| `sonnet-nothink/` | claude-code:sonnet, `MAX_THINKING_TOKENS=0` | 30 | frontier tier; CLI scaffolding, no sampling control |

Per game: `<game_id>.jsonl` (full records) and `<game_id>.pgn`. Each
directory also carries the run `manifest-*.json` (full CLI invocation,
engine build and settings, seeds, chess960 position set, offbook prefixes,
prompt/parser versions).

The two qwen pilots predate the halt-at-first-illegal rule (PLAN.md
amendment v1.3), so their games may contain several illegal attempts;
sonnet games halt at the first. Event times and censoring semantics are
identical across the boundary — only taxonomy denominators differ.

## Record types in each JSONL

- `game_start` — the verbatim system prompt the model received.
- `prefix_move` — a move from the seeded random opening (offbook only).
- `attempt` — one LLM call: the exact request `prompt`, the verbatim
  `raw_output`, the extracted `candidate`, its `parse_class`
  (`legal` / `illegal` / `ambiguous` / `invalid` / `truncated`), the
  `extraction` stage that found it, `fen_before`, Stockfish eval, token
  counts, latency, and infrastructure flags (`finish_reason`,
  `context_overflow`).
- `engine_move` — Stockfish's reply.
- `game` — the final summary: survival times on both scales
  (`survival_llm_moves`, `survival_plies`), first-event indices by class,
  termination, result, class counts, and the full operating configuration
  (temperature actually applied, context window, thinking budget, engine
  settings, prompt/parser versions).

## Reproducing the analysis

```sh
uv sync --extra analysis
uv run chessbench-analyze data/sonnet-nothink
```

Any subset works, and several directories can be passed at once. The
figures and reports in `docs/results/` were generated exactly this way.
