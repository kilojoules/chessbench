# Chess Illegal-Move Benchmark — Design Plan

**Working title:** time-to-first-illegal-move in LLM chess, as a function of
(a) whether the board state is provided each turn and (b) standard vs Chess960.

## 1. What prior art says (researched 2026-07-31)

### Already done — don't re-do as the headline

- **Illegal-move rates in full games** are well measured: Acher 2023
  (gpt-3.5-turbo-instruct ~1750 Elo, illegal moves in 16% of games / 0.3% of
  moves; GPT-4 32% of games), Karvonen's `chess_gpt_eval` (<0.1% illegal over
  8,205 moves with 5-retry policy), the LLM CHESS leaderboard + paper
  (arXiv 2512.01992, 50+ models), SPIN-Bench, ChessArena, dubesor.de,
  Silicon Gambit.
- **Board-representation ablations** exist: LLM CHESS ablates Unicode/ASCII/FEN
  (effects are model-specific — e.g. for o4-mini, FEN > ASCII > Unicode);
  KAIST/KRAFTON (arXiv 2507.00726) found FEN vs PGN vs both barely matters for
  puzzle RL, but **SAN notation ≫ UCI** (pretraining prevalence).
- **History-only state tracking is known-hard**: PGN2FEN shows non-reasoning
  models collapse at ~20 halfmoves when reconstructing a FEN from a PGN;
  o3 holds ~90%+ to 100 halfmoves. Toshniwal et al. (AAAI 2022) is the
  pre-LLM foundation.
- **Memorization → illegal moves link is established, but not via 960**: TUM
  paper (arXiv 2601.16823) found GPT-5's illegal-move rate jumps to **33.8%**
  on out-of-distribution positions (random placements) vs near-zero
  in-distribution — but single-move-from-FEN, not full games, and random
  positions, not Chess960.

### Genuinely open — the novelty this benchmark claims

1. **Chess960 full-game play with general-purpose chat LLMs.** No published
   work or open repo does this. (DeepMind's ICML 2025 MAV model played 960 but
   had 1.2B Chess960 positions in training — it's a specialist, so it tests
   nothing about memorization control. PGN2FEN explicitly lists 960 as future
   work.)
2. **The crossed 2×2 design**: board-visibility × variant. Nobody has run it.
   (Silicon Gambit attempted FEN-only vs history-only and self-retracted the
   result; LLM CHESS's ablation left a `get_board` tool available, so
   visibility was the agent's choice, not a controlled condition.)
3. **Survival-analysis framing** of legality (time-to-first-illegal as a
   censored outcome, hazard-by-ply curves). Closest existing metric is
   "average moves survived"; no formal treatment exists.

**Verdict:** the idea is sound and the specific 2×2 + survival framing is a
real gap. Both hypotheses have supporting circumstantial evidence; note that
H1 (no board → earlier illegal moves) may partially *reverse* for some
models — several results suggest LLMs track state better from move history
than from a FEN they must parse. That makes H1 a genuinely open question,
which is good.

## 2. Key design decisions

### D1. Raw LLM loop, not a tool-using agent (pi verdict)

The full pi coding agent is the wrong harness for the primary experiment:
its default tools (bash, read/write) would let the model install python-chess
or an engine — measuring "does it think to use tools", not state tracking —
and its agent loop adds auto-compaction/retry/scaffolding, all uncontrolled
variables. The same applies to any coding-agent harness.

- **Primary harness:** a thin purpose-built loop (~300–500 lines Python),
  python-chess + Stockfish + a provider-agnostic LLM client
  (litellm, or native SDKs behind a small interface).
- **Optional secondary arm ("agentic skyline"):** pi in RPC mode with
  `--no-builtin-tools`, one custom `submit_move` tool, custom system prompt.
  Interesting as a separate condition, never mixed into the main cells.
  (pi's reusable `pi-ai` library is also a fine TS client if we ever want TS.)

### D2. Conditions (the experiment matrix)

Two factors, fully crossed, per model:

| Factor | Levels |
|---|---|
| Board visibility | `history-only` (blindfold: start FEN + SAN move list) vs `history+board` (same, plus current FEN and ASCII diagram each turn) |
| Variant | `standard` (Scharnagl #518) vs `chess960` (fixed drawn set of start positions) |

Symmetry rules that keep the comparison fair:
- **Always** include the starting FEN in the prompt, even in standard games
  (so the 960 cells differ only in *which* position, not in prompt shape).
- The system prompt always states the variant and castling rules.
- Optional third visibility level later: `board-only` (current FEN, no
  history) — completes the triangle Silicon Gambit attempted.

### D3. Move protocol

- **Notation: SAN** (research says SAN ≫ UCI for LLMs; UCI would inflate
  failures for the wrong reason). Log raw output verbatim regardless.
- Strict output protocol: model must end with `MOVE: <san>` (regex-extracted).
- **No legal-move list in the prompt** (providing it is known to mask exactly
  the signal we measure).
- Temperature 0 (or provider default if pinned; record it). Fixed seeds where
  supported.

### D4. Failure taxonomy (maps 1:1 to python-chess exceptions)

| Class | python-chess | Counts as "illegal" for H1/H2? |
|---|---|---|
| Malformed / no move extracted | regex miss or `InvalidMoveError` | separate category ("format failure") |
| Well-formed but illegal | `IllegalMoveError` | **yes — primary event** |
| Ambiguous SAN | `AmbiguousMoveError` | yes, but tracked separately |

All three subclass `ValueError`. Every attempt is logged with its class.

**Retry policy:** on any failure, one feedback message ("that move is
illegal/invalid in this position; reply with a legal move") — up to 3 attempts
per ply, then the game is recorded as *lost by illegality* at that ply.
The **primary metric ignores retries**: the event is the *first* illegal
attempt of the game (retries only affect game continuation, letting us also
collect post-failure data).

### D5. Opponent (Stockfish) configuration

- `brew install stockfish` (stable 18). python-chess `chess.engine` manages
  `UCI_Chess960` automatically — never set it manually; just pass a
  `chess960=True` board.
- **Strength:** Skill Level ~2–4 with a fixed nodes cap (e.g.
  `Limit(nodes=20000)`) for reproducibility. Rationale: a full-strength
  engine ends games in ~30 moves, censoring the survival data; we want long
  games = long exposure windows. Don't trust `UCI_Elo` at low values (floor
  1320, calibrated at 120s+1s time control — meaningless under node caps).
- Record engine version, options, and limits in every game log.

### D6. Chess960 gotchas (verified against python-chess 1.11.2)

- `Board.from_chess960_pos(n)`, Scharnagl n ∈ [0, 959]; **518 = standard**.
- Show the LLM `board.shredder_fen()` (castling rights as file letters,
  e.g. `HFhf`) — plain `fen()` emits X-FEN `KQkq` which is ambiguous for 960.
- SAN castling is always `O-O`/`O-O-O` in both variants (no new failure mode).
- python-chess `parse_uci` silently accepts and normalizes *both* castling
  encodings (`e1g1` and `e1h1`) — one more reason to use SAN for the protocol.
- Position set: draw ~10 Chess960 numbers once with a recorded RNG seed;
  reuse the same set across all models and both visibility conditions
  (paired design). Exclude #518.

### D7. Metrics

Primary: **ply of first illegal attempt**, treated as survival data.
Censoring events: natural game end (checkmate/stalemate/draw), move cap
(200 plies), engine win. Analysis: Kaplan–Meier curves per cell, log-rank
tests, Cox regression with visibility, variant, and their interaction as
covariates (frailty/cluster term per model).

Secondary (all nearly free to collect):
- illegal attempts per 100 plies (with retries)
- **hazard-by-ply curve** — the money plot. Predicted signature: standard
  games show low hazard in the opening (book memory) rising with depth;
  960 games show high hazard from ply 1. That shape difference *is* the
  memorization story.
- format-failure and ambiguous-SAN rates (separate lines)
- centipawn loss per move (Stockfish eval is already running — free) →
  does move *quality* degrade before *legality* breaks?
- game results, game length, tokens/latency/cost per move

### D8. Sample size / cost

~30 games per cell × 4 cells = 120 games/model; at ~40 model calls/game
that's ~5k calls per model — modest. Pilot first (below) before spending.

## 3. Logging schema (JSONL, one record per move attempt)

```json
{
  "game_id": "…", "model": "…", "variant": "chess960", "sp_id": 402,
  "visibility": "history-only", "ply": 37, "attempt": 1,
  "fen_before": "…", "san_history": "…",
  "prompt_tokens": 0, "raw_output": "…",
  "parse_class": "illegal|invalid|ambiguous|legal",
  "move_san": null, "move_uci": null,
  "sf_eval_cp_before": 0, "cpl": null,
  "latency_ms": 0, "output_tokens": 0, "ts": "…"
}
```

Plus one per-game record (result, termination reason, censoring status,
engine config, prompt template hash, seeds). Dump PGN per game too.

## 4. Phases

- **Phase 0 — environment** (½ day): `brew install stockfish`; uv project;
  `python-chess`, `litellm`, `lifelines` (survival analysis); smoke-test a
  Stockfish-vs-Stockfish 960 game end to end.
- **Phase 1 — harness** (1–2 days): game runner; prompt builder with the
  three switches (variant, visibility, notation); parser + taxonomy; retry
  logic; JSONL + PGN logging; a `--dry-run` mode with a scripted fake LLM
  for testing the harness itself.
- **Phase 2 — pilot** (½ day, 1–2 cheap models, ~5 games/cell): tune the
  prompt template; verify the blindfold condition truly leaks no board;
  verify 960 castling round-trips; decide ASCII-board-vs-FEN-vs-both for the
  informed condition; sanity-check game lengths under the chosen Stockfish
  setting.
- **Phase 3 — main run**: model roster (mix of reasoning and non-reasoning
  models) × 2×2 × 30 games, paired 960 position set. Resumable runner
  (idempotent by game_id) so crashes don't burn money.
- **Phase 4 — analysis & writeup**: KM curves, Cox model, hazard-by-ply
  figure, CPL-vs-ply overlay, per-error-class tables.
- **Phase 5 (optional)** — agentic arm via pi (`--no-builtin-tools` +
  `submit_move`), and/or the `board-only` third visibility level.

## 5. Build vs fork

**Build from scratch.** Reasons: `chess_gpt_eval` has **no license** (can't
legally fork) and is unmaintained; `llm_chess` is Apache-2.0 and maintained
but its AG2 agentic dialog protocol entangles "illegal move" with
"hallucinated tool call" and hard-codes the standard start — fighting it
costs more than 300 lines of clean code. Steal *designs*, not code:
TextArena's `is_open`/`show_valid` toggles, `chess_gpt_eval`'s retry/CSV
scheme, dubesor's strict-JSON protocol. Keep prompts close to LLM CHESS
where possible for rough comparability.

## 6. Implementation notes (decisions made during build + adversarial review)

- **Event definition in code:** `event` / `survival_plies` fire on the first
  *illegal-or-ambiguous* attempt (per D4); `first_illegal_*` and
  `first_ambiguous_*` are also recorded separately, and `invalid` (format)
  failures only ever appear in `first_failure_*`.
- **Truncation is infrastructure, not chess.** A `finish_reason == "length"`
  response never enters the taxonomy or consumes a chess attempt; it is
  retried (≤2 extra calls), and persistent truncation ends the game as
  `llm_truncated` / `censored_infra` — never a loss. `finish_reason` and
  `reasoning_tokens` are logged per attempt so contamination is filterable.
  Raise `--max-tokens` (16k+) for reasoning models.
- **Sampling honesty:** litellm's `drop_params` can silently discard
  `temperature=0` for models that reject it. The client probes support at
  startup and records `effective_temperature` (null = provider default) in
  every game record.
- **Eval isolation:** logged evals run on a *separate* full-strength
  Stockfish process; analysing on the playing engine would warm its
  hash/history tables and change the opponent's moves, making `--eval-depth`
  an accidental treatment.
- **Color/position decorrelation:** naive `i % positions` + `i % 2` aliases
  color with position parity (each 960 position always the same color);
  color is offset by the replicate cycle instead, so every position sees
  both colors and each cell stays 15/15.
- **Canonical-SAN echo (deliberate, documented):** "Moves played so far"
  shows canonical SAN — including check marks and corrected disambiguation
  of the model's own moves (PGN convention, symmetric across cells). This
  weakly confirms check status of the model's prior moves in the blindfold
  condition; acceptable, but state it in the writeup.
- **Ambiguous-move feedback never confirms** that the move was legal for 2+
  pieces (that would leak board state into the blindfold cells); the standing
  disambiguation rule lives in the system prompt instead.
- **Resume unit:** a game counts as done only when its JSONL has the final
  `{"type": "game"}` record *and* its PGN exists; the PGN is written first.
- **Staged extraction (parser v3, from qwen3:4b pilot data):** small models
  ignore the `MOVE:` protocol — they end with "**e4**", "**Answer:** d2-d4",
  or `$$ \boxed{Nf3} $$`, and sometimes *quote* the protocol instructions in
  prose. Stages: (1) last line-anchored `MOVE:` line, adopted
  unconditionally; (2) last inline `MOVE:` mention, only if SAN-shaped;
  (3) last `\boxed{}` answer, if SAN-shaped; (4) a bare SAN-shaped final
  line. Prose never matches; refusals stay `invalid`. Each attempt records
  `extraction: protocol|protocol-inline|fallback` for strict-protocol
  sensitivity analyses. Offline re-parse of the pilot showed the strict
  parser had misclassified 5 *real illegal attempts* (and 9 legal moves) as
  format failures — strictness was suppressing the benchmark's own signal.
- **Forfeit naming:** exhausting the attempt budget is `llm_forfeit` /
  `loss_forfeit` with `forfeit_attempt_classes` recording the mix — a
  pure-"invalid" (format) forfeit is censoring, not a survival event, and
  must not be conflated with an illegal-move loss.

## 7. Key references

- Acher illegal-move studies: https://blog.mathieuacher.com/GPTsChessEloRatingLegalMoves/
- Karvonen chess_gpt_eval + world models: https://github.com/adamkarvonen/chess_gpt_eval , arXiv 2403.15498
- LLM CHESS leaderboard/paper: https://github.com/maxim-saplin/llm_chess , arXiv 2512.01992
- TUM memorization/OOD illegal moves: arXiv 2601.16823
- DeepMind planning + 960 (specialist): arXiv 2412.12119
- PGN2FEN (history→state collapse): https://www.aidancooper.co.uk/pgn2fen-benchmark/
- SAN≫UCI, representation ablation: arXiv 2507.00726
- SPIN-Bench: arXiv 2503.12349 ; ChessArena: arXiv 2509.24239
- python-chess docs: https://python-chess.readthedocs.io/
- Stockfish UCI options: https://github.com/official-stockfish/Stockfish/wiki/UCI-Protocol-and-Stockfish-Commands
- pi harness: https://github.com/badlogic/pi-mono
