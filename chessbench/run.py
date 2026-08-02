from __future__ import annotations

import argparse
import functools
import json
import queue
import re
import statistics
import sys
import threading
import time
import zlib
from pathlib import Path

from .config import PROMPT_VERSION, STANDARD_SP_ID, VARIANTS, VISIBILITIES, EngineConfig, GameSpec
from .engine import Engine
from .game import play_game
from .llm import FakeLLM, LiteLLMClient
from .positions import draw_chess960_positions, draw_offbook_prefixes

VIS_SLUG = {"history-only": "blind", "history+board": "board"}
VARIANT_SLUG = {"standard": "standard", "chess960": "chess960", "standard-offbook": "offbook"}

# Progress lines must land in redirected logs immediately (long detached runs
# are watched via `tail -f`); default block buffering hides them for hours.
print = functools.partial(print, flush=True)


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model)


def game_done(path: Path) -> bool:
    """A game is complete iff its JSONL contains a {"type": "game"} record
    AND its PGN exists (both are needed for analysis)."""
    if not path.exists() or not path.with_suffix(".pgn").exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            return any(json.loads(line).get("type") == "game" for line in f if line.strip())
    except (OSError, json.JSONDecodeError):
        return False


def build_specs(model: str, variants: list[str], visibilities: list[str], games_per_cell: int,
                positions: list[int], prefixes: list[list[str]],
                args: argparse.Namespace) -> list[GameSpec]:
    slug = model_slug(model)
    specs = []
    for variant in variants:
        for visibility in visibilities:
            for i in range(games_per_cell):
                prefix: tuple[str, ...] = ()
                prefix_id = None
                if variant == "standard":
                    sp_id = STANDARD_SP_ID
                    color = "white" if i % 2 == 0 else "black"
                    unit = f"sp{sp_id:03d}"
                elif variant == "standard-offbook":
                    sp_id = STANDARD_SP_ID
                    prefix_id = i % len(prefixes)
                    prefix = tuple(prefixes[prefix_id])
                    # Same cycle-offset decorrelation as chess960 positions.
                    color = "white" if (i % len(prefixes) + i // len(prefixes)) % 2 == 0 else "black"
                    unit = f"pfx{prefix_id:02d}"
                else:
                    sp_id = positions[i % len(positions)]
                    # Offset color by the replicate cycle so it doesn't alias
                    # with position parity: every position gets both colors.
                    color = "white" if (i % len(positions) + i // len(positions)) % 2 == 0 else "black"
                    unit = f"sp{sp_id:03d}"
                game_id = (
                    f"{slug}__{VARIANT_SLUG[variant]}__{VIS_SLUG[visibility]}__{unit}__g{i:03d}__{color[0]}"
                )
                specs.append(
                    GameSpec(
                        game_id=game_id,
                        model=model,
                        variant=variant,
                        visibility=visibility,
                        sp_id=sp_id,
                        llm_color=color,
                        game_index=i,
                        opening_prefix=prefix,
                        prefix_id=prefix_id,
                        max_plies=args.max_plies,
                        max_attempts=args.max_attempts,
                        temperature=args.temperature,
                    )
                )
    return specs


def make_llm(spec: GameSpec, args: argparse.Namespace):
    if spec.model.startswith("fake:"):
        policy = spec.model.split(":", 1)[1]
        seed = zlib.crc32(spec.game_id.encode())
        return FakeLLM(policy=policy, seed=seed)
    return LiteLLMClient(
        model=spec.model,
        temperature=spec.temperature,
        max_tokens=args.max_tokens,
        timeout=args.llm_timeout,
        seed=args.llm_seed,
        num_ctx=args.num_ctx,
    )


def summarize(records: list[dict]) -> None:
    cells = {}
    for r in records:
        cells.setdefault((r["variant"], r["visibility"]), []).append(r)
    print("\ncell summary:")
    for (variant, visibility), rs in sorted(cells.items()):
        events = [r for r in rs if r["event"]]
        med = statistics.median(r["survival_plies"] for r in rs) if rs else None
        print(
            f"  {variant:9s} x {visibility:14s}: {len(rs):3d} games, "
            f"{len(events):3d} with illegal event, median survival {med} plies"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="chessbench",
        description="LLM-vs-Stockfish illegal-move benchmark (see PLAN.md)",
    )
    p.add_argument("--model", action="append", required=True,
                   help="litellm model string (e.g. anthropic/claude-sonnet-5) or fake:first|random|always-illegal; repeatable")
    p.add_argument("--variant", choices=[*VARIANTS, "both", "all"], default="all",
                   help="'both' = standard+chess960 (the original 2x2); 'all' adds standard-offbook")
    p.add_argument("--visibility", choices=[*VISIBILITIES, "both"], default="both")
    p.add_argument("--games-per-cell", type=int, default=30)
    p.add_argument("--n-positions", type=int, default=10, help="size of the fixed Chess960 start-position set")
    p.add_argument("--position-seed", type=int, default=2026)
    p.add_argument("--prefix-plies", type=int, default=6, help="random opening length for standard-offbook")
    p.add_argument("--prefix-seed", type=int, default=2027)
    p.add_argument("--out", type=Path, default=Path("runs/dev"))
    p.add_argument("--stockfish", default="stockfish")
    p.add_argument("--skill", type=int, default=3, help="Stockfish Skill Level (0-20)")
    p.add_argument("--nodes", type=int, default=20_000, help="Stockfish node cap per move")
    p.add_argument("--eval-depth", type=int, default=10, help="depth for logged evals; 0 disables")
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="completion budget; raise substantially (16k+) for reasoning models, "
                        "whose thinking tokens count against it")
    p.add_argument("--llm-seed", type=int, default=None,
                   help="sampling seed passed to providers that support it")
    p.add_argument("--llm-timeout", type=float, default=600.0,
                   help="per-request timeout in seconds; local thinking models can need minutes per move")
    p.add_argument("--num-ctx", type=int, default=None,
                   help="ollama context window (its 4096 default silently truncates long thinking; "
                        "set >= prompt + max-tokens)")
    p.add_argument("--parallel", type=int, default=1,
                   help="concurrent games (needs OLLAMA_NUM_PARALLEL >= this for local models)")
    p.add_argument("--list", action="store_true", help="print the planned games and exit")
    args = p.parse_args(argv)

    # Context-shifting guard: ollama's default 4096-token window silently
    # drops the oldest context (including the system prompt) when thinking
    # runs long — a validity bug, not a performance one. Never run an ollama
    # model without an explicit, sufficient window.
    if args.num_ctx is None and any(m.startswith("ollama") for m in args.model):
        args.num_ctx = args.max_tokens + 4096
        print(f"[guard] --num-ctx not set for an ollama model; defaulting to "
              f"{args.num_ctx} (max-tokens + 4096) to prevent silent context shifting")
    if args.num_ctx is not None and args.num_ctx < args.max_tokens + 1024:
        p.error(f"--num-ctx {args.num_ctx} leaves under 1024 tokens for the prompt at "
                f"--max-tokens {args.max_tokens}; generation would context-shift or truncate")

    if args.variant == "all":
        variants = list(VARIANTS)
    elif args.variant == "both":
        variants = ["standard", "chess960"]
    else:
        variants = [args.variant]
    visibilities = list(VISIBILITIES) if args.visibility == "both" else [args.visibility]
    positions = draw_chess960_positions(args.n_positions, args.position_seed)
    prefixes = draw_offbook_prefixes(args.n_positions, args.prefix_plies, args.prefix_seed)

    all_specs = []
    for model in args.model:
        all_specs.extend(
            build_specs(model, variants, visibilities, args.games_per_cell, positions, prefixes, args)
        )

    if args.list:
        for s in all_specs:
            print(s.game_id)
        print(f"\n{len(all_specs)} games planned; chess960 positions: {positions}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    engine_cfg = EngineConfig(path=args.stockfish, skill_level=args.skill, nodes=args.nodes,
                              eval_depth=args.eval_depth)

    with Engine(EngineConfig(path=args.stockfish, skill_level=args.skill,
                             nodes=args.nodes, eval_depth=0)) as probe:
        engine_name = probe.name
    manifest = {
        "argv": sys.argv[1:] if argv is None else argv,
        "models": args.model,
        "variants": variants,
        "visibilities": visibilities,
        "games_per_cell": args.games_per_cell,
        "chess960_positions": positions,
        "position_seed": args.position_seed,
        "offbook_prefixes": [" ".join(p) for p in prefixes],
        "prefix_plies": args.prefix_plies,
        "prefix_seed": args.prefix_seed,
        "engine": {"name": engine_name, "skill_level": args.skill, "nodes": args.nodes},
        "max_plies": args.max_plies,
        "max_attempts": args.max_attempts,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "llm_seed": args.llm_seed,
        "num_ctx": args.num_ctx,
        "parallel": args.parallel,
        "prompt_version": PROMPT_VERSION,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (args.out / f"manifest-{stamp}.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Health probe: one trivial completion per real model BEFORE burning
    # games — a dead server (e.g. crashed GPU backend) can answer 200s with
    # empty bodies, which must never be graded as chess.
    for model in args.model:
        if model.startswith("fake:"):
            continue
        probe_llm = LiteLLMClient(model=model, temperature=args.temperature,
                                  max_tokens=256, timeout=args.llm_timeout,
                                  num_ctx=args.num_ctx)
        try:
            r = probe_llm.complete([{"role": "user", "content": "Reply with the single word: ready"}])
        except Exception as e:
            p.error(f"health probe failed for {model}: {type(e).__name__}: {e}")
        if not r.text.strip() and not (r.output_tokens or 0):
            p.error(f"health probe for {model} returned an empty zero-token response — "
                    "the serving backend looks dead (restart it and re-run)")
        print(f"[probe] {model} ok ({r.output_tokens} tokens, {r.latency_ms} ms)")

    todo = [s for s in all_specs if not game_done(args.out / f"{s.game_id}.jsonl")]
    n_skipped = len(all_specs) - len(todo)

    completed: list[dict] = []
    lock = threading.Lock()
    stop = threading.Event()
    state = {"done": 0, "consecutive_errors": 0}
    work: queue.Queue = queue.Queue()
    for s in todo:
        work.put(s)

    def worker() -> None:
        # Each worker owns its Stockfish pair; game files are disjoint, so
        # workers only share the progress lock and the error circuit breaker.
        with Engine(engine_cfg) as engine:
            while not stop.is_set():
                try:
                    spec = work.get_nowait()
                except queue.Empty:
                    return
                llm = make_llm(spec, args)
                try:
                    rec = play_game(spec, llm, engine, args.out)
                except Exception as e:
                    # One bad game must not kill a long run; its file stays
                    # incomplete, so a re-run resumes it. Repeated back-to-back
                    # failures mean something systemic (server down, bad key) —
                    # stop burning games.
                    with lock:
                        state["done"] += 1
                        state["consecutive_errors"] += 1
                        print(f"[{state['done']}/{len(todo)}] {spec.game_id}: "
                              f"ERROR {type(e).__name__}: {e}")
                        if state["consecutive_errors"] >= 3:
                            print("3 consecutive game errors — aborting; "
                                  "fix the cause and re-run to resume")
                            stop.set()
                    continue
                with lock:
                    state["done"] += 1
                    state["consecutive_errors"] = 0
                    completed.append(rec)
                    print(
                        f"[{state['done']}/{len(todo)}] {spec.game_id}: {rec['llm_result']} "
                        f"({rec['termination']}, {rec['plies']} plies"
                        + (f", first event at ply {rec['first_event_ply']}" if rec["event"] else "")
                        + ")"
                    )

    n_workers = max(1, min(args.parallel, len(todo) or 1))
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if n_skipped:
        print(f"\nskipped {n_skipped} already-completed games (resume)")
    if completed:
        summarize(completed)
    return 1 if stop.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
