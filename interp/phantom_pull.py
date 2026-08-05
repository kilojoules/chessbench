"""Phantom pull: how much probability does the model put on standard-chess
book moves in positions where they are ILLEGAL?

Spontaneous phantoms are rare (the book move has to win the argmax), which
makes them a weak signal. This measures the underlying pull continuously:
for every chess960 opening position we score the model's log-probability of
each legal move AND of the classic book moves (Nf3, Nc3, e4, ... ), then ask
where the illegal book moves rank and how much mass they carry — in both
visibility conditions, against the standard start as a baseline.

Usage: interp/.venv/bin/python interp/phantom_pull.py [--n 30]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessbench.config import GameSpec  # noqa: E402
from chessbench.prompts import board_fen, move_request, system_prompt  # noqa: E402

# The opening book the phantom moves came from (observed in the benchmark's
# logged phantoms, most-common first).
BOOK = ["Nf3", "Nc3", "e4", "d4", "Bc4", "Bb5", "Nf6", "Nc6", "Qh5", "e5", "d5"]


def build_prompt(tok, variant: str, sp_id: int, visibility: str):
    spec = GameSpec(game_id="pull", model="hf", variant=variant,
                    visibility=visibility, sp_id=sp_id, llm_color="white",
                    game_index=0)
    board = chess.Board.from_chess960_pos(sp_id) if variant == "chess960" else chess.Board()
    text = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt(spec)},
         {"role": "user", "content": move_request(spec, board, board_fen(board), [])}],
        tokenize=False, add_generation_prompt=True) + "MOVE: "
    return text, board


@torch.no_grad()
def score_moves(model, tok, prefix: str, moves: list[str]) -> dict[str, float]:
    """Mean-token log-prob of each candidate move string after the prefix."""
    pre = tok(prefix, return_tensors="pt").input_ids.to(model.device)
    out: dict[str, float] = {}
    for mv in moves:
        cont = tok(mv, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        ids = torch.cat([pre, cont], dim=1)
        logits = model(ids).logits[0, pre.shape[1] - 1: -1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        tot = lp[torch.arange(cont.shape[1]), cont[0]].sum().item()
        out[mv] = tot / cont.shape[1]  # length-normalized
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("interp/pull_results.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="mps").eval()

    rng = random.Random(args.seed)
    positions = sorted(rng.sample([i for i in range(960) if i != 518], args.n))
    rows = []

    # Baseline: the standard start, where the book moves are legal.
    for visibility in ("history-only", "history+board"):
        prefix, board = build_prompt(tok, "standard", 518, visibility)
        legal = sorted(board.san(m) for m in board.legal_moves)
        scores = score_moves(model, tok, prefix, sorted(set(legal + BOOK)))
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        rows.append({"variant": "standard", "sp_id": 518, "visibility": visibility,
                     "top": ranked[0][0],
                     "book_ranks": {b: [i for i, (m, _) in enumerate(ranked) if m == b][0]
                                    for b in BOOK if b in scores},
                     "illegal_book": []})
        print(f"standard {visibility:14s} top={ranked[0][0]}", flush=True)

    for sp in positions:
        board0 = chess.Board.from_chess960_pos(sp)
        legal = sorted(board0.san(m) for m in board0.legal_moves)
        illegal_book = [b for b in BOOK if b not in legal]
        for visibility in ("history-only", "history+board"):
            prefix, board = build_prompt(tok, "chess960", sp, visibility)
            scores = score_moves(model, tok, prefix, sorted(set(legal + illegal_book)))
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            order = [m for m, _ in ranked]
            # Pull metric: rank of the best ILLEGAL book move among all
            # candidates, and whether it outranks every legal move.
            best_ib = min((order.index(b) for b in illegal_book), default=None)
            rows.append({
                "variant": "chess960", "sp_id": sp, "visibility": visibility,
                "top": order[0], "top_is_illegal_book": order[0] in illegal_book,
                "best_illegal_book": order[best_ib] if best_ib is not None else None,
                "best_illegal_book_rank": best_ib,
                "n_legal": len(legal), "illegal_book": illegal_book,
            })
            mark = "  <-- BOOK WINS" if order[0] in illegal_book else ""
            print(f"sp{sp:03d} {visibility:14s} top={order[0]:6s} "
                  f"best illegal book={order[best_ib] if best_ib is not None else '-':6s} "
                  f"(rank {best_ib}){mark}", flush=True)

    args.out.write_text(json.dumps(rows, indent=2))
    c960 = [r for r in rows if r["variant"] == "chess960"]
    for vis in ("history-only", "history+board"):
        sub = [r for r in c960 if r["visibility"] == vis]
        wins = sum(r["top_is_illegal_book"] for r in sub)
        ranks = [r["best_illegal_book_rank"] for r in sub if r["best_illegal_book_rank"] is not None]
        import statistics
        print(f"\n{vis:14s}: illegal book move ranked #1 in {wins}/{len(sub)} positions | "
              f"median rank of best illegal book move: {statistics.median(ranks):.0f} "
              f"(of ~{statistics.median(r['n_legal'] for r in sub):.0f} legal moves)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
