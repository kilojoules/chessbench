"""History test: does score("Nf3") track the board, or is it a string prior?

Story (a) — the model believes/tracks a board — predicts Nf3's score
depends on whether a knight can actually reach f3 in the current position.
Story (b) — a frequency prior over move strings with no board model —
predicts score("Nf3") is roughly invariant to the position.

Ply-1 Chess960 can't separate these, so we use histories on the STANDARD
board. Memorized openings confound the comparison (corpora also never
repeat Nf3 after 1.Nf3), so the main test uses seeded RANDOM legal games:
out-of-distribution sequences the model cannot have memorized, split by
whether "Nf3" is legal at the endpoint. Board-tracking predicts a legality
gap; a string prior predicts none.

Anchors (hand-picked histories) calibrate the extremes, including the
knight-return sequence 1.Nf3 Nc6 2.Ng1 Nb8 — corpus-frequency ~zero but
Nf3 fully legal again.

Usage: interp/.venv/bin/python interp/phantom_history.py [--n 16]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessbench.config import GameSpec  # noqa: E402
from chessbench.prompts import board_fen, move_request, system_prompt  # noqa: E402

ANCHORS = [
    ("ply1", []),
    ("after 1.e4 e5 (Nf3 legal, book)", ["e4", "e5"]),
    ("after 1.Nf3 Nf6 (Nf3 illegal: own knight sits on f3)", ["Nf3", "Nf6"]),
    ("after 1.Nf3 Nc6 2.Ng1 Nb8 (Nf3 legal again, corpus-zero)",
     ["Nf3", "Nc6", "Ng1", "Nb8"]),
]


def nf3_status(board: chess.Board) -> str:
    try:
        board.parse_san("Nf3")
        return "legal"
    except chess.AmbiguousMoveError:
        return "ambiguous"
    except ValueError:
        return "illegal"


def build_prompt(tok, history: list[str], visibility: str) -> tuple[str, chess.Board]:
    spec = GameSpec(game_id="hist", model="hf", variant="standard",
                    visibility=visibility, sp_id=518, llm_color="white",
                    game_index=0)
    start = chess.Board()
    board = chess.Board()
    for san in history:
        board.push_san(san)
    text = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt(spec)},
         {"role": "user", "content": move_request(spec, board, board_fen(start), history)}],
        tokenize=False, add_generation_prompt=True) + "MOVE: "
    return text, board


@torch.no_grad()
def score_moves(model, tok, prefix: str, moves: list[str]) -> dict[str, float]:
    pre = tok(prefix, return_tensors="pt").input_ids.to(model.device)
    out: dict[str, float] = {}
    for mv in moves:
        cont = tok(mv, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        ids = torch.cat([pre, cont], dim=1)
        logits = model(ids).logits[0, pre.shape[1] - 1: -1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        out[mv] = (lp[torch.arange(cont.shape[1]), cont[0]].sum() / cont.shape[1]).item()
    return out


def random_endpoints(rng: random.Random, n_per_group: int, depths=(6, 8, 10)):
    """Seeded random legal games on the standard board, White to move at the
    end, grouped by whether 'Nf3' is legal there (ambiguous skipped)."""
    groups: dict[str, list[list[str]]] = {"legal": [], "illegal": []}
    while any(len(v) < n_per_group for v in groups.values()):
        depth = rng.choice(depths)  # even → White to move
        board = chess.Board()
        hist: list[str] = []
        ok = True
        for _ in range(depth):
            moves = list(board.legal_moves)
            if not moves:
                ok = False
                break
            mv = rng.choice(moves)
            hist.append(board.san(mv))
            board.push(mv)
        if not ok or board.is_game_over():
            continue
        status = nf3_status(board)
        if status != "ambiguous" and len(groups[status]) < n_per_group:
            groups[status].append(hist)
    return groups


def score_position(model, tok, history: list[str], visibility: str) -> dict:
    prefix, board = build_prompt(tok, history, visibility)
    status = nf3_status(board)
    legal = sorted(board.san(m) for m in board.legal_moves)
    cands = sorted(set(legal) | {"Nf3"})
    sc = score_moves(model, tok, prefix, cands)
    order = [m for m, _ in sorted(sc.items(), key=lambda kv: -kv[1])]
    others = [sc[m] for m in legal if m != "Nf3"]
    return {"history": history, "visibility": visibility, "nf3_legal": status == "legal",
            "n_cand": len(cands), "nf3_rank": order.index("Nf3"),
            "nf3_lp": sc["Nf3"], "best_other_lp": max(others),
            "median_other_lp": statistics.median(others), "top": order[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=16, help="random positions per legality group")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=Path("interp/history_results.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="mps").eval()
    rng = random.Random(args.seed)

    rows = []
    print("=== anchors ===", flush=True)
    for label, hist in ANCHORS:
        for vis in ("history-only", "history+board"):
            r = score_position(model, tok, hist, vis)
            r["anchor"] = label
            rows.append(r)
            print(f"{label:55s} {vis:14s} rank {r['nf3_rank']:2d}/{r['n_cand']} "
                  f"lp {r['nf3_lp']:+.2f} (best other {r['best_other_lp']:+.2f}) top={r['top']}",
                  flush=True)

    print("\n=== random games ===", flush=True)
    groups = random_endpoints(rng, args.n)
    for status, hists in groups.items():
        for hist in hists:
            for vis in ("history-only", "history+board"):
                r = score_position(model, tok, hist, vis)
                r["anchor"] = None
                rows.append(r)
                print(f"[Nf3 {status:7s}] {vis:14s} rank {r['nf3_rank']:2d}/{r['n_cand']} "
                      f"lp {r['nf3_lp']:+.2f} gap {r['nf3_lp']-r['best_other_lp']:+.2f} "
                      f"| {' '.join(hist[:6])}...", flush=True)

    args.out.write_text(json.dumps(rows, indent=2))
    print("\n=== summary (random games only) ===")
    rand = [r for r in rows if r["anchor"] is None]
    for vis in ("history-only", "history+board"):
        for legal in (True, False):
            s = [r for r in rand if r["visibility"] == vis and r["nf3_legal"] == legal]
            print(f"{vis:14s} Nf3 {'legal  ' if legal else 'illegal'}: "
                  f"mean lp {statistics.mean(r['nf3_lp'] for r in s):+.3f} | "
                  f"mean lp vs median-legal {statistics.mean(r['nf3_lp']-r['median_other_lp'] for r in s):+.3f} | "
                  f"median rank {statistics.median(r['nf3_rank'] for r in s):.0f}/{statistics.median(r['n_cand'] for r in s):.0f} | "
                  f"top1 {sum(r['top']=='Nf3' for r in s)}/{len(s)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
