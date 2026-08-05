"""Causal test: steer the model along a 'standard back rank' direction and
watch the phantom pull move.

The pull experiment shows illegal book moves sit at rank ~2 in chess960
positions. That is correlational. Here we extract a direction in the
residual stream and ADD it (should strengthen the phantom pull) or SUBTRACT
it (should weaken it), which is a causal claim about the representation.

CONTRAST DESIGN (the important part): the positive and negative classes use
the SAME chess960 system prompt and the SAME prompt format, differing only
in the actual piece placement — because chess960 position #518 IS the
standard array. So:

    positive class = chess960 framing, sp 518   (standard back rank)
    negative class = chess960 framing, sp != 518 (shuffled back rank)

That isolates board identity from every textual difference (variant
paragraph, castling rules, FEN style), which a naive standard-vs-960
contrast would confound.

Usage: interp/.venv/bin/python interp/phantom_steer.py [--layer 18] [--n 12]
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

BOOK = ["Nf3", "Nc3", "Bc4", "Bb5", "Nf6", "Nc6", "Qh5"]  # shuffled-piece book moves
STANDARD_SP = 518


def build_prefix(tok, sp_id: int, visibility: str) -> tuple[str, chess.Board]:
    """Always chess960 framing — only the piece placement varies."""
    spec = GameSpec(game_id="steer", model="hf", variant="chess960",
                    visibility=visibility, sp_id=sp_id, llm_color="white", game_index=0)
    board = chess.Board.from_chess960_pos(sp_id)
    text = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt(spec)},
         {"role": "user", "content": move_request(spec, board, board_fen(board), [])}],
        tokenize=False, add_generation_prompt=True) + "MOVE: "
    return text, board


@torch.no_grad()
def last_token_resid(model, tok, prefix: str, layer: int) -> torch.Tensor:
    ids = tok(prefix, return_tensors="pt").to(model.device)
    out = model(**ids, output_hidden_states=True)
    return out.hidden_states[layer][0, -1].float().cpu()


@torch.no_grad()
def score_moves(model, tok, prefix: str, moves: list[str]) -> dict[str, float]:
    pre = tok(prefix, return_tensors="pt").input_ids.to(model.device)
    out = {}
    for mv in moves:
        cont = tok(mv, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        ids = torch.cat([pre, cont], dim=1)
        logits = model(ids).logits[0, pre.shape[1] - 1: -1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        out[mv] = (lp[torch.arange(cont.shape[1]), cont[0]].sum() / cont.shape[1]).item()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--layer", type=int, default=18, help="residual layer to steer")
    ap.add_argument("--n", type=int, default=12, help="chess960 test positions")
    ap.add_argument("--n-vec", type=int, default=16, help="positions per contrast class")
    ap.add_argument("--alphas", default="-2,-1,0,1,2")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("interp/steer_results.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="mps").eval()
    rng = random.Random(args.seed)
    pool = [i for i in range(960) if i != STANDARD_SP]
    vec_positions = rng.sample(pool, args.n_vec)
    test_positions = sorted(rng.sample([p for p in pool if p not in vec_positions], args.n))

    # ---- steering vector: standard back rank minus shuffled back rank ----
    print(f"building steering vector at layer {args.layer} ...", flush=True)
    pos_acts, neg_acts = [], []
    for visibility in ("history-only", "history+board"):
        p, _ = build_prefix(tok, STANDARD_SP, visibility)
        pos_acts.append(last_token_resid(model, tok, p, args.layer))
        for sp in vec_positions:
            n, _ = build_prefix(tok, sp, visibility)
            neg_acts.append(last_token_resid(model, tok, n, args.layer))
    vec = torch.stack(pos_acts).mean(0) - torch.stack(neg_acts).mean(0)
    vec = vec / vec.norm()
    print(f"vector norm-1, dim {vec.shape[0]}", flush=True)

    # ---- steer and re-score ----
    handle = None
    scale = [0.0]

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] = h[:, -1, :] + scale[0] * vec.to(h.device, h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    layer_mod = model.model.layers[args.layer]
    handle = layer_mod.register_forward_hook(hook)

    alphas = [float(a) for a in args.alphas.split(",")]
    rows = []
    for sp in test_positions:
        board = chess.Board.from_chess960_pos(sp)
        legal = sorted(board.san(m) for m in board.legal_moves)
        illegal_book = [b for b in BOOK if b not in legal]
        if not illegal_book:
            continue
        for visibility in ("history-only", "history+board"):
            prefix, _ = build_prefix(tok, sp, visibility)
            cands = sorted(set(legal + illegal_book))
            for a in alphas:
                # residual norms are ~50-100; scale alpha by a typical norm
                scale[0] = a * 8.0
                sc = score_moves(model, tok, prefix, cands)
                order = [m for m, _ in sorted(sc.items(), key=lambda kv: -kv[1])]
                rank = min(order.index(b) for b in illegal_book)
                gap = max(sc[b] for b in illegal_book) - max(sc[m] for m in legal)
                rows.append({"sp_id": sp, "visibility": visibility, "alpha": a,
                             "best_illegal_book_rank": rank, "book_minus_legal": gap,
                             "top": order[0], "book_wins": order[0] in illegal_book})
            print(f"sp{sp:03d} {visibility:14s} " + " ".join(
                f"a={r['alpha']:+.0f}:rank{r['best_illegal_book_rank']}"
                for r in rows[-len(alphas):]), flush=True)
    handle.remove()

    args.out.write_text(json.dumps(rows, indent=2))
    print("\n=== steering effect (all positions pooled) ===")
    for a in alphas:
        sub = [r for r in rows if r["alpha"] == a]
        print(f"alpha {a:+.0f}: median rank of best illegal book move "
              f"{statistics.median(r['best_illegal_book_rank'] for r in sub):.1f} | "
              f"mean logprob gap (book - best legal) {statistics.mean(r['book_minus_legal'] for r in sub):+.3f} | "
              f"illegal book wins outright: {sum(r['book_wins'] for r in sub)}/{len(sub)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
