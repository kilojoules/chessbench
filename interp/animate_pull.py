"""Animate the phantom pull: Qwen's favourite move follows it onto boards
where the piece isn't there.

Intro: the standard start, where Nf3 is the model's #1 choice. Then a cycle
of Chess960 positions with no knight on g1 — in every one, 'Nf3' still
ranks #2 of ~28 candidates. The ghost frame executes the impossible move:
the phantom knight lands on f3 (red) while g1 (red) shows the piece that is
actually there.

Usage: uv run python interp/animate_pull.py [-o interp/phantom_pull.gif]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessbench.animate import RED_SQ, FrameRenderer  # noqa: E402

PIECE_NAME = {"R": "rook", "B": "bishop", "Q": "queen", "K": "king"}
CAP_BG = (98, 26, 24)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("interp/pull_results.json"))
    ap.add_argument("-o", "--out", type=Path, default=Path("interp/phantom_pull.gif"))
    ap.add_argument("--square", type=int, default=64)
    ap.add_argument("--max-positions", type=int, default=10)
    args = ap.parse_args()

    rows = [r for r in json.loads(args.results.read_text())
            if r["variant"] == "chess960" and r["visibility"] == "history-only"
            and r["best_illegal_book"] == "Nf3"]

    rend = FrameRenderer(args.square, "Qwen2.5-3B · the phantom pull",
                         "Chess960, White to move — where does “Nf3” rank?")
    frames, durs = [], []

    # Intro: the habit being tracked.
    board = chess.Board()
    frames.append(rend.frame(board.fen(), None, "standard chess: the model's favourite first move…"))
    durs.append(2000)
    board.push_san("Nf3")
    frames.append(rend.frame(board.fen(), ("g1", "f3"), "…is Nf3 — its #1 choice of 20 moves"))
    durs.append(2600)

    shown = 0
    for r in rows:
        board = chess.Board.from_chess960_pos(r["sp_id"])
        g1 = board.piece_at(chess.G1)
        if g1.piece_type == chess.KNIGHT:
            continue  # 'Nf3' is merely ambiguous there, not impossible
        n_cand = r["n_legal"] + len(r["illegal_book"])
        frames.append(rend.frame(
            board.fen(), None,
            f"Chess960 #{r['sp_id']}: g1 holds a {PIECE_NAME[g1.symbol()]} — Nf3 is impossible"))
        durs.append(1900)
        ghost = board.copy()
        ghost.set_piece_at(chess.F3, chess.Piece(chess.KNIGHT, chess.WHITE))
        frames.append(rend.frame(
            ghost.fen(), None,
            f"yet “Nf3” still ranks #2 of {n_cand} candidates",
            caption_bg=CAP_BG, border=RED_SQ, red={"g1", "f3"}))
        durs.append(2600)
        shown += 1
        if shown >= args.max_positions:
            break

    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=durs, loop=0, optimize=True)
    print(f"wrote {args.out} ({shown} positions, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
