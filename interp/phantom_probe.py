"""Step 0 of the phantom-standard interpretability work: does the HF bf16
model reproduce the phantom behaviour the benchmark logged from the
Q4-quantized ollama build?

Everything downstream (logit lens, board probes, causal patching) is
meaningless if the model we can hook doesn't exhibit the phenomenon, so
this gate runs first.

Usage:
    interp/.venv/bin/python interp/phantom_probe.py [--n 24] [--model ...]
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
from chessbench.parsing import classify_move  # noqa: E402
from chessbench.prompts import board_fen, move_request, system_prompt  # noqa: E402


def phantom_standard(history: list[str], candidate: str | None) -> bool:
    board = chess.Board()
    try:
        for san in history:
            board.push_san(san)
        board.parse_san(candidate or "")
        return True
    except ValueError:
        return False


def make_prompt(sp_id: int, visibility: str) -> tuple[str, str, chess.Board]:
    """A chess960 opening position in the benchmark's exact prompt format."""
    spec = GameSpec(game_id="interp", model="hf", variant="chess960",
                    visibility=visibility, sp_id=sp_id, llm_color="white",
                    game_index=0)
    board = chess.Board.from_chess960_pos(sp_id)
    return system_prompt(spec), move_request(spec, board, board_fen(board), []), board


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=24, help="chess960 positions to test")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("interp/gate_results.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="mps")
    model.eval()

    rng = random.Random(args.seed)
    positions = sorted(rng.sample([i for i in range(960) if i != 518], args.n))

    rows = []
    for visibility in ("history-only", "history+board"):
        for sp in positions:
            sys_p, user_p, board = make_prompt(sp, visibility)
            text = tok.apply_chat_template(
                [{"role": "system", "content": sys_p},
                 {"role": "user", "content": user_p}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=200, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            pr = classify_move(board, reply)
            is_phantom = pr.parse_class == "illegal" and phantom_standard([], pr.candidate)
            rows.append({"sp_id": sp, "visibility": visibility, "candidate": pr.candidate,
                         "parse_class": pr.parse_class, "phantom": is_phantom})
            flag = "PHANTOM" if is_phantom else pr.parse_class
            print(f"sp{sp:03d} {visibility:14s} -> {str(pr.candidate):8s} [{flag}]", flush=True)

    n = len(rows)
    ill = [r for r in rows if r["parse_class"] == "illegal"]
    ph = [r for r in rows if r["phantom"]]
    print(f"\n{n} positions | illegal: {len(ill)} ({len(ill)/n:.0%}) | "
          f"phantom: {len(ph)} ({len(ph)/max(len(ill),1):.0%} of illegal)")
    from collections import Counter
    print("phantom moves:", Counter(r["candidate"] for r in ph).most_common())
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
