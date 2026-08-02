from __future__ import annotations

import json
import time
from pathlib import Path

import chess
import chess.pgn

from .config import PARSER_VERSION, PROMPT_VERSION, GameSpec
from .parsing import ParseResult, classify_move
from .prompts import board_fen, move_request, retry_feedback, start_board, system_prompt

# Extra API calls allowed per ply when the response is truncated
# (finish_reason == "length"); truncation is an infrastructure failure, not a
# chess failure, so it never consumes a chess attempt or enters the taxonomy.
MAX_TRUNCATION_RETRIES = 2


def play_game(spec: GameSpec, llm, engine, out_dir: Path) -> dict:
    """Play one game and stream per-attempt records to <game_id>.jsonl.

    The final line of the JSONL file is the {"type": "game"} record (also
    returned); its presence marks the game complete for resumption. A PGN of
    the played moves is written alongside.
    """
    board = start_board(spec.variant, spec.sp_id)
    start_fen = board_fen(board)
    llm_is_white = spec.llm_color == "white"
    san_history: list[str] = []
    sys_prompt = system_prompt(spec)

    termination = None  # outcome name | "move_cap" | "llm_forfeit" | "llm_truncated"
    winner = None  # "llm" | "engine" | None
    result_str = "*"
    counts = {"legal": 0, "illegal": 0, "ambiguous": 0, "invalid": 0, "truncated": 0}
    # (ply, llm_move_index) of firsts. The survival EVENT (PLAN.md D4) is the
    # first illegal-or-ambiguous attempt; illegal/ambiguous are also tracked
    # separately, and "failure" covers any taxonomy non-legal (not truncation).
    first_event = None
    first_illegal = None
    first_ambiguous = None
    first_failure = None
    forfeit_classes = None
    llm_move_index = 0
    ts_start = time.time()

    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / f"{spec.game_id}.jsonl"
    with records_path.open("w", encoding="utf-8") as f:

        def emit(rec: dict) -> None:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

        # standard-offbook: play the seeded random prefix onto the board.
        # These moves appear in the model's history but are nobody's play;
        # LLM exposure (llm_move_index) starts after them.
        for san in spec.opening_prefix:
            move = board.parse_san(san)
            emit(
                {
                    "type": "prefix_move",
                    "game_id": spec.game_id,
                    "ply": board.ply() + 1,
                    "move_san": san,
                    "move_uci": board.uci(move),
                    "fen_before": board_fen(board),
                    "ts": time.time(),
                }
            )
            san_history.append(san)
            board.push(move)

        while True:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                termination = outcome.termination.name.lower()
                result_str = outcome.result()
                if outcome.winner is not None:
                    llm_won = (outcome.winner == chess.WHITE) == llm_is_white
                    winner = "llm" if llm_won else "engine"
                break
            # Cap counts played plies excluding the prefix, so offbook games
            # get the same exposure window as the other variants.
            if board.ply() - len(spec.opening_prefix) >= spec.max_plies:
                termination = "move_cap"
                break

            ply = board.ply() + 1
            if (board.turn == chess.WHITE) == llm_is_white:
                llm_move_index += 1
                eval_before = engine.eval_cp_white(board)
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": move_request(spec, board, start_fen, san_history)},
                ]
                played = False
                infra_truncated = False
                attempt = 0
                trunc_tries = 0
                ply_classes: list[str] = []
                while attempt < spec.max_attempts:
                    resp = llm.complete(messages, board=board)
                    truncated = resp.finish_reason == "length"
                    if truncated:
                        trunc_tries += 1
                        pr = ParseResult(
                            "truncated", None, None, None, None,
                            "response truncated (finish_reason=length)",
                        )
                    else:
                        attempt += 1
                        pr = classify_move(board, resp.text)
                    ply_classes.append(pr.parse_class)
                    counts[pr.parse_class] += 1
                    if pr.parse_class in ("illegal", "ambiguous", "invalid") and first_failure is None:
                        first_failure = (ply, llm_move_index)
                    if pr.parse_class in ("illegal", "ambiguous") and first_event is None:
                        first_event = (ply, llm_move_index)
                    if pr.parse_class == "illegal" and first_illegal is None:
                        first_illegal = (ply, llm_move_index)
                    if pr.parse_class == "ambiguous" and first_ambiguous is None:
                        first_ambiguous = (ply, llm_move_index)
                    emit(
                        {
                            "type": "attempt",
                            "game_id": spec.game_id,
                            "model": spec.model,
                            "variant": spec.variant,
                            "visibility": spec.visibility,
                            "sp_id": spec.sp_id,
                            "llm_color": spec.llm_color,
                            "ply": ply,
                            "llm_move_index": llm_move_index,
                            "attempt": attempt if not truncated else attempt + 1,
                            "trunc_try": trunc_tries if truncated else None,
                            "fen_before": board_fen(board),
                            "raw_output": resp.text,
                            "extracted": pr.extracted,
                            "candidate": pr.candidate,
                            "parse_class": pr.parse_class,
                            "extraction": pr.extraction,
                            "move_san": pr.move_san,
                            "move_uci": pr.move_uci,
                            "parse_error": pr.error,
                            "finish_reason": resp.finish_reason,
                            "reasoning": resp.reasoning,
                            "eval_cp_white_before": eval_before,
                            "prompt_tokens": resp.prompt_tokens,
                            "output_tokens": resp.output_tokens,
                            "reasoning_tokens": resp.reasoning_tokens,
                            "latency_ms": resp.latency_ms,
                            "ts": time.time(),
                        }
                    )
                    if truncated:
                        if trunc_tries > MAX_TRUNCATION_RETRIES:
                            infra_truncated = True
                            break
                        continue  # re-request; no feedback, no chess attempt consumed
                    if pr.parse_class == "legal":
                        san_history.append(pr.move_san)
                        board.push_san(pr.move_san)
                        played = True
                        break
                    messages.append({"role": "assistant", "content": resp.text})
                    messages.append({"role": "user", "content": retry_feedback(pr)})
                if infra_truncated:
                    # Persistent truncation is a harness/budget problem, not a
                    # chess result: censor the game, never record a loss.
                    termination = "llm_truncated"
                    break
                if not played:
                    # Forfeit: failed to produce a legal move within the
                    # attempt budget. The class mix is recorded because a
                    # pure-format forfeit (all "invalid") is NOT a survival
                    # event, while an illegal/ambiguous one is.
                    termination = "llm_forfeit"
                    forfeit_classes = ply_classes
                    winner = "engine"
                    result_str = "0-1" if llm_is_white else "1-0"
                    break
            else:
                move = engine.play(board, game=spec.game_id)
                san = board.san(move)
                emit(
                    {
                        "type": "engine_move",
                        "game_id": spec.game_id,
                        "ply": ply,
                        "move_san": san,
                        "move_uci": board.uci(move),
                        "fen_before": board_fen(board),
                        "ts": time.time(),
                    }
                )
                san_history.append(san)
                board.push(move)

        if termination == "llm_forfeit":
            llm_result = "loss_forfeit"
        elif termination == "llm_truncated":
            llm_result = "censored_infra"
        elif termination == "move_cap":
            llm_result = "censored_cap"
        elif winner == "llm":
            llm_result = "win"
        elif winner == "engine":
            llm_result = "loss"
        else:
            llm_result = "draw"

        # Survival framing (PLAN.md D4): event = an illegal-or-ambiguous
        # attempt occurred; otherwise censored at the plies actually reached.
        event = first_event is not None
        game_record = {
            "type": "game",
            "game_id": spec.game_id,
            "model": spec.model,
            "variant": spec.variant,
            "visibility": spec.visibility,
            "sp_id": spec.sp_id,
            "llm_color": spec.llm_color,
            "game_index": spec.game_index,
            "max_plies": spec.max_plies,
            "max_attempts": spec.max_attempts,
            "temperature": spec.temperature,
            "effective_temperature": getattr(llm, "effective_temperature", None),
            "prompt_version": PROMPT_VERSION,
            "parser_version": PARSER_VERSION,
            "engine_name": getattr(engine, "name", "unknown"),
            "engine_skill_level": getattr(getattr(engine, "cfg", None), "skill_level", None),
            "engine_nodes": getattr(getattr(engine, "cfg", None), "nodes", None),
            "start_fen": start_fen,
            "final_fen": board_fen(board),
            "plies": board.ply(),
            "n_llm_moves_requested": llm_move_index,
            "san_history": " ".join(san_history),
            "counts": counts,
            "first_event_ply": first_event[0] if first_event else None,
            "first_event_llm_move": first_event[1] if first_event else None,
            "first_illegal_ply": first_illegal[0] if first_illegal else None,
            "first_illegal_llm_move": first_illegal[1] if first_illegal else None,
            "first_ambiguous_ply": first_ambiguous[0] if first_ambiguous else None,
            "first_ambiguous_llm_move": first_ambiguous[1] if first_ambiguous else None,
            "first_failure_ply": first_failure[0] if first_failure else None,
            "first_failure_llm_move": first_failure[1] if first_failure else None,
            "event": event,
            "survival_plies": first_event[0] if event else board.ply(),
            # Primary survival time scale per the pre-registration: LLM move
            # index (each model move = one trial; unaffected by color or the
            # offbook prefix, unlike ply).
            "survival_llm_moves": first_event[1] if event else llm_move_index,
            "prefix_plies": len(spec.opening_prefix),
            "opening_prefix": " ".join(spec.opening_prefix) or None,
            "prefix_id": spec.prefix_id,
            "forfeit_attempt_classes": forfeit_classes,
            "termination": termination,
            "winner": winner,
            "result": result_str,
            "llm_result": llm_result,
            "ts_start": ts_start,
            "ts_end": time.time(),
            "duration_s": round(time.time() - ts_start, 2),
        }
        # PGN before the completion record: the game only counts as done
        # (game_done in run.py) once both artifacts exist, so a crash between
        # the two writes gets replayed on resume instead of losing the PGN.
        _write_pgn(spec, board, result_str, getattr(engine, "name", "engine"), out_dir)
        emit(game_record)

    return game_record


def _write_pgn(spec: GameSpec, board: chess.Board, result_str: str, engine_name: str, out_dir: Path) -> None:
    game = chess.pgn.Game.from_board(board)
    llm_is_white = spec.llm_color == "white"
    game.headers["Event"] = f"chessbench {spec.variant} / {spec.visibility}"
    game.headers["Site"] = spec.game_id
    game.headers["White"] = spec.model if llm_is_white else engine_name
    game.headers["Black"] = engine_name if llm_is_white else spec.model
    game.headers["Result"] = result_str
    (out_dir / f"{spec.game_id}.pgn").write_text(str(game) + "\n", encoding="utf-8")
