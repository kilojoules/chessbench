from __future__ import annotations

import chess

from .config import GameSpec
from .parsing import ParseResult


def start_board(variant: str, sp_id: int) -> chess.Board:
    if variant == "chess960":
        return chess.Board.from_chess960_pos(sp_id)
    return chess.Board()  # "standard" and "standard-offbook"


def board_fen(board: chess.Board) -> str:
    # Shredder-FEN spells castling rights as rook files (e.g. "HFhf"),
    # which is unambiguous for Chess960; plain fen() emits X-FEN "KQkq".
    return board.shredder_fen() if board.chess960 else board.fen()


def ascii_board(board: chess.Board) -> str:
    rows = str(board).splitlines()
    lines = [f"{8 - i} | {row}" for i, row in enumerate(rows)]
    lines.append("  +----------------")
    lines.append("    a b c d e f g h")
    return "\n".join(lines)


def movetext(san_history: list[str]) -> str:
    parts = []
    for i in range(0, len(san_history), 2):
        num = i // 2 + 1
        chunk = f"{num}. {san_history[i]}"
        if i + 1 < len(san_history):
            chunk += f" {san_history[i + 1]}"
        parts.append(chunk)
    return " ".join(parts)


def system_prompt(spec: GameSpec) -> str:
    color = "White" if spec.llm_color == "white" else "Black"
    if spec.variant == "chess960":
        variant_par = (
            "This is a game of Chess960 (Fischer Random chess). The back-rank pieces "
            "started on randomized squares; the exact starting position is given in each "
            "move request. All normal chess rules apply. Castling follows Chess960 rules: "
            "castling kingside puts your king on g1 and your rook on f1 (g8 and f8 for "
            "Black); castling queenside puts your king on c1 and your rook on d1 (c8 and "
            "d8 for Black), regardless of where king and rook started. Write castling in "
            "SAN as O-O or O-O-O."
        )
    elif spec.variant == "standard-offbook":
        variant_par = (
            "This is a game of standard chess, played from the normal starting position. "
            "The opening was randomized: the first few moves for BOTH sides were played by "
            "a neutral random move generator (they appear in the move history), and the "
            "game continues from the resulting position. All normal chess rules apply."
        )
    else:
        variant_par = "This is a game of standard chess, played from the normal starting position."
    return f"""You are playing a game of chess against an opponent. {variant_par}

You are playing {color}.

On each turn, decide on your move and reply with it in standard algebraic notation (SAN). You may reason briefly first, but the final line of your reply must have exactly this format:
MOVE: <your move>

Examples: "MOVE: e4", "MOVE: Nf3", "MOVE: O-O", "MOVE: exd8=Q+".

If a move could be made by more than one of your pieces, disambiguate it with the originating file or rank (e.g. "MOVE: Nbd2", "MOVE: R1e2").

Only legal moves are accepted."""


def move_request(spec: GameSpec, board: chess.Board, start_fen: str, san_history: list[str]) -> str:
    color = "White" if board.turn == chess.WHITE else "Black"
    lines = [f"Starting position (FEN): {start_fen}"]
    if san_history:
        lines.append(f"Moves played so far: {movetext(san_history)}")
    else:
        lines.append("Moves played so far: (none — the game is just starting)")
    if spec.visibility == "history+board":
        lines.append("")
        lines.append(f"Current position (FEN): {board_fen(board)}")
        lines.append("Current board (uppercase = White, lowercase = Black):")
        lines.append(ascii_board(board))
    lines.append("")
    lines.append(
        f"It is your move — you are {color} (move {board.fullmove_number}). "
        "Reply with your move, ending with the line 'MOVE: <san>'."
    )
    return "\n".join(lines)


def retry_feedback(pr: ParseResult) -> str:
    # Deliberately minimal and identical across conditions: feedback must not
    # leak board state into the history-only (blindfold) condition. The
    # ambiguous branch must not confirm the move is legal for 2+ pieces —
    # that is genuine current-board information; the conditional phrasing
    # (plus the standing rule in the system prompt) leaks nothing.
    if pr.parse_class == "invalid":
        first = "Your reply did not contain a syntactically valid chess move."
    elif pr.parse_class == "ambiguous":
        first = (
            f"Your move '{pr.candidate}' was not accepted in the current position. "
            "If it could be made by more than one of your pieces, write it fully "
            "disambiguated (e.g. 'Nbd2' or 'R1e2')."
        )
    else:  # illegal
        first = f"Your move '{pr.candidate}' is illegal in the current position."
    return f"{first} Reply again with a legal move, ending with the line 'MOVE: <san>'."
