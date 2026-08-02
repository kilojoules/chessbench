from __future__ import annotations

import random

import chess

from .config import STANDARD_SP_ID


def draw_offbook_prefixes(n: int = 10, plies: int = 6, seed: int = 2027) -> list[list[str]]:
    """Draw the fixed set of random opening prefixes for standard-offbook.

    Each prefix is `plies` uniformly random legal moves from the standard
    start (SAN), seeded so the same set is reused across all cells and
    models. Prefixes that reach a game-over position are redrawn.
    """
    rng = random.Random(seed)
    prefixes: list[list[str]] = []
    while len(prefixes) < n:
        board = chess.Board()
        sans: list[str] = []
        for _ in range(plies):
            if board.is_game_over():
                break
            move = rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
            sans.append(board.san(move))
            board.push(move)
        if len(sans) == plies and not board.is_game_over():
            prefixes.append(sans)
    return prefixes


def draw_chess960_positions(n: int = 10, seed: int = 2026) -> list[int]:
    """Draw the fixed set of Chess960 start positions used across all cells.

    The same seeded draw is reused for every model and both visibility
    conditions so games are paired on start position. #518 (the standard
    start) is excluded so the chess960 condition is always off-book.
    """
    rng = random.Random(seed)
    pool = [i for i in range(960) if i != STANDARD_SP_ID]
    return sorted(rng.sample(pool, n))
