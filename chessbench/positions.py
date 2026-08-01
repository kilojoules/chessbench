from __future__ import annotations

import random

from .config import STANDARD_SP_ID


def draw_chess960_positions(n: int = 10, seed: int = 2026) -> list[int]:
    """Draw the fixed set of Chess960 start positions used across all cells.

    The same seeded draw is reused for every model and both visibility
    conditions so games are paired on start position. #518 (the standard
    start) is excluded so the chess960 condition is always off-book.
    """
    rng = random.Random(seed)
    pool = [i for i in range(960) if i != STANDARD_SP_ID]
    return sorted(rng.sample(pool, n))
