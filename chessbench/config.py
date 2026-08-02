from __future__ import annotations

from dataclasses import dataclass

VISIBILITIES = ("history-only", "history+board")
# "standard-offbook" = standard rules/geometry with a seeded random opening
# prefix — the bridge condition separating "off-book" from "off-geometry"
# (PLAN.md pre-registration).
VARIANTS = ("standard", "chess960", "standard-offbook")

# Scharnagl number of the standard chess starting position.
STANDARD_SP_ID = 518

# Bump whenever the prompt templates change in a way that affects comparability.
PROMPT_VERSION = "2"

# Bump whenever move-extraction rules change (e.g. the fallback extractor).
PARSER_VERSION = "3"


@dataclass(frozen=True)
class EngineConfig:
    path: str = "stockfish"
    # Low skill + node cap => weak but reproducible opponent and long games
    # (long games = long exposure windows for the survival metric).
    skill_level: int = 3
    nodes: int = 20_000
    # Depth for the white-POV eval logged before each LLM move; 0 disables.
    eval_depth: int = 10


@dataclass(frozen=True)
class GameSpec:
    game_id: str
    model: str
    variant: str  # "standard" | "chess960"
    visibility: str  # "history-only" | "history+board"
    sp_id: int  # Scharnagl start-position number; 518 for standard
    llm_color: str  # "white" | "black"
    game_index: int  # index of this game within its condition cell
    # standard-offbook only: the seeded random opening prefix (SAN), played
    # onto the board before the game loop; empty for other variants.
    opening_prefix: tuple[str, ...] = ()
    prefix_id: int | None = None  # index into the drawn prefix set
    # Appends qwen-style "/no_think" to the system prompt, disabling hybrid
    # thinking. A per-model operating parameter (recorded per game), not a
    # benchmark condition change.
    no_think: bool = False
    max_plies: int = 200
    max_attempts: int = 3  # attempts per ply before loss-by-illegality
    temperature: float = 0.0
