from __future__ import annotations

import re
from dataclasses import dataclass

import chess

# The protocol: the model's reply must end with a line "MOVE: <san>".
# Primary extraction is LINE-ANCHORED (a MOVE line, last one wins) so prose
# that merely quotes the instructions ("the format MOVE: <san> is required")
# cannot shadow the real answer. Tolerates markdown around the label.
MOVE_LINE_RE = re.compile(r"^[ \t>*_`#-]*MOVE[*_`]*\s*:[ \t*_`]*(\S+)", re.IGNORECASE | re.MULTILINE)
# Secondary: inline mention, only adopted if the token is SAN-shaped (see
# classify flow). The lookbehind keeps words like "REMOVE:" from matching.
MOVE_RE = re.compile(r"[*_`]*(?<![A-Za-z])MOVE[*_`]*\s*:[\s*_`]*(\S+)", re.IGNORECASE)
# Math-mode declared answers: $$ \boxed{Nf3} $$ (an RL-era habit).
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")

# Mild normalization so trivial formatting (markdown, trailing punctuation,
# "12.e4"-style prefixes) doesn't get graded as a chess failure. Anything
# beyond this is the model's problem and lands in the taxonomy below.
_WRAP_CHARS = "`*_\"'<>()[]"
_TRAIL_CHARS = ".,;:!?" + _WRAP_CHARS
_MOVENUM_PREFIX_RE = re.compile(r"^\d+\.+")
_CASTLE_RE = re.compile(r"(?i)^[0oO]-[0oO](-[0oO])?([+#])?$")

# Fallback for protocol non-compliance (small models end with "**e4**" or
# "**Answer:** a3" instead of "MOVE: e4"): accept the last non-empty line iff,
# after stripping markdown and an optional answer label, it is a SINGLE
# SAN-shaped token. Prose lines never match, so this cannot scrape moves out
# of running text. Tagged extraction="fallback" so strict-protocol analysis
# can exclude these.
_FALLBACK_LABEL_RE = re.compile(
    r"^(final answer|best move|answer|move)[\s*_`]*[:\-]*[\s*_`]*", re.IGNORECASE
)
# SAN plus tolerated long-algebraic forms ("e2e4", "d2-d4" — python-chess
# parses both).
_SAN_SHAPE_RE = re.compile(
    r"^(O-O(-O)?|0-0(-0)?|[KQRBN]?[a-h]?[1-8]?[-x]?[a-h][1-8](=?[QRBNqrbn])?)[+#]?$"
)


@dataclass
class ParseResult:
    """Failure taxonomy (PLAN.md D4), mapping 1:1 onto python-chess exceptions:

    - "legal":     parsed and legal; move_san/move_uci hold canonical forms
    - "illegal":   well-formed SAN but illegal here (IllegalMoveError) — the
                   primary survival event
    - "ambiguous": SAN matching several legal moves (AmbiguousMoveError)
    - "invalid":   no MOVE line, or syntactically invalid (InvalidMoveError)
    """

    parse_class: str
    extracted: str | None  # raw token captured after "MOVE:"
    candidate: str | None  # normalized candidate handed to the SAN parser
    move_san: str | None
    move_uci: str | None
    error: str | None
    extraction: str | None = None  # "protocol" | "fallback" | None


def _clean_token(tok: str) -> str | None:
    cand = tok.strip(_WRAP_CHARS)
    cand = _MOVENUM_PREFIX_RE.sub("", cand)
    cand = cand.rstrip(_TRAIL_CHARS).strip()
    if _CASTLE_RE.match(cand):
        cand = cand.upper().replace("0", "O")
    return cand or None


def extract_candidate(raw_output: str) -> tuple[str | None, str | None]:
    """Protocol stage: last line-anchored 'MOVE: <token>' line."""
    matches = MOVE_LINE_RE.findall(raw_output or "")
    if not matches:
        return None, None
    return matches[-1], _clean_token(matches[-1])


def fallback_candidate(raw_output: str) -> str | None:
    """Last non-empty line, iff (after markdown/label stripping) it is a
    single SAN-shaped token. Prose can never match."""
    lines = [ln.strip() for ln in (raw_output or "").splitlines() if ln.strip()]
    if not lines:
        return None
    line = lines[-1].strip(_WRAP_CHARS + " \t")
    line = _FALLBACK_LABEL_RE.sub("", line)
    line = line.strip(_WRAP_CHARS + " \t").rstrip(".,;:!?").strip(_WRAP_CHARS)
    if not _SAN_SHAPE_RE.match(line):
        return None
    if _CASTLE_RE.match(line):
        line = line.upper().replace("0", "O")
    return line


def find_candidate(raw_output: str) -> tuple[str | None, str | None, str | None]:
    """Returns (extracted, candidate, extraction) via staged extraction:
    1. "protocol":  last line-anchored MOVE line (adopted unconditionally —
       a garbage MOVE line is the model's declared answer and stays invalid);
    2. "protocol-inline": last inline MOVE mention, only if SAN-shaped
       (prose quoting the instructions must not win);
    3. "fallback": last \\boxed{...} declared answer, if SAN-shaped;
    4. "fallback": bare SAN-shaped final line.
    """
    raw = raw_output or ""
    extracted, cand = extract_candidate(raw)
    if extracted is not None:
        return extracted, cand, "protocol"
    inline = MOVE_RE.findall(raw)
    if inline:
        cand = _clean_token(inline[-1])
        if cand and _SAN_SHAPE_RE.match(cand):
            return inline[-1], cand, "protocol-inline"
    boxed = BOXED_RE.findall(raw)
    if boxed:
        cand = _clean_token(boxed[-1])
        if cand and _SAN_SHAPE_RE.match(cand):
            return boxed[-1], cand, "fallback"
    cand = fallback_candidate(raw)
    if cand is not None:
        return None, cand, "fallback"
    return None, None, None


def classify_move(board: chess.Board, raw_output: str) -> ParseResult:
    extracted, cand, extraction = find_candidate(raw_output)
    if cand is None:
        return ParseResult("invalid", extracted, None, None, None, "no MOVE line found")
    try:
        move = board.parse_san(cand)
    except chess.AmbiguousMoveError as e:
        return ParseResult("ambiguous", extracted, cand, None, None, str(e), extraction)
    except chess.IllegalMoveError as e:
        return ParseResult("illegal", extracted, cand, None, None, str(e), extraction)
    except chess.InvalidMoveError as e:
        return ParseResult("invalid", extracted, cand, None, None, str(e), extraction)
    if not bool(move):  # "--" parses as a null move; not a chess move
        return ParseResult("invalid", extracted, cand, None, None, "null move", extraction)
    return ParseResult("legal", extracted, cand, board.san(move), board.uci(move), None, extraction)
