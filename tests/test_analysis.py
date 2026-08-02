import chess
import pytest

pytest.importorskip("lifelines")
pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from chessbench.analysis import (  # noqa: E402
    classify_illegal,
    games_frame,
    load_run_dirs,
    main,
    phantom_standard,
)
from chessbench.config import GameSpec  # noqa: E402
from chessbench.game import play_game  # noqa: E402
from chessbench.llm import FakeLLM  # noqa: E402


class StubEngine:
    name = "stub-engine"

    def play(self, board, game=None):
        return next(iter(board.legal_moves))

    def eval_cp_white(self, board):
        return None


def _make_run(tmp_path, n=4):
    """Games across two cells with events at varied moves."""
    i = 0
    for variant in ("standard", "chess960"):
        for vis in ("history-only", "history+board"):
            for g in range(n):
                i += 1
                spec = GameSpec(
                    game_id=f"g{i:02d}", model="fake:first", variant=variant,
                    visibility=vis, sp_id=518 if variant == "standard" else 105,
                    llm_color="white" if g % 2 == 0 else "black", game_index=g,
                    max_plies=16, max_attempts=3,
                )
                llm = FakeLLM(policy="first", illegal_at={(g % 3) + 1} if g % 2 == 0 else frozenset())
                play_game(spec, llm, StubEngine(), tmp_path)


def test_end_to_end_report(tmp_path):
    _make_run(tmp_path)
    out = tmp_path / "analysis"
    main([str(tmp_path), "-o", str(out)])
    report = (out / "report.md").read_text()
    assert (out / "km.png").exists()
    assert (out / "hazard.png").exists()
    assert "Per-cell summary" in report
    assert "error taxonomy" in report.lower()
    assert "Sensitivity" in report


def test_games_frame_survival_columns(tmp_path):
    _make_run(tmp_path, n=2)
    games, illegals, overflow = load_run_dirs([tmp_path])
    assert sum(overflow.values()) == 0  # FakeLLM never overflows
    df = games_frame(games)
    assert set(["T", "E", "blind", "v960", "black", "unit"]).issubset(df.columns)
    assert (df["T"] > 0).all()
    assert df["E"].sum() >= 1
    assert len(illegals) >= 1
    # events recorded on the LLM-move-index scale
    ev = df[df["E"] == 1]
    assert (ev["T"] <= 8).all()


def test_classify_illegal_piece_cannot_reach():
    # Ke5 from the start: king cannot reach e5
    assert classify_illegal(chess.STARTING_FEN, "Ke5", False) == "piece-cannot-reach"


def test_classify_illegal_castling():
    assert classify_illegal(chess.STARTING_FEN, "O-O", False) == "illegal-castling"


def test_classify_illegal_own_capture():
    # Rxa2/Ra2: rook a1 "capturing" own pawn a2
    assert classify_illegal(chess.STARTING_FEN, "Ra2", False) == "own-piece-capture"


def test_classify_illegal_into_check():
    # King in check from rook; moving another piece = pseudo-legal but illegal
    fen = "k7/8/8/8/8/8/r7/K2N4 w - - 0 1"  # black rook a2 gives check
    assert classify_illegal(fen, "Nf2", False) == "into-check-or-pin"


def test_phantom_standard():
    # In a 960 game with empty history, e4 is legal from the standard start
    assert phantom_standard([], "e4") is True
    assert phantom_standard([], "Ke3") is False
