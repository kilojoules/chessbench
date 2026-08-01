import chess

from chessbench.config import STANDARD_SP_ID
from chessbench.positions import draw_chess960_positions


def test_draw_is_deterministic():
    assert draw_chess960_positions(10, 2026) == draw_chess960_positions(10, 2026)


def test_draw_excludes_standard_start():
    positions = draw_chess960_positions(200, 1)
    assert STANDARD_SP_ID not in positions


def test_draw_properties():
    positions = draw_chess960_positions(10, 2026)
    assert len(positions) == len(set(positions)) == 10
    assert positions == sorted(positions)
    assert all(0 <= p <= 959 for p in positions)


def test_scharnagl_518_is_standard():
    assert chess.Board.from_chess960_pos(518).fen() == chess.STARTING_FEN


def test_chess960_castling_king_takes_rook_encoding():
    # King b1, rooks a1/e1 with Shredder castling rights; O-O must be legal
    # and encode as king-takes-rook UCI. (No black piece may attack the
    # king's b1->g1 path, so black holds only a far-away king.)
    board = chess.Board("1k6/8/8/8/8/8/8/RK2R3 w AE - 0 1", chess960=True)
    move = board.parse_san("O-O")
    assert board.uci(move) == "b1e1"
