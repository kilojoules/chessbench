import chess

from chessbench.animate import _event_ghost, _ghost_move


def test_ghost_executes_illegal_king_move():
    # Ke5 from the start: king e1 lifted to e5, both squares red.
    fen, frm, to = _ghost_move(chess.STARTING_FEN, "Ke5", True)
    assert (frm, to) == ("e1", "e5")
    board = chess.Board(fen, chess960=True)
    assert board.piece_at(chess.E1) is None
    assert board.piece_at(chess.E5).symbol() == "K"


def test_ghost_pawn_push():
    # d5 as White from the start: the d2 pawn is the intended mover.
    fen, frm, to = _ghost_move(chess.STARTING_FEN, "d5", True)
    assert (frm, to) == ("d2", "d5")


def test_ghost_phantom_standard_in_960():
    # Nf3 in a 960 start with no knight that reaches f3: nearest knight
    # is ghosted there anyway — visualizing the imagined move.
    start = chess.Board.from_chess960_pos(105)  # QNRBBNKR: knights b1, f1
    fen, frm, to = _ghost_move(start.shredder_fen(), "Nf3", True)
    assert to == "f3"
    assert frm in ("b1", "f1")
    assert chess.Board(fen, chess960=True).piece_at(chess.F3).piece_type == chess.KNIGHT


def test_ghost_castling_falls_back():
    assert _ghost_move(chess.STARTING_FEN, "O-O", True) is None
    fen, red = _event_ghost(chess.STARTING_FEN,
                            [{"candidate": "O-O", "class": "illegal"}], True)
    assert fen == chess.STARTING_FEN  # unghostable: position unchanged
    assert red == set()  # castling contributes no squares


def test_event_ghost_red_squares():
    fen, red = _event_ghost(chess.STARTING_FEN,
                            [{"candidate": "Ke5", "class": "illegal"}], True)
    assert red == {"e1", "e5"}
