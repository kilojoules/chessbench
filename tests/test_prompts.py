import chess

from chessbench.config import GameSpec
from chessbench.prompts import (
    ascii_board,
    board_fen,
    move_request,
    movetext,
    start_board,
    system_prompt,
)


def make_spec(variant="standard", visibility="history-only", sp_id=518, color="white"):
    return GameSpec(
        game_id="test",
        model="fake:first",
        variant=variant,
        visibility=visibility,
        sp_id=sp_id,
        llm_color=color,
        game_index=0,
    )


def played_board(sans):
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def test_movetext_pairing():
    assert movetext(["e4", "e5", "Nf3"]) == "1. e4 e5 2. Nf3"
    assert movetext([]) == ""


def test_blindfold_prompt_leaks_no_current_board():
    spec = make_spec(visibility="history-only")
    board = played_board(["e4", "e5", "Nf3"])
    start_fen = chess.Board().fen()
    prompt = move_request(spec, board, start_fen, ["e4", "e5", "Nf3"])
    assert "Current position" not in prompt
    assert "Current board" not in prompt
    # The current piece placement must not appear anywhere.
    assert board.fen().split()[0] not in prompt
    # But start FEN and history must.
    assert start_fen in prompt
    assert "1. e4 e5 2. Nf3" in prompt


def test_board_visibility_includes_fen_and_diagram():
    spec = make_spec(visibility="history+board")
    board = played_board(["e4", "e5"])
    prompt = move_request(spec, board, chess.Board().fen(), ["e4", "e5"])
    assert f"Current position (FEN): {board.fen()}" in prompt
    assert "a b c d e f g h" in prompt


def test_chess960_uses_shredder_fen():
    board = start_board("chess960", 0)
    assert board.chess960
    fen = board_fen(board)
    assert fen == board.shredder_fen()
    assert "HF" in fen  # rook-file castling rights, not KQkq


def test_standard_board_is_plain():
    board = start_board("standard", 518)
    assert not board.chess960
    assert board_fen(board) == chess.STARTING_FEN


def test_system_prompt_mentions_variant_and_color():
    sp = system_prompt(make_spec(variant="chess960", color="black"))
    assert "Chess960" in sp
    assert "playing Black" in sp
    sp_std = system_prompt(make_spec(variant="standard"))
    assert "standard chess" in sp_std
    assert "Chess960" not in sp_std
    assert "MOVE:" in sp


def test_ascii_board_shape():
    art = ascii_board(chess.Board())
    lines = art.splitlines()
    assert lines[0].startswith("8 |")
    assert lines[7].startswith("1 |")
    assert "a b c d e f g h" in lines[-1]


def test_move_request_states_side_to_move():
    spec = make_spec(color="black")
    board = played_board(["e4"])
    prompt = move_request(spec, board, chess.Board().fen(), ["e4"])
    assert "you are Black" in prompt
