"""Integration tests that need a real Stockfish binary; skipped if absent."""

import shutil

import chess
import pytest

from chessbench.config import EngineConfig, GameSpec
from chessbench.engine import Engine
from chessbench.game import play_game
from chessbench.llm import FakeLLM

pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish not installed"
)


@pytest.fixture(scope="module")
def engine():
    with Engine(EngineConfig(nodes=500, skill_level=3, eval_depth=6)) as e:
        yield e


def test_engine_plays_chess960(engine):
    board = chess.Board.from_chess960_pos(0)
    for _ in range(20):
        if board.is_game_over():
            break
        move = engine.play(board, game="it960")
        assert move in board.legal_moves
        board.push(move)
    assert board.ply() > 0


def test_eval_returns_centipawns(engine):
    cp = engine.eval_cp_white(chess.Board())
    assert isinstance(cp, int)
    assert -200 < cp < 200  # start position is near equal


def test_full_game_fake_llm_vs_stockfish(tmp_path, engine):
    spec = GameSpec(
        game_id="it-full",
        model="fake:random",
        variant="chess960",
        visibility="history+board",
        sp_id=402,
        llm_color="black",
        game_index=0,
        max_plies=40,
        max_attempts=3,
    )
    rec = play_game(spec, FakeLLM(policy="random", seed=9), engine, tmp_path)
    assert rec["termination"] is not None
    assert rec["counts"]["illegal"] == 0
    assert rec["engine_name"].startswith("Stockfish")
