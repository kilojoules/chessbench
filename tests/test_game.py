import json

import chess
import pytest

from chessbench.config import GameSpec
from chessbench.game import play_game
from chessbench.llm import FakeLLM, LLMResponse, _illegal_san
from chessbench.run import game_done


class StubEngine:
    """Plays the first legal move; no Stockfish needed."""

    name = "stub-engine"

    def play(self, board, game=None):
        return next(iter(board.legal_moves))

    def eval_cp_white(self, board):
        return None


def make_spec(tmp_id, **kw):
    defaults = dict(
        game_id=tmp_id,
        model="fake:first",
        variant="standard",
        visibility="history-only",
        sp_id=518,
        llm_color="white",
        game_index=0,
        max_plies=30,
        max_attempts=3,
    )
    defaults.update(kw)
    return GameSpec(**defaults)


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_clean_game_runs_to_completion(tmp_path):
    spec = make_spec("clean")
    rec = play_game(spec, FakeLLM(policy="random", seed=42), StubEngine(), tmp_path)
    assert rec["event"] is False
    assert rec["first_illegal_ply"] is None
    assert rec["counts"]["illegal"] == 0
    assert rec["termination"] is not None
    assert rec["survival_plies"] == rec["plies"]
    records = read_records(tmp_path / "clean.jsonl")
    assert records[-1]["type"] == "game"
    attempts = [r for r in records if r["type"] == "attempt"]
    assert all(r["parse_class"] == "legal" for r in attempts)
    assert (tmp_path / "clean.pgn").exists()


def test_always_illegal_loses_at_first_move(tmp_path):
    spec = make_spec("lose")
    rec = play_game(spec, FakeLLM(policy="always-illegal"), StubEngine(), tmp_path)
    assert rec["termination"] == "llm_forfeit"
    assert rec["llm_result"] == "loss_forfeit"
    assert rec["forfeit_attempt_classes"] == ["illegal", "illegal", "illegal"]
    assert rec["winner"] == "engine"
    assert rec["result"] == "0-1"  # LLM was white and lost
    assert rec["event"] is True
    assert rec["first_illegal_ply"] == 1
    assert rec["first_illegal_llm_move"] == 1
    assert rec["counts"]["illegal"] == 3  # max_attempts exhausted
    attempts = read_records(tmp_path / "lose.jsonl")[:-1]
    assert [a["attempt"] for a in attempts] == [1, 2, 3]


def test_retry_recovery_continues_game(tmp_path):
    spec = make_spec("recover", max_plies=12)
    llm = FakeLLM(policy="first", illegal_at={2})
    rec = play_game(spec, llm, StubEngine(), tmp_path)
    assert rec["termination"] != "llm_forfeit"
    assert rec["event"] is True
    # LLM is white: its 2nd move is game ply 3.
    assert rec["first_illegal_ply"] == 3
    assert rec["first_illegal_llm_move"] == 2
    assert rec["survival_plies"] == 3
    assert rec["survival_llm_moves"] == 2
    assert rec["counts"]["illegal"] == 1
    # The failed ply has two attempts: illegal then legal.
    attempts = [r for r in read_records(tmp_path / "recover.jsonl") if r["type"] == "attempt"]
    ply3 = [a for a in attempts if a["ply"] == 3]
    assert [a["parse_class"] for a in ply3] == ["illegal", "legal"]


def test_black_llm_ply_numbering(tmp_path):
    spec = make_spec("black", llm_color="black", max_plies=10)
    llm = FakeLLM(policy="first", illegal_at={1})
    rec = play_game(spec, llm, StubEngine(), tmp_path)
    # LLM is black: its 1st move is game ply 2.
    assert rec["first_illegal_ply"] == 2
    assert rec["first_illegal_llm_move"] == 1


def test_chess960_game_runs(tmp_path):
    spec = make_spec("c960", variant="chess960", sp_id=0, max_plies=20)
    rec = play_game(spec, FakeLLM(policy="random", seed=7), StubEngine(), tmp_path)
    assert rec["event"] is False
    assert rec["start_fen"].endswith("w HFhf - 0 1")
    pgn_text = (tmp_path / "c960.pgn").read_text()
    assert "FEN" in pgn_text  # non-standard start recorded in headers


def test_move_cap_censors(tmp_path):
    spec = make_spec("cap", max_plies=6)
    rec = play_game(spec, FakeLLM(policy="random", seed=1), StubEngine(), tmp_path)
    assert rec["termination"] == "move_cap"
    assert rec["llm_result"] == "censored_cap"
    assert rec["plies"] == 6
    assert rec["event"] is False


def test_game_done_detection(tmp_path):
    spec = make_spec("done")
    play_game(spec, FakeLLM(policy="random", seed=3), StubEngine(), tmp_path)
    assert game_done(tmp_path / "done.jsonl")
    assert not game_done(tmp_path / "missing.jsonl")
    # An incomplete file (no game record) is not done.
    (tmp_path / "partial.jsonl").write_text('{"type": "attempt"}\n')
    assert not game_done(tmp_path / "partial.jsonl")
    # A game whose PGN is missing must be replayed.
    (tmp_path / "done.pgn").unlink()
    assert not game_done(tmp_path / "done.jsonl")


def _ambiguous_960_sp():
    """A Chess960 start where 'Nc3' is ambiguous on move 1 (knights on b1+d1)."""
    for n in range(960):
        board = chess.Board.from_chess960_pos(n)
        try:
            board.parse_san("Nc3")
        except chess.AmbiguousMoveError:
            return n
        except (chess.IllegalMoveError, chess.InvalidMoveError):
            continue
    pytest.fail("no ambiguous 960 start found")


def test_ambiguous_counts_as_survival_event(tmp_path):
    sp = _ambiguous_960_sp()
    spec = make_spec("ambig", variant="chess960", sp_id=sp, max_plies=2)
    llm = FakeLLM(script=["MOVE: Nc3", "MOVE: a3"])
    rec = play_game(spec, llm, StubEngine(), tmp_path)
    assert rec["event"] is True
    assert rec["first_event_ply"] == 1
    assert rec["first_ambiguous_ply"] == 1
    assert rec["first_illegal_ply"] is None
    assert rec["survival_plies"] == 1
    assert rec["counts"]["ambiguous"] == 1
    assert rec["termination"] == "move_cap"


class TruncatingLLM:
    """Returns finish_reason='length' for the first n calls, then plays legally."""

    def __init__(self, n_truncations):
        self.n = n_truncations

    def complete(self, messages, board=None):
        if self.n > 0:
            self.n -= 1
            return LLMResponse("I was thinking about", 0, 0, 0, finish_reason="length")
        san = sorted(board.san(m) for m in board.legal_moves)[0]
        return LLMResponse(f"MOVE: {san}", 0, 0, 0, finish_reason="stop")


def test_truncation_is_not_a_chess_failure(tmp_path):
    spec = make_spec("trunc-ok", max_plies=4)
    rec = play_game(spec, TruncatingLLM(2), StubEngine(), tmp_path)
    assert rec["termination"] == "move_cap"
    assert rec["counts"]["truncated"] == 2
    assert rec["counts"]["invalid"] == 0
    assert rec["event"] is False
    assert rec["first_failure_ply"] is None
    attempts = [r for r in read_records(tmp_path / "trunc-ok.jsonl") if r["type"] == "attempt"]
    truncs = [a for a in attempts if a["parse_class"] == "truncated"]
    assert [t["trunc_try"] for t in truncs] == [1, 2]
    assert all(t["finish_reason"] == "length" for t in truncs)


def test_offbook_game_plays_prefix_first(tmp_path):
    from chessbench.positions import draw_offbook_prefixes
    from chessbench.prompts import move_request, system_prompt

    prefix = tuple(draw_offbook_prefixes(1, 6, 2027)[0])
    spec = make_spec("offbook", variant="standard-offbook", opening_prefix=prefix,
                     prefix_id=0, max_plies=8)
    rec = play_game(spec, FakeLLM(policy="random", seed=11), StubEngine(), tmp_path)
    assert rec["prefix_plies"] == 6
    assert rec["opening_prefix"] == " ".join(prefix)
    assert rec["san_history"].startswith(" ".join(prefix))
    # Exposure cap excludes the prefix: 8 playable plies after 6 prefix plies.
    assert rec["plies"] == 14 or rec["termination"] != "move_cap"
    records = read_records(tmp_path / "offbook.jsonl")
    assert [r["move_san"] for r in records if r["type"] == "prefix_move"] == list(prefix)
    # LLM exposure starts after the prefix; survival time scale unaffected.
    first_attempt = next(r for r in records if r["type"] == "attempt")
    assert first_attempt["ply"] == 7
    assert first_attempt["llm_move_index"] == 1
    # Prompts: prefix moves appear in history; system prompt explains the setup.
    assert "randomized" in system_prompt(spec)
    import chess as _c
    board = _c.Board()
    for san in prefix:
        board.push_san(san)
    prompt = move_request(spec, board, _c.STARTING_FEN, list(prefix))
    assert prefix[0] in prompt


def test_persistent_truncation_censors_not_loses(tmp_path):
    spec = make_spec("trunc-bad")
    rec = play_game(spec, TruncatingLLM(99), StubEngine(), tmp_path)
    assert rec["termination"] == "llm_truncated"
    assert rec["llm_result"] == "censored_infra"
    assert rec["winner"] is None
    assert rec["result"] == "*"
    assert rec["counts"]["truncated"] == 3  # initial call + MAX_TRUNCATION_RETRIES
    assert rec["counts"]["illegal"] == 0
    assert rec["event"] is False


def test_illegal_san_helper_is_wellformed_illegal():
    board = chess.Board()
    san = _illegal_san(board)
    try:
        board.parse_san(san)
        raise AssertionError("expected IllegalMoveError")
    except chess.IllegalMoveError:
        pass
