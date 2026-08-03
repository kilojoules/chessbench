from chessbench.config import GameSpec
from chessbench.game import play_game
from chessbench.llm import FakeLLM
from chessbench.viz import build_html, load_run


class StubEngine:
    name = "stub-engine"

    def play(self, board, game=None):
        return next(iter(board.legal_moves))

    def eval_cp_white(self, board):
        return None


def _spec(gid, **kw):
    defaults = dict(
        game_id=gid, model="fake:first", variant="chess960", sp_id=105,
        llm_color="white", game_index=0, max_plies=10, max_attempts=3,
        visibility="history-only",
    )
    defaults.update(kw)
    return GameSpec(**defaults)


def test_viewer_roundtrip(tmp_path):
    play_game(_spec("v1"), FakeLLM(policy="random", seed=5, illegal_at={2}), StubEngine(), tmp_path)
    play_game(_spec("v2", variant="standard", sp_id=518), FakeLLM(policy="first"), StubEngine(), tmp_path)
    games = load_run(tmp_path)
    assert len(games) == 2
    g1 = next(g for g in games if g["id"] == "v1")
    # The game halts at the illegal attempt (llm move 2, game ply 3): the
    # attempt lands in final_fails and only two plies were played.
    assert len(g1["plies"]) == 2
    assert g1["final_fails"] and g1["final_fails"][0]["class"] == "illegal"
    assert all("fen" in p for p in g1["plies"])
    html = build_html(games, "test run")
    assert "v1" in html and "v2" in html
    assert "</script>" in html  # template intact after JSON splice


def test_incomplete_game_skipped(tmp_path):
    (tmp_path / "broken.jsonl").write_text('{"type": "attempt"}\n')
    assert load_run(tmp_path) == []
