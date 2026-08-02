import argparse
from collections import defaultdict

import chess

from chessbench.positions import draw_chess960_positions, draw_offbook_prefixes
from chessbench.run import build_specs


def _args():
    return argparse.Namespace(max_plies=200, max_attempts=3, temperature=0.0, no_think=False)


POSITIONS = draw_chess960_positions(10, 2026)
PREFIXES = draw_offbook_prefixes(10, 6, 2027)


def test_chess960_positions_get_both_colors():
    specs = build_specs("m", ["chess960"], ["history-only"], 30, POSITIONS, PREFIXES, _args())
    colors_by_pos = defaultdict(set)
    for s in specs:
        colors_by_pos[s.sp_id].add(s.llm_color)
    assert all(colors == {"white", "black"} for colors in colors_by_pos.values())


def test_cell_color_balance():
    for variant in ("standard", "chess960", "standard-offbook"):
        specs = build_specs("m", [variant], ["history-only"], 30, POSITIONS, PREFIXES, _args())
        whites = sum(1 for s in specs if s.llm_color == "white")
        assert whites == 15


def test_game_ids_unique():
    specs = build_specs(
        "anthropic/claude-sonnet-5", ["standard", "chess960", "standard-offbook"],
        ["history-only", "history+board"], 30, POSITIONS, PREFIXES, _args(),
    )
    ids = [s.game_id for s in specs]
    assert len(ids) == len(set(ids)) == 180


def test_offbook_prefixes_deterministic_and_legal():
    assert PREFIXES == draw_offbook_prefixes(10, 6, 2027)
    assert len(PREFIXES) == 10
    for sans in PREFIXES:
        assert len(sans) == 6
        board = chess.Board()
        for san in sans:
            board.push_san(san)  # raises if illegal
        assert not board.is_game_over()


def test_offbook_specs_carry_prefix():
    specs = build_specs("m", ["standard-offbook"], ["history-only"], 10, POSITIONS, PREFIXES, _args())
    for s in specs:
        assert len(s.opening_prefix) == 6
        assert s.prefix_id is not None
        assert f"pfx{s.prefix_id:02d}" in s.game_id
    assert len({s.prefix_id for s in specs}) == 10


def test_standard_specs_have_no_prefix():
    specs = build_specs("m", ["standard", "chess960"], ["history-only"], 4, POSITIONS, PREFIXES, _args())
    assert all(s.opening_prefix == () and s.prefix_id is None for s in specs)


def test_no_think_wires_native_parameter():
    from chessbench.run import make_llm
    from chessbench.config import GameSpec
    spec = GameSpec(game_id="t", model="ollama_chat/x", variant="standard",
                    visibility="history-only", sp_id=518, llm_color="white", game_index=0)
    args = argparse.Namespace(max_tokens=1024, llm_timeout=60, llm_seed=None,
                              num_ctx=8192, no_think=True)
    llm = make_llm(spec, args)
    assert llm.think is False
    args.no_think = False
    assert make_llm(spec, args).think is None


def test_num_ctx_guard_rejects_unsafe_window():
    import pytest
    from chessbench.run import main
    with pytest.raises(SystemExit):
        main(["--model", "ollama_chat/x", "--num-ctx", "4096",
              "--max-tokens", "12288", "--list"])


def test_num_ctx_guard_autodefaults_for_ollama(capsys):
    from chessbench.run import main
    rc = main(["--model", "ollama_chat/x", "--max-tokens", "8192", "--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "defaulting to 12288" in out  # max-tokens + 4096


def test_num_ctx_guard_silent_for_api_models(capsys):
    from chessbench.run import main
    rc = main(["--model", "anthropic/claude-sonnet-5", "--list"])
    assert rc == 0
    assert "[guard]" not in capsys.readouterr().out
