import argparse
from collections import defaultdict

from chessbench.positions import draw_chess960_positions
from chessbench.run import build_specs


def _args():
    return argparse.Namespace(max_plies=200, max_attempts=3, temperature=0.0)


def test_chess960_positions_get_both_colors():
    positions = draw_chess960_positions(10, 2026)
    specs = build_specs("m", ["chess960"], ["history-only"], 30, positions, _args())
    colors_by_pos = defaultdict(set)
    for s in specs:
        colors_by_pos[s.sp_id].add(s.llm_color)
    assert all(colors == {"white", "black"} for colors in colors_by_pos.values())


def test_cell_color_balance():
    positions = draw_chess960_positions(10, 2026)
    for variant in ("standard", "chess960"):
        specs = build_specs("m", [variant], ["history-only"], 30, positions, _args())
        whites = sum(1 for s in specs if s.llm_color == "white")
        assert whites == 15


def test_game_ids_unique():
    positions = draw_chess960_positions(10, 2026)
    specs = build_specs(
        "anthropic/claude-sonnet-5", ["standard", "chess960"],
        ["history-only", "history+board"], 30, positions, _args(),
    )
    ids = [s.game_id for s in specs]
    assert len(ids) == len(set(ids)) == 120
