"""Parallel-runner integration test (needs stockfish; quick node caps)."""

import json
import shutil

import pytest

from chessbench.run import main

pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish not installed"
)


def test_parallel_run_completes_all_games(tmp_path):
    rc = main([
        "--model", "fake:random",
        "--variant", "both",
        "--games-per-cell", "1",
        "--parallel", "2",
        "--nodes", "300",
        "--eval-depth", "0",
        "--max-plies", "8",
        "--out", str(tmp_path),
    ])
    assert rc == 0
    jsonls = list(tmp_path.glob("*.jsonl"))
    assert len(jsonls) == 4
    for f in jsonls:
        recs = [json.loads(line) for line in f.open()]
        assert recs[-1]["type"] == "game"
        assert f.with_suffix(".pgn").exists()
    # resume: everything skips
    rc = main([
        "--model", "fake:random", "--variant", "both", "--games-per-cell", "1",
        "--parallel", "2", "--nodes", "300", "--eval-depth", "0",
        "--max-plies", "8", "--out", str(tmp_path),
    ])
    assert rc == 0
