from __future__ import annotations

import chess
import chess.engine

from .config import EngineConfig


class Engine:
    """Stockfish wrapper. python-chess manages UCI_Chess960 automatically
    based on the board it is handed — never set it via configure()."""

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.engine = chess.engine.SimpleEngine.popen_uci(cfg.path)
        self.engine.configure({"Skill Level": cfg.skill_level})
        self.limit = chess.engine.Limit(nodes=cfg.nodes)
        self.name = self.engine.id.get("name", "unknown")
        # Evals run on a SEPARATE full-strength process: analysing on the
        # playing engine would warm its hash/history tables (and force
        # ucinewgame churn), so the --eval-depth instrumentation flag would
        # change the opponent's actual moves.
        self.eval_engine = None
        if cfg.eval_depth > 0:
            self.eval_engine = chess.engine.SimpleEngine.popen_uci(cfg.path)

    def play(self, board: chess.Board, game: object = None) -> chess.Move:
        # `game` keys ucinewgame so state doesn't bleed across games.
        result = self.engine.play(board, self.limit, game=game)
        assert result.move is not None
        return result.move

    def eval_cp_white(self, board: chess.Board) -> int | None:
        """White-POV eval in centipawns (mate mapped to ±100000), from a
        full-strength engine. For plots only; rigorous CPL is computed
        offline from the PGNs (PLAN.md D7)."""
        if self.eval_engine is None:
            return None
        info = self.eval_engine.analyse(board, chess.engine.Limit(depth=self.cfg.eval_depth))
        score = info["score"].white()
        return score.score(mate_score=100_000)

    def close(self) -> None:
        self.engine.quit()
        if self.eval_engine is not None:
            self.eval_engine.quit()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
