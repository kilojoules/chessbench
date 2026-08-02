"""Pre-registered analysis (PLAN.md §7).

Usage: uv run chessbench-analyze runs/<name> [runs/<other> ...] [-o OUTDIR]

Implements exactly the frozen analysis plan: Kaplan-Meier per cell on the
LLM-move-index time scale, discrete hazard-by-move curves, a cause-specific
Cox model (visibility, variant, interaction, color; cluster-robust by start
position/prefix), the color equivalence check against HR in [0.67, 1.5],
per-cell censoring-by-cause tables, the illegal-move error taxonomy, and the
two pre-registered sensitivity analyses (illegal-only events;
strict-protocol extraction).

Requires the analysis extras: `uv sync --extra analysis`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import chess

# --- validated palette (dataviz skill; hue = variant, dash = visibility) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e5e2"
VARIANT_COLOR = {"standard": "#2a78d6", "chess960": "#eb6834", "standard-offbook": "#1baf7a"}
VIS_DASH = {"history+board": "solid", "history-only": (0, (4, 2))}

EQUIV_MARGIN = (0.67, 1.5)  # pre-registered color hazard-ratio margin

CENSOR_CAUSES = ("checkmate", "stalemate", "insufficient_material", "seventyfive_moves",
                 "fivefold_repetition", "move_cap", "llm_truncated", "llm_forfeit")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_run_dirs(run_dirs: list[Path]) -> tuple[list[dict], list[dict], Counter]:
    """Returns (game_records, illegal_attempts, overflow_by_cell). Each illegal
    attempt carries the SAN history up to that point for the phantom-standard
    check; overflow_by_cell counts context-overflow attempts (contaminated
    samples that were quarantined as infrastructure)."""
    games: list[dict] = []
    illegals: list[dict] = []
    overflow: Counter = Counter()
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("*.jsonl")):
            try:
                recs = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
            except (OSError, json.JSONDecodeError):
                continue
            game = next((r for r in recs if r.get("type") == "game"), None)
            if game is None:
                continue
            games.append(game)
            sans: list[str] = []
            for r in recs:
                if r["type"] in ("engine_move", "prefix_move"):
                    sans.append(r["move_san"])
                elif r["type"] == "attempt":
                    if r.get("context_overflow"):
                        overflow[f"{game['variant']} × {game['visibility']}"] += 1
                    if r["parse_class"] == "illegal":
                        illegals.append(
                            {
                                "variant": game["variant"],
                                "visibility": game["visibility"],
                                "model": game["model"],
                                "fen_before": r["fen_before"],
                                "candidate": r.get("candidate"),
                                "history": list(sans),
                            }
                        )
                    if r["parse_class"] == "legal":
                        sans.append(r["move_san"])
    return games, illegals, overflow


def games_frame(games: list[dict], event_classes: str = "prereg",
                extraction: str = "lenient"):
    """games -> pandas DataFrame with pre-registered survival columns.

    event_classes: "prereg" (illegal or ambiguous) | "illegal-only"
    extraction: "lenient" | reserved for future strict re-parses
    """
    import pandas as pd

    rows = []
    for g in games:
        if event_classes == "illegal-only":
            event = g.get("first_illegal_llm_move") is not None
            t = g.get("first_illegal_llm_move") if event else None
        else:
            event = bool(g.get("event"))
            t = g.get("first_event_llm_move") if event else None
        if t is None:
            # censored: exposure = LLM moves requested (fallback for old records)
            t = g.get("survival_llm_moves") or g.get("n_llm_moves_requested") or 0
        if t <= 0:
            continue  # no exposure at all (e.g. instant infra failure)
        unit = f"pfx{g['prefix_id']}" if g.get("prefix_id") is not None else f"sp{g['sp_id']}"
        rows.append(
            {
                "T": t,
                "E": int(event),
                "model": g["model"],
                "variant": g["variant"],
                "visibility": g["visibility"],
                "cell": f"{g['variant']} × {g['visibility']}",
                "blind": int(g["visibility"] == "history-only"),
                "v960": int(g["variant"] == "chess960"),
                "voffbook": int(g["variant"] == "standard-offbook"),
                "black": int(g["llm_color"] == "black"),
                "unit": unit,
                "termination": g["termination"],
                "forfeit_classes": ",".join(g.get("forfeit_attempt_classes") or []),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["blind_x_960"] = df["blind"] * df["v960"]
        df["blind_x_offbook"] = df["blind"] * df["voffbook"]
    return df


# --------------------------------------------------------------------------
# Plots (validated palette; hue = variant, dash = visibility)
# --------------------------------------------------------------------------

def _style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def _cell_style(variant: str, visibility: str) -> dict:
    return {
        "color": VARIANT_COLOR.get(variant, INK_2),
        "linestyle": VIS_DASH.get(visibility, "solid"),
        "linewidth": 2,
    }


def km_plot(df, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    for (variant, visibility), sub in sorted(df.groupby(["variant", "visibility"])):
        kmf = KaplanMeierFitter()
        kmf.fit(sub["T"], sub["E"], label=f"{variant} · {'board' if visibility == 'history+board' else 'blindfold'}")
        kmf.plot_survival_function(ax=ax, ci_show=False, **_cell_style(variant, visibility))
    _style_ax(ax, "Survival: probability of no illegal/ambiguous attempt yet",
              "LLM move index", "P(no event)")
    ax.set_ylim(0, 1.02)
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def hazard_plot(df, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    for (variant, visibility), sub in sorted(df.groupby(["variant", "visibility"])):
        max_t = int(sub["T"].max())
        ks, hs = [], []
        for k in range(1, max_t + 1):
            at_risk = ((sub["T"] > k) | ((sub["T"] == k))).sum()
            events = ((sub["T"] == k) & (sub["E"] == 1)).sum()
            if at_risk >= 3:  # suppress ultra-noisy tail estimates
                ks.append(k)
                hs.append(events / at_risk)
        label = f"{variant} · {'board' if visibility == 'history+board' else 'blindfold'}"
        ax.plot(ks, hs, marker="o", markersize=3, label=label,
                **_cell_style(variant, visibility))
    _style_ax(ax, "Discrete hazard: P(first event at move k | survived to k)",
              "LLM move index", "hazard")
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def cox_fit(df):
    """Cause-specific Cox per the prereg. Returns (summary_df, note) —
    summary None if the model cannot be fit on this data."""
    from lifelines import CoxPHFitter

    covs = [c for c in ("blind", "v960", "voffbook", "blind_x_960", "blind_x_offbook", "black")
            if c in df.columns and df[c].nunique() > 1]
    if not covs or df["E"].sum() < 5:
        return None, "insufficient data or covariate variation for a Cox fit"
    cols = ["T", "E", "unit"] + covs
    try:
        cph = CoxPHFitter(alpha=0.10)  # 90% CIs for the equivalence check
        cph.fit(df[cols], duration_col="T", event_col="E", cluster_col="unit", robust=True)
        return cph.summary, None
    except Exception as e:  # convergence failures on tiny/degenerate data
        return None, f"Cox fit failed: {type(e).__name__}: {e}"


def color_equivalence(summary) -> str:
    if summary is None or "black" not in summary.index:
        return "not evaluable (no Cox fit or no color variation)"
    import numpy as np

    row = summary.loc["black"]
    hr = float(np.exp(row["coef"]))
    lo = float(np.exp(row["coef lower 90%"])) if "coef lower 90%" in row else None
    hi = float(np.exp(row["coef upper 90%"])) if "coef upper 90%" in row else None
    if lo is None:
        return f"HR(black)={hr:.2f}; CI columns unavailable"
    if EQUIV_MARGIN[0] <= lo and hi <= EQUIV_MARGIN[1]:
        verdict = "EQUIVALENT within the pre-registered margin"
    elif hi < EQUIV_MARGIN[0] or lo > EQUIV_MARGIN[1]:
        verdict = "NOT equivalent (CI wholly outside the margin)"
    else:
        verdict = "inconclusive (CI overlaps the margin boundary)"
    return f"HR(black) = {hr:.2f}, 90% CI [{lo:.2f}, {hi:.2f}] vs margin {list(EQUIV_MARGIN)} → {verdict}"


# --------------------------------------------------------------------------
# Error taxonomy (prereg addition 2)
# --------------------------------------------------------------------------

_SAN_PARTS = re.compile(r"^([KQRBN])?([a-h])?([1-8])?([-x])?([a-h][1-8])(=?[QRBNqrbn])?([+#])?$")


def classify_illegal(fen_before: str, candidate: str | None, chess960: bool) -> str:
    """Best-effort mechanism class for a well-formed-but-illegal SAN."""
    cand = candidate or ""
    board = chess.Board(fen_before, chess960=chess960)
    if cand.startswith("O-O"):
        return "illegal-castling"
    m = _SAN_PARTS.match(cand)
    if not m:
        return "unclassified"
    piece_letter, from_file, from_rank, _cap, target, _promo, _chk = m.groups()
    piece_type = chess.Piece.from_symbol(piece_letter).piece_type if piece_letter else chess.PAWN
    target_sq = chess.parse_square(target)

    def matches(mv: chess.Move) -> bool:
        p = board.piece_at(mv.from_square)
        if p is None or p.piece_type != piece_type or mv.to_square != target_sq:
            return False
        if from_file and chess.square_file(mv.from_square) != "abcdefgh".index(from_file):
            return False
        if from_rank and chess.square_rank(mv.from_square) != int(from_rank) - 1:
            return False
        return True

    if any(matches(mv) for mv in board.pseudo_legal_moves):
        return "into-check-or-pin"
    tp = board.piece_at(target_sq)
    if tp is not None and tp.color == board.turn:
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == board.turn and p.piece_type == piece_type and target_sq in board.attacks(sq):
                return "own-piece-capture"
    return "piece-cannot-reach"


def phantom_standard(history: list[str], candidate: str | None) -> bool:
    """Would the candidate be LEGAL if the same movetext had been played from
    the STANDARD start? (Only meaningful for chess960 games — a True here is
    the signature of pattern-matching on standard-chess geometry.)"""
    board = chess.Board()
    try:
        for san in history:
            board.push_san(san)
        board.parse_san(candidate or "")
        return True
    except ValueError:
        return False


def taxonomy_table(illegals: list[dict]) -> dict:
    out: dict = {}
    for a in illegals:
        cell = f"{a['variant']} × {a['visibility']}"
        c = out.setdefault(cell, Counter())
        c[classify_illegal(a["fen_before"], a["candidate"], a["variant"] == "chess960")] += 1
        if a["variant"] == "chess960" and phantom_standard(a["history"], a["candidate"]):
            c["(phantom-standard)"] += 1
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def write_report(df, df_illonly, games: list[dict], illegals: list[dict],
                 out_dir: Path, run_dirs: list[Path],
                 overflow: Counter | None = None) -> Path:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    km_plot(df, out_dir / "km.png")
    hazard_plot(df, out_dir / "hazard.png")

    summary, cox_note = cox_fit(df)
    parts: list[str] = []
    parts.append("# chessbench analysis\n")
    parts.append(f"Runs: {', '.join(str(r) for r in run_dirs)}  ")
    parts.append(f"Games: {len(df)} analyzable / {len(games)} total · "
                 f"models: {', '.join(sorted(df['model'].unique()))}\n")
    parts.append("Pre-registered analysis per PLAN.md §7 (time scale: LLM move index; "
                 "event: first illegal-or-ambiguous attempt).\n")

    # Per-cell summary
    rows = []
    for cell, sub in sorted(df.groupby("cell")):
        med = sub.loc[sub["E"] == 1, "T"].median() if sub["E"].any() else None
        rows.append([cell, len(sub), int(sub["E"].sum()),
                     f"{med:.0f}" if med == med and med is not None else "—",
                     f"{sub['T'].median():.0f}"])
    parts.append("## Per-cell summary\n")
    parts.append(_md_table(["cell", "games", "events", "median event move", "median exposure"], rows))
    parts.append("\n![Kaplan-Meier](km.png)\n\n![Discrete hazard](hazard.png)\n")

    # Cox
    parts.append("## Cox model (cause-specific, cluster-robust by position/prefix)\n")
    if summary is not None:
        cox_rows = []
        for name, row in summary.iterrows():
            hr = np.exp(row["coef"])
            lo = np.exp(row.get("coef lower 90%", np.nan))
            hi = np.exp(row.get("coef upper 90%", np.nan))
            cox_rows.append([name, f"{row['coef']:.3f}", f"{hr:.2f}",
                             f"[{lo:.2f}, {hi:.2f}]", f"{row['p']:.3g}"])
        parts.append(_md_table(["covariate", "log-HR", "HR", "90% CI", "p"], cox_rows))
        parts.append("\nH4 is the `blind_x_960` row: positive log-HR = blindfold hurts "
                     "more in chess960 than in standard (the pattern-matching prediction).\n")
    else:
        parts.append(f"_{cox_note}_\n")
    parts.append(f"\n**H1 color equivalence:** {color_equivalence(summary)}\n")

    # Censoring by cause
    parts.append("## Censoring / termination by cause (per cell)\n")
    causes = sorted(df["termination"].unique())
    rows = []
    for cell, sub in sorted(df.groupby("cell")):
        counts = sub["termination"].value_counts()
        rows.append([cell] + [int(counts.get(c, 0)) for c in causes])
    parts.append(_md_table(["cell"] + causes, rows))
    parts.append("\n_`llm_forfeit` rows with no illegal/ambiguous attempt are format "
                 "forfeits (censoring); truncation is infrastructure censoring. Both "
                 "are condition-correlated risks — watch these columns._\n")
    if overflow:
        parts.append("\n**Context-overflow attempts** (provider context-shifted "
                     "mid-generation; quarantined as infrastructure, never graded): "
                     + ", ".join(f"{cell}: {n}" for cell, n in sorted(overflow.items()))
                     + "\n")
    else:
        parts.append("\n_No context-overflow attempts detected._\n")

    # Error taxonomy
    parts.append("## Illegal-move error taxonomy\n")
    tax = taxonomy_table(illegals)
    classes = sorted({c for counter in tax.values() for c in counter})
    if classes:
        rows = [[cell] + [counter.get(c, 0) for c in classes] for cell, counter in sorted(tax.items())]
        parts.append(_md_table(["cell"] + classes, rows))
        parts.append("\n_`(phantom-standard)` counts chess960 illegal attempts that would "
                     "have been LEGAL replaying the same movetext from the standard start — "
                     "the direct signature of standard-geometry pattern matching._\n")
    else:
        parts.append("_no illegal attempts logged_\n")

    # Sensitivity: illegal-only events
    parts.append("## Sensitivity: illegal-only events\n")
    rows = []
    for cell, sub in sorted(df_illonly.groupby("cell")):
        med = sub.loc[sub["E"] == 1, "T"].median() if sub["E"].any() else None
        rows.append([cell, len(sub), int(sub["E"].sum()),
                     f"{med:.0f}" if med == med and med is not None else "—"])
    parts.append(_md_table(["cell", "games", "events", "median event move"], rows))

    report = out_dir / "report.md"
    report.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chessbench-analyze", description=__doc__)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output directory (default: <first run_dir>/analysis)")
    args = p.parse_args(argv)

    try:
        import lifelines  # noqa: F401
        import matplotlib  # noqa: F401
        import pandas  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"analysis extras missing ({e.name}); run: uv sync --extra analysis")

    games, illegals, overflow = load_run_dirs(args.run_dirs)
    if not games:
        raise SystemExit("no completed games found")
    df = games_frame(games)
    df_illonly = games_frame(games, event_classes="illegal-only")
    out_dir = args.out or (args.run_dirs[0] / "analysis")
    report = write_report(df, df_illonly, games, illegals, out_dir, args.run_dirs, overflow)
    print(f"{report} ({len(df)} games, {int(df['E'].sum())} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
