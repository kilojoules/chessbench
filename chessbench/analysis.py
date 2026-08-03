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
                                "start_fen": game["start_fen"],
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


def games_raster(df, out_path: Path) -> None:
    """Every game as a row: exposure bar to first event (dot) or censoring
    (open arrow), grouped by cell, sorted by survival. Honest about the
    tiny discrete time scale in a way step curves are not."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted(df["cell"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 9.5), dpi=200)
    y = 0
    yticks, ylabels = [], []
    for cell in cells:
        cell_df = df[df["cell"] == cell]
        variant, visibility = cell_df.iloc[0]["variant"], cell_df.iloc[0]["visibility"]
        color = VARIANT_COLOR.get(variant, INK_2)
        y_start = y
        # Sub-block per playing color: the color control, visible — each
        # cell shows its as-White and as-Black silhouettes side by side.
        for black in (0, 1):
            sub = cell_df[cell_df["black"] == black].sort_values(
                ["T", "E"], ascending=[False, True])
            if sub.empty:
                continue
            block_start = y
            for _, r in sub.iterrows():
                ax.hlines(y, 0, r["T"], color=color, linewidth=2,
                          alpha=0.45 if visibility == "history-only" else 0.85)
                if r["E"]:
                    ax.plot(r["T"], y, "o", color=color, markersize=3.5)
                else:
                    ax.plot(r["T"], y, ">", color=color, markersize=4,
                            markerfacecolor="none")
                y += 1
            ax.text(-0.25, (block_start + y - 1) / 2, "B" if black else "W",
                    ha="right", va="center", fontsize=7, color=INK_2)
            y += 1  # thin gap between color blocks
        yticks.append((y_start + y - 2) / 2)
        ylabels.append(cell.replace(" × history+board", " · board")
                       .replace(" × history-only", " · blind"))
        y += 2  # gap between cells
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8, color=INK)
    ax.invert_yaxis()
    _style_ax(ax, "Every game: exposure until first illegal/ambiguous attempt",
              "LLM move index (dot = event, open arrow = censored)", "")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def means_plot(df, out_path: Path) -> None:
    """The simple headline figure: mean LLM moves until the first illegal
    attempt, per cell (restricted mean survival time, which equals the plain
    mean when nothing is censored), with bootstrap 90% CIs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(0)
    tau = float(df["T"].max())

    def rmst(t, e):
        # mean of min(T, tau); with full events this is the plain mean
        return float(np.minimum(t, tau).mean())

    rows = []
    order = [("standard", "history+board"), ("standard", "history-only"),
             ("standard-offbook", "history+board"), ("standard-offbook", "history-only"),
             ("chess960", "history+board"), ("chess960", "history-only")]
    for variant, vis in order:
        sub = df[(df["variant"] == variant) & (df["visibility"] == vis)]
        if sub.empty:
            continue
        t = sub["T"].to_numpy()
        m = rmst(t, sub["E"].to_numpy())
        boots = [rmst(t[rng.integers(0, len(t), len(t))], None) for _ in range(500)]
        lo, hi = np.percentile(boots, [5, 95])
        rows.append((variant, vis, m, lo, hi, len(sub)))

    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=200)
    ys = np.arange(len(rows))[::-1]
    for yi, (variant, vis, m, lo, hi, n) in zip(ys, rows):
        color = VARIANT_COLOR.get(variant, INK_2)
        blind = vis == "history-only"
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=1.6)
        ax.plot(m, yi, "o", color=color, markersize=8,
                markerfacecolor=SURFACE if blind else color)
        ax.text(hi + 0.08, yi, f"{m:.1f}", va="center", fontsize=9, color=INK)
    labels = [f"{v.replace('standard-offbook', 'offbook')} · {'blind' if vis == 'history-only' else 'board'}"
              for v, vis, *_ in rows]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(0, None)
    _style_ax(ax, "Mean LLM moves until the first illegal attempt (90% CI)",
              "moves (higher = survives longer; filled = board shown, open = blindfold)", "")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def effects_figure(summary, illegals: list[dict], out_path: Path) -> None:
    """One figure, two panels, one visual grammar. Panel A: WHEN it fails
    (Cox hazard ratios). Panel B: HOW it fails (mechanism rates). Encodings
    mean exactly one thing everywhere: fill = board visibility (filled =
    board shown, open = blindfold), hue = variant identity. Panel A rows are
    model coefficients, not cells, so they use a different mark (small solid
    squares); significance is carried by position vs HR=1, with clearing
    intervals inked darker than straddling ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    fig, (axf, axm) = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=200,
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---------------- Panel A: when it fails ----------------
    ROWS = [
        ("v960", "chess960 (H3)", VARIANT_COLOR["chess960"]),
        ("voffbook", "offbook", VARIANT_COLOR["standard-offbook"]),
        ("blind", "blindfold (H2)", INK),
        ("blind_x_960", "blindfold \u00d7 960 (H4)", VARIANT_COLOR["chess960"]),
        ("blind_x_offbook", "blindfold \u00d7 offbook", VARIANT_COLOR["standard-offbook"]),
        ("black", "plays Black (H1)", INK),
    ]
    if summary is not None:
        rows = [(n, lab, c) for n, lab, c in ROWS if n in summary.index]
        ys = np.arange(len(rows))[::-1]
        for yi, (name, lab, _color) in zip(ys, rows):
            r = summary.loc[name]
            lo, hi = r.get("coef lower 90%"), r.get("coef upper 90%")
            # Monochrome by design: significance in a forest plot is
            # geometric (the interval crosses 1 or it does not) — no ink
            # rule needed, and hue stays reserved for variant identity in
            # the right panel (row labels carry the tie by name).
            axf.plot([lo, hi], [yi, yi], color=INK, solid_capstyle="butt", linewidth=2)
            axf.plot(r["coef"], yi, "s", color=INK, markersize=5, zorder=3)
            if name == "black":
                # The equivalence band belongs to this test alone: a small
                # labeled strip behind the plays-Black row.
                axf.add_patch(Rectangle((np.log(EQUIV_MARGIN[0]), yi - 0.38),
                                        np.log(EQUIV_MARGIN[1]) - np.log(EQUIV_MARGIN[0]),
                                        0.76, facecolor=GRID, edgecolor="none", zorder=0))
                axf.text(np.log(EQUIV_MARGIN[1]) + 0.06, yi, "H1 margin",
                         fontsize=6.8, color=INK_2, va="center")
        axf.axvline(0, color=INK_2, linewidth=0.8)
        axf.set_yticks(ys)
        axf.set_yticklabels([lab for _, lab, _ in rows], fontsize=9, color=INK)
        ticks = [0.25, 0.5, 1, 2, 4, 8]
        axf.set_xticks([np.log(t) for t in ticks])
        axf.set_xticklabels([str(t) for t in ticks])
        _style_ax(axf, "When it fails \u2014 pre-registered hazard ratios (90% CI)",
                  "hazard ratio (log scale) \u2014 right of 1 = fails sooner", "")
    else:
        axf.text(0.5, 0.5, "no Cox fit", ha="center", color=INK_2)

    # ---------------- Panel B: how it fails ----------------
    ORDER = ["standard", "standard-offbook", "chess960"]
    SHORT = {"standard": "standard", "standard-offbook": "offbook", "chess960": "chess960"}

    def cell_stats(metric, variant, vis):
        pool = [a for a in illegals if a["variant"] == variant and a["visibility"] == vis]
        if not pool:
            return None, 0, 0
        if metric == "stale-state":
            hits = sum(stale_state(a) for a in pool)
        else:
            hits = sum(phantom_standard(a["history"], a["candidate"]) for a in pool)
        return hits / len(pool), hits, len(pool)

    rows_b = [("stale-state", v) for v in ORDER] + [("phantom-standard", v) for v in ORDER]
    ys = np.arange(len(rows_b))[::-1]
    labels_b = []
    for yi, (metric, variant) in zip(ys, rows_b):
        color = VARIANT_COLOR[variant]
        rb, hb, nb = cell_stats(metric, variant, "history+board")
        rl, hl_, nl = cell_stats(metric, variant, "history-only")
        labels_b.append(f"{metric}\n{SHORT[variant]}")
        if rb is None or rl is None:
            continue
        axm.plot([rb, rl], [yi, yi], color=color, linewidth=1.6, zorder=1)
        axm.plot(rb, yi, "o", color=color, markersize=7, zorder=2)
        axm.plot(rl, yi, "o", color=color, markersize=7, markerfacecolor=SURFACE, zorder=2)
        if metric == "phantom-standard":
            # Counts on the dots that matter \u2014 including the structural zeros,
            # so "phantoms exist only in chess960" is drawn, not implied.
            if hb == 0 and hl_ == 0:
                # Both conditions sit at zero: draw both marks (open ring
                # over the filled dot) and say so.
                axm.plot(0, yi, "o", color=color, markersize=7, zorder=2)
                axm.plot(0, yi, "o", color=color, markersize=7,
                         markerfacecolor=SURFACE, zorder=3, fillstyle="left")
                axm.text(0.035, yi, "0 (both conditions)", fontsize=7.5,
                         color=INK_2, va="center")
            else:
                axm.text(rb, yi + 0.32, str(hb), fontsize=7.5, color=INK, ha="center")
                axm.text(rl, yi + 0.32, str(hl_), fontsize=7.5, color=INK, ha="center")
    axm.set_yticks(ys)
    axm.set_yticklabels(labels_b, fontsize=8.5, color=INK)
    axm.set_xlim(-0.03, None)
    axm.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    _style_ax(axm, "How it fails \u2014 mechanism rates",
              "share of that cell's illegal attempts", "")
    fig.text(0.005, 0.005,
             "filled = board shown, open = blindfold \u00b7 stale-state = the attempt was legal 1\u20136 "
             "plies earlier \u00b7 phantom-standard = legal replaying the same moves from the standard "
             "start (structurally 0 outside chess960).",
             fontsize=6.8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
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
    the STANDARD start? A True is the signature of pattern-matching on
    standard-chess geometry: the model played the board that isn't there.

    Precise semantics: reconstruct the counterfactual position by replaying
    the game's full SAN history from the standard starting array; the
    attempt is phantom iff the replay succeeds (every played move is also
    legal from the standard start) AND the candidate parses as fully legal
    and unambiguous there. Any failure anywhere returns False.

    Interpretation notes (see README "The phantom-standard detector,
    precisely"):
    - CONSERVATIVE lower bound: once one real 960 move is impossible from
      the standard array, the replay breaks and no later attempt in that
      game can be flagged — detection is biased toward the opening.
    - STRUCTURAL ZERO outside chess960: for standard/offbook games the
      reconstruction equals reality, so an illegal move stays illegal and
      the detector cannot false-positive; the zero rows are the control.
    - OVERLAY: a phantom attempt also carries its base mechanism class, so
      taxonomy columns do not sum.
    - Tests exactly ONE counterfactual (the standard start); other
      hallucinated boards land in stale-state or piece-cannot-reach."""
    board = chess.Board()
    try:
        for san in history:
            board.push_san(san)
        board.parse_san(candidate or "")
        return True
    except ValueError:
        return False


def stale_state(a: dict, max_back: int = 6) -> bool:
    """Was the illegal candidate LEGAL at any of the previous `max_back`
    positions of this game? A stale-state error means the model played
    against an outdated board — the direct signature of state-tracking lag
    (particularly damning in board-shown cells, where the current position
    was displayed in the very prompt)."""
    board = chess.Board(a["start_fen"], chess960=a["variant"] == "chess960")
    boards = [board.copy()]
    try:
        for san in a["history"]:
            board.push_san(san)
            boards.append(board.copy())
    except ValueError:
        return False
    for back in range(2, min(max_back + 1, len(boards)) + 1):
        try:
            boards[-back].parse_san(a["candidate"] or "")
            return True
        except ValueError:
            continue
    return False


def taxonomy_table(illegals: list[dict]) -> dict:
    out: dict = {}
    for a in illegals:
        cell = f"{a['variant']} × {a['visibility']}"
        c = out.setdefault(cell, Counter())
        c[classify_illegal(a["fen_before"], a["candidate"], a["variant"] == "chess960")] += 1
        if a["variant"] == "chess960" and phantom_standard(a["history"], a["candidate"]):
            c["(phantom-standard)"] += 1
        if stale_state(a):
            c["(stale-state)"] += 1
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
    games_raster(df, out_dir / "games.png")
    means_plot(df, out_dir / "means.png")

    summary, cox_note = cox_fit(df)
    effects_figure(summary, illegals, out_dir / "effects.png")
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
    parts.append("\n![Every game](games.png)\n\n![Mean moves to first illegal attempt](means.png)\n\n"
                 "![Effects and mechanisms](effects.png)\n\n"
                 "![Kaplan-Meier](km.png)\n\n![Discrete hazard](hazard.png)\n")

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
        parts.append("\n_CIs: Wald intervals on log-HR with cluster-robust (sandwich) SEs "
                     "clustered by start position/prefix, exp-transformed; 90% level chosen "
                     "so the H1 equivalence test follows the TOST convention (90% CI inside "
                     "the margin = alpha 0.05 equivalence)._\n")
    else:
        parts.append(f"_{cox_note}_\n")
    parts.append(f"\n**H1 color equivalence:** {color_equivalence(summary)}\n")

    # Color control
    parts.append("## Color control (White vs Black, per cell)\n")
    rows = []
    for cell, sub in sorted(df.groupby("cell")):
        row = [cell]
        for black in (0, 1):
            s = sub[sub["black"] == black]
            ev = s.loc[s["E"] == 1, "T"]
            med = f"{ev.median():.0f}" if len(ev) else "—"
            row.append(f"{len(s)} ({int(s['E'].sum())} ev, med {med})")
        rows.append(row)
    parts.append(_md_table(["cell", "as White: n (events, median move)",
                            "as Black: n (events, median move)"], rows))
    parts.append("\n_Color is balanced within every cell and enters the Cox model as "
                 "the `black` covariate; H1 (equivalence) above is the formal test._\n")

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
                     "the direct signature of standard-geometry pattern matching. It is a "
                     "conservative lower bound (the full history must replay from the standard "
                     "array, so detection is biased toward the opening) and a structural zero "
                     "outside chess960 (the reconstruction equals reality there), which makes "
                     "the non-960 rows a built-in control. Both parenthesized classes are "
                     "overlays on the base classes, so columns do not sum. "
                     "`(stale-state)` counts attempts legal at a position 1–6 plies earlier — "
                     "state-tracking lag; in board-shown cells these contradict the very board "
                     "displayed in the prompt._\n")
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
