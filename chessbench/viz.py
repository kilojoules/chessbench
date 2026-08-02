"""Generate a self-contained HTML viewer animating the games of a run.

Usage: uv run chessbench-viz runs/<name> [-o viewer.html]

Reads every completed game JSONL in the run directory and embeds a replayable
animation: board playback, per-ply annotations of failed attempts (what the
model tried, and why it was rejected), and game metadata. No external assets —
one file, openable anywhere.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import chess


def load_game(path: Path) -> dict | None:
    recs = []
    try:
        with path.open(encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    game = next((r for r in recs if r.get("type") == "game"), None)
    if game is None:
        return None

    board = chess.Board(game["start_fen"], chess960=game["variant"] == "chess960")
    plies = []
    pending_fails: list[dict] = []

    def fail_entry(r: dict) -> dict:
        raw = r.get("raw_output") or ""
        return {
            "class": r["parse_class"],
            "candidate": r.get("candidate"),
            # The MOVE line sits at the end of the reply; keep the tail.
            "excerpt": raw[-300:],
        }

    for r in recs:
        if r["type"] == "attempt":
            if r["parse_class"] == "legal":
                move = board.parse_san(r["move_san"])
                frm = chess.square_name(move.from_square)
                to = chess.square_name(move.to_square)
                board.push(move)
                plies.append(
                    {
                        "san": r["move_san"],
                        "by": "llm",
                        "from": frm,
                        "to": to,
                        "fen": board.fen(),
                        "fails": pending_fails,
                        "eval": r.get("eval_cp_white_before"),
                    }
                )
                pending_fails = []
            else:
                pending_fails.append(fail_entry(r))
        elif r["type"] in ("engine_move", "prefix_move"):
            move = board.parse_uci(r["move_uci"])
            frm = chess.square_name(move.from_square)
            to = chess.square_name(move.to_square)
            san = r["move_san"]
            board.push(move)
            by = "engine" if r["type"] == "engine_move" else "prefix"
            plies.append(
                {"san": san, "by": by, "from": frm, "to": to, "fen": board.fen(), "fails": []}
            )

    return {
        "id": game["game_id"],
        "model": game["model"],
        "variant": game["variant"],
        "visibility": game["visibility"],
        "sp_id": game["sp_id"],
        "llm_color": game["llm_color"],
        "start_fen": game["start_fen"],
        "plies": plies,
        "final_fails": pending_fails,  # attempts that ended the game (forfeit)
        "termination": game["termination"],
        "llm_result": game["llm_result"],
        "result": game["result"],
        "first_event_ply": game.get("first_event_ply"),
        "survival_plies": game.get("survival_plies"),
    }


def load_run(run_dir: Path) -> list[dict]:
    games = [load_game(p) for p in sorted(run_dir.glob("*.jsonl"))]
    return [g for g in games if g is not None]


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chessbench viewer</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1c1c1c; --muted: #6b6b6b; --panel: #f4f4f4;
    --light-sq: #f0d9b5; --dark-sq: #b58863; --hl: rgba(205, 210, 30, 0.55);
    --accent: #2563eb; --illegal: #dc2626; --invalid: #6b7280;
    --ambiguous: #d97706; --truncated: #7c3aed;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #16181d; --fg: #e8e8e8; --muted: #9a9a9a; --panel: #22252c; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 -apple-system, "Segoe UI", sans-serif;
         background: var(--bg); color: var(--fg); }
  .wrap { display: flex; flex-wrap: wrap; gap: 20px; padding: 20px; max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 12px; margin-bottom: 12px; }
  select { font: inherit; padding: 6px 8px; border-radius: 6px; border: 1px solid var(--muted);
           background: var(--panel); color: var(--fg); max-width: 100%; }
  .board-col { flex: 0 0 auto; }
  .board { display: grid; grid-template-columns: repeat(8, 48px); grid-template-rows: repeat(8, 48px);
           border: 2px solid var(--dark-sq); border-radius: 4px; overflow: hidden; }
  .sq { display: flex; align-items: center; justify-content: center; font-size: 34px;
        position: relative; user-select: none; }
  .sq.light { background: var(--light-sq); } .sq.dark { background: var(--dark-sq); }
  .sq.hl::after { content: ""; position: absolute; inset: 0; background: var(--hl); }
  .pc { position: relative; z-index: 1; }
  .pc.w { color: #fff; text-shadow: 0 0 2px #000, 0 1px 1px #000; }
  .pc.b { color: #111; text-shadow: 0 0 2px #fff8; }
  .files { display: grid; grid-template-columns: repeat(8, 48px); color: var(--muted);
           font-size: 11px; text-align: center; margin-top: 2px; }
  .controls { display: flex; gap: 6px; margin-top: 10px; align-items: center; }
  .controls button { font: inherit; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--muted);
                     background: var(--panel); color: var(--fg); cursor: pointer; }
  .controls button:hover { border-color: var(--accent); }
  .plyinfo { margin-top: 8px; font-size: 13px; color: var(--muted); min-height: 20px; }
  .side { flex: 1 1 320px; min-width: 280px; }
  .meta { background: var(--panel); border-radius: 8px; padding: 12px; font-size: 13px; margin-bottom: 12px; }
  .meta b { font-weight: 600; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 9px; font-size: 11px;
           color: #fff; margin-left: 4px; }
  .badge.illegal { background: var(--illegal); } .badge.invalid { background: var(--invalid); }
  .badge.ambiguous { background: var(--ambiguous); } .badge.truncated { background: var(--truncated); }
  .fails { background: var(--panel); border-left: 3px solid var(--illegal); border-radius: 6px;
           padding: 10px 12px; margin: 10px 0; font-size: 13px; }
  .fails .ex { color: var(--muted); font-family: ui-monospace, monospace; font-size: 11px;
               white-space: pre-wrap; word-break: break-word; margin-top: 4px; max-height: 90px; overflow-y: auto; }
  .moves { background: var(--panel); border-radius: 8px; padding: 10px 12px;
           max-height: 380px; overflow-y: auto; line-height: 2; }
  .mv { cursor: pointer; padding: 2px 5px; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 13px; }
  .mv:hover { outline: 1px solid var(--accent); }
  .mv.cur { background: var(--accent); color: #fff; }
  .mv.hasfail { border-bottom: 2px solid var(--illegal); }
  .mvnum { color: var(--muted); font-size: 12px; margin-right: 1px; }
  @media (max-width: 560px) {
    .board { grid-template-columns: repeat(8, 11vw); grid-template-rows: repeat(8, 11vw); }
    .files { grid-template-columns: repeat(8, 11vw); }
    .sq { font-size: 7.5vw; }
  }
</style>
</head>
<body>
<div class="wrap">
  <div style="flex-basis: 100%">
    <h1>chessbench game viewer</h1>
    <div class="sub">__RUN_LABEL__</div>
    <select id="gamesel"></select>
  </div>
  <div class="board-col">
    <div class="board" id="board"></div>
    <div class="files"><span>a</span><span>b</span><span>c</span><span>d</span><span>e</span><span>f</span><span>g</span><span>h</span></div>
    <div class="controls">
      <button id="first" title="start">&#8676;</button>
      <button id="prev" title="back">&#8592;</button>
      <button id="play">&#9654;</button>
      <button id="next" title="forward">&#8594;</button>
      <button id="last" title="end">&#8677;</button>
      <select id="speed"><option value="1200">slow</option><option value="600" selected>normal</option><option value="250">fast</option></select>
    </div>
    <div class="plyinfo" id="plyinfo"></div>
  </div>
  <div class="side">
    <div class="meta" id="meta"></div>
    <div id="failbox"></div>
    <div class="moves" id="moves"></div>
  </div>
</div>
<script>
const GAMES = __GAMES_JSON__;
const GLYPH = {p:"\\u265F", n:"\\u265E", b:"\\u265D", r:"\\u265C", q:"\\u265B", k:"\\u265A"};
let cur = 0, idx = 0, timer = null;

function fenBoard(fen) {
  const rows = fen.split(" ")[0].split("/");
  const out = [];
  for (const row of rows) {
    const cells = [];
    for (const ch of row) {
      if (ch >= "1" && ch <= "8") for (let i = 0; i < +ch; i++) cells.push(null);
      else cells.push(ch);
    }
    out.push(cells);
  }
  return out; // [rank8 ... rank1]
}
function sqName(f, r) { return "abcdefgh"[f] + (8 - r); }

function render() {
  const g = GAMES[cur];
  const fen = idx === 0 ? g.start_fen : g.plies[idx - 1].fen;
  const cells = fenBoard(fen);
  const hl = idx > 0 ? [g.plies[idx - 1].from, g.plies[idx - 1].to] : [];
  const board = document.getElementById("board");
  board.innerHTML = "";
  for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
    const d = document.createElement("div");
    d.className = "sq " + ((r + f) % 2 ? "dark" : "light") + (hl.includes(sqName(f, r)) ? " hl" : "");
    const pc = cells[r][f];
    if (pc) {
      const s = document.createElement("span");
      const white = pc === pc.toUpperCase();
      s.className = "pc " + (white ? "w" : "b");
      s.textContent = GLYPH[pc.toLowerCase()];
      d.appendChild(s);
    }
    board.appendChild(d);
  }
  const info = document.getElementById("plyinfo");
  if (idx === 0) info.textContent = "start position";
  else {
    const p = g.plies[idx - 1];
    const who = p.by === "llm" ? g.model : (p.by === "prefix" ? "random opening" : "stockfish");
    info.textContent = "ply " + idx + ": " + p.san + " (" + who + ")"
      + (p.eval != null ? " \\u00b7 eval " + (p.eval / 100).toFixed(2) : "");
  }
  renderFails();
  const list = document.getElementById("moves");
  [...list.children].forEach((el, i) => el.classList.toggle("cur", i === idx - 1));
  if (idx > 0 && list.children[idx - 1]) list.children[idx - 1].scrollIntoView({block: "nearest"});
}

function renderFails() {
  const g = GAMES[cur];
  const box = document.getElementById("failbox");
  box.innerHTML = "";
  // fails attached to the NEXT ply happened from the currently shown position
  const fails = idx < g.plies.length ? g.plies[idx].fails
              : (idx === g.plies.length ? g.final_fails : []);
  for (const fl of fails) {
    const d = document.createElement("div");
    d.className = "fails";
    d.innerHTML = "tried <b>" + esc(fl.candidate ?? "(no move)") + "</b>"
      + '<span class="badge ' + fl.class + '">' + fl.class + "</span>"
      + '<div class="ex">' + esc(fl.excerpt) + "</div>";
    box.appendChild(d);
  }
  if (idx === g.plies.length && g.final_fails.length) {
    const d = document.createElement("div");
    d.className = "fails";
    d.innerHTML = "<b>game over:</b> " + esc(g.termination) + " \\u2192 " + esc(g.llm_result);
    box.appendChild(d);
  }
}

function esc(s) { const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; }

function pickGame(i) {
  cur = i; idx = 0; stop();
  const g = GAMES[cur];
  document.getElementById("meta").innerHTML =
    "<b>" + esc(g.model) + "</b> as " + g.llm_color + " \\u00b7 " + g.variant +
    (g.variant === "chess960" ? " (sp " + g.sp_id + ")" : "") + " \\u00b7 " + esc(g.visibility) +
    "<br>result: <b>" + esc(g.llm_result) + "</b> (" + esc(g.termination) + ", " + g.result + ")" +
    (g.first_event_ply != null ? "<br>first illegal/ambiguous event: ply " + g.first_event_ply : "<br>no illegal events");
  const list = document.getElementById("moves");
  list.innerHTML = "";
  g.plies.forEach((p, i2) => {
    const s = document.createElement("span");
    s.className = "mv" + (p.fails.length ? " hasfail" : "");
    s.innerHTML = (i2 % 2 === 0 ? '<span class="mvnum">' + (i2 / 2 + 1) + ".</span>" : "") + esc(p.san);
    s.onclick = () => { idx = i2 + 1; stop(); render(); };
    list.appendChild(s);
    list.appendChild(document.createTextNode(" "));
  });
  render();
}

function step(d) { const n = GAMES[cur].plies.length; idx = Math.max(0, Math.min(n, idx + d)); render(); }
function stop() { if (timer) { clearInterval(timer); timer = null; document.getElementById("play").innerHTML = "&#9654;"; } }
function playpause() {
  if (timer) { stop(); return; }
  document.getElementById("play").innerHTML = "&#9646;&#9646;";
  timer = setInterval(() => {
    if (idx >= GAMES[cur].plies.length) stop(); else step(1);
  }, +document.getElementById("speed").value);
}

const sel = document.getElementById("gamesel");
GAMES.forEach((g, i) => {
  const o = document.createElement("option");
  o.value = i;
  o.textContent = g.id + " \\u2014 " + g.llm_result;
  sel.appendChild(o);
});
sel.onchange = () => pickGame(+sel.value);
document.getElementById("first").onclick = () => { idx = 0; stop(); render(); };
document.getElementById("last").onclick = () => { idx = GAMES[cur].plies.length; stop(); render(); };
document.getElementById("prev").onclick = () => { stop(); step(-1); };
document.getElementById("next").onclick = () => { stop(); step(1); };
document.getElementById("play").onclick = playpause;
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") { stop(); step(-1); }
  if (e.key === "ArrowRight") { stop(); step(1); }
  if (e.key === " ") { e.preventDefault(); playpause(); }
});
if (GAMES.length) pickGame(0);
else document.body.insertAdjacentHTML("beforeend", "<p style='padding:20px'>no completed games found</p>");
</script>
</body>
</html>
"""


def build_html(games: list[dict], run_label: str) -> str:
    games_json = json.dumps(games, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__GAMES_JSON__", games_json).replace(
        "__RUN_LABEL__", html.escape(run_label)
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chessbench-viz", description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output HTML path (default: <run_dir>/viewer.html)")
    args = p.parse_args(argv)

    games = load_run(args.run_dir)
    out = args.out or (args.run_dir / "viewer.html")
    label = f"{args.run_dir} — {len(games)} games"
    out.write_text(build_html(games, label), encoding="utf-8")
    print(f"{out}: {len(games)} games embedded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
