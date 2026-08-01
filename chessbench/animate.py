"""Render completed games as animated GIFs.

Usage: uv run chessbench-anim runs/<name> [-o docs/media] [--square 56]

One GIF per completed game: board playback with last-move highlighting,
red "attempt" frames showing what the model tried before each failure, and
a closing result frame. Reuses the same game reconstruction as the HTML
viewer (chessbench.viz).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .viz import load_run

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
LIGHT_HL = (205, 210, 106)
DARK_HL = (170, 162, 58)
BG = (24, 26, 31)
FG = (232, 232, 232)
RED = (190, 38, 38)

GLYPH = {"p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}

_PIECE_FONT_CANDIDATES = [
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_TEXT_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
        except OSError:
            continue
        return font
    return ImageFont.load_default()


def _piece_font(size: int):
    for path in _PIECE_FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, size)
        except OSError:
            continue
        # Verify the font actually has chess glyphs.
        if font.getmask(GLYPH["k"]).getbbox() is not None:
            return font
    raise SystemExit(
        "no font with chess glyphs found — install one or extend _PIECE_FONT_CANDIDATES"
    )


def _fen_cells(fen: str) -> list[list[str | None]]:
    rows = []
    for row in fen.split(" ")[0].split("/"):
        cells: list[str | None] = []
        for ch in row:
            if ch.isdigit():
                cells.extend([None] * int(ch))
            else:
                cells.append(ch)
        rows.append(cells)
    return rows  # rank 8 first


class FrameRenderer:
    def __init__(self, square: int, header: str):
        self.sq = square
        self.header = header
        self.header_h = int(square * 0.55)
        self.caption_h = int(square * 0.65)
        self.w = square * 8
        self.h = self.header_h + square * 8 + self.caption_h
        self.piece_font = _piece_font(int(square * 0.78))
        self.text_font = _load_font(_TEXT_FONT_CANDIDATES, int(square * 0.30))

    def frame(self, fen: str, hl: tuple[str, str] | None, caption: str,
              caption_bg: tuple[int, int, int] = BG, border: tuple[int, int, int] | None = None) -> Image.Image:
        img = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(img)
        d.text((8, (self.header_h - int(self.sq * 0.30)) // 2), self.header,
               font=self.text_font, fill=FG)
        cells = _fen_cells(fen)
        hl_squares = set(hl) if hl else set()
        for r in range(8):
            for f in range(8):
                name = "abcdefgh"[f] + str(8 - r)
                dark = (r + f) % 2 == 1
                if name in hl_squares:
                    color = DARK_HL if dark else LIGHT_HL
                else:
                    color = DARK if dark else LIGHT
                x0, y0 = f * self.sq, self.header_h + r * self.sq
                d.rectangle([x0, y0, x0 + self.sq - 1, y0 + self.sq - 1], fill=color)
                pc = cells[r][f]
                if pc:
                    white = pc.isupper()
                    glyph = GLYPH[pc.lower()]
                    d.text(
                        (x0 + self.sq // 2, y0 + self.sq // 2),
                        glyph,
                        font=self.piece_font,
                        anchor="mm",
                        fill=(255, 255, 255) if white else (24, 24, 24),
                        stroke_width=2 if white else 1,
                        stroke_fill=(20, 20, 20) if white else (235, 235, 235),
                    )
        cap_y = self.header_h + 8 * self.sq
        d.rectangle([0, cap_y, self.w, self.h], fill=caption_bg)
        d.text((8, cap_y + (self.caption_h - int(self.sq * 0.30)) // 2), caption,
               font=self.text_font, fill=FG)
        if border:
            for i in range(4):
                d.rectangle([i, self.header_h + i, self.w - 1 - i, cap_y - 1 - i], outline=border)
        return img


def animate_game(game: dict, out_path: Path, square: int = 56,
                 ply_ms: int = 700, fail_ms: int = 1400, end_ms: int = 2500) -> None:
    header = f"{game['model']} · {game['variant']} · {game['visibility']}"
    rend = FrameRenderer(square, header)
    frames: list[Image.Image] = []
    durations: list[int] = []

    frames.append(rend.frame(game["start_fen"], None, "start position"))
    durations.append(ply_ms)

    prev_fen = game["start_fen"]
    for i, ply in enumerate(game["plies"], 1):
        if ply["fails"]:
            tried = ", ".join(f"{f['candidate'] or '(no move)'} ({f['class']})" for f in ply["fails"])
            frames.append(rend.frame(prev_fen, None, f"✗ tried: {tried}", RED, border=RED))
            durations.append(fail_ms)
        num = (i + 1) // 2
        dots = "." if i % 2 == 1 else "…"
        who = game["model"].split("/")[-1] if ply["by"] == "llm" else "stockfish"
        frames.append(rend.frame(ply["fen"], (ply["from"], ply["to"]), f"{num}{dots} {ply['san']}  ({who})"))
        durations.append(ply_ms)
        prev_fen = ply["fen"]

    if game["final_fails"]:
        tried = ", ".join(f"{f['candidate'] or '(no move)'} ({f['class']})" for f in game["final_fails"])
        frames.append(rend.frame(prev_fen, None, f"✗ tried: {tried}", RED, border=RED))
        durations.append(fail_ms)
    frames.append(rend.frame(prev_fen, None, f"{game['llm_result']} ({game['termination']})",
                             (50, 53, 60)))
    durations.append(end_ms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chessbench-anim", description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output directory (default: <run_dir>/anim)")
    p.add_argument("--square", type=int, default=56, help="square size in px")
    args = p.parse_args(argv)

    out_dir = args.out or (args.run_dir / "anim")
    games = load_run(args.run_dir)
    for g in games:
        out = out_dir / f"{g['id']}.gif"
        animate_game(g, out, square=args.square)
        print(f"{out} ({len(g['plies'])} plies)")
    if not games:
        print("no completed games found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
