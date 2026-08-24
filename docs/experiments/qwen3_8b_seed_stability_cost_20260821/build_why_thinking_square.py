#!/usr/bin/env python3
"""Render a square, single-method explanation for Qwen3-8B thinking mode."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_figures import BLACK, GRID, MUTED, WHITE, YELLOW, Svg, convert


EPISODES = (16, 32, 48, 64)
BLUE = "#2563EB"
PALE_YELLOW = "#FFF7CC"
PALE_BLUE = "#EEF4FF"
PALE_GRAY = "#F5F5F3"


def load_series(path: Path) -> tuple[list[float], list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def values(mode: str, condition: str) -> list[float]:
        selected = {
            int(row["episode"]): float(row["strict_accuracy_pct"])
            for row in rows
            if row["mode"] == mode
            and row["training_condition"] == condition
            and row["method"] == "OPSD"
            and int(row["episode"]) in EPISODES
        }
        if set(selected) != set(EPISODES):
            raise ValueError(f"incomplete {mode} series: {selected}")
        return [selected[episode] for episode in EPISODES]

    return values("thinking", "thinking_seed0"), values("non-thinking", "nothinking_seed0")


def multiline(svg: Svg, x: float, y: float, lines: list[str], *, size: float = 18,
              fill: str = BLACK, weight: int = 450, line_height: float = 28) -> None:
    for index, line in enumerate(lines):
        svg.text(x, y + index * line_height, line, fill=fill, font_size=size, font_weight=weight)


def explanation_card(svg: Svg, x: float, number: str, title: str, lines: list[str],
                     fill: str, accent: str) -> None:
    svg.rect(x, 835, 330, 245, rx=22, fill=fill, stroke="#DDDAD2", stroke_width=1.2,
             filter="url(#shadow)")
    svg.circle(x + 37, 879, 22, fill=accent)
    svg.text(x + 37, 886, number, fill=WHITE if accent != YELLOW else BLACK,
             font_size=19, font_weight=800, text_anchor="middle")
    svg.text(x + 72, 886, title, fill=BLACK, font_size=21, font_weight=780)
    multiline(svg, x + 28, 934, lines, size=17, fill="#344054", line_height=27)


def build(thinking: list[float], nonthinking: list[float]) -> str:
    svg = Svg(1200, 1200)
    svg.text(60, 76, "Why Qwen3-8B uses thinking mode", fill=BLACK, font_size=39, font_weight=800)
    svg.text(60, 116, "DeepMath-64 training → AMC23/AIME24/AIME25 · 143 problems · strict accuracy",
             fill=MUTED, font_size=17)

    svg.rect(55, 155, 1090, 610, rx=26, fill="#FCFCFB", stroke="#E1DED6", stroke_width=1.2,
             filter="url(#shadow)")
    svg.text(88, 204, "Thinking preserves accuracy on multi-step math", fill=BLACK,
             font_size=25, font_weight=780)
    svg.text(88, 236, "same model · same training method · same seed · same decoding seed",
             fill=MUTED, font_size=15)

    gap = thinking[-1] - nonthinking[-1]
    svg.rect(820, 184, 286, 70, rx=18, fill=PALE_YELLOW, stroke=YELLOW, stroke_width=1.8)
    svg.text(846, 211, "EPISODE 64", fill=MUTED, font_size=12, font_weight=750)
    svg.text(846, 239, f"+{gap:.2f} pp", fill=BLACK, font_size=26, font_weight=850)
    svg.text(1016, 238, "with thinking", fill=BLACK, font_size=14, font_weight=650)

    left, right, top, bottom = 160.0, 1030.0, 300.0, 665.0
    ymin, ymax = 20.0, 70.0
    xmap = lambda episode: left + (episode - 16.0) / 48.0 * (right - left)
    ymap = lambda value: bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    for value in (20, 30, 40, 50, 60, 70):
        y = ymap(value)
        svg.line(left, y, right, y, stroke=GRID, stroke_width=1.2)
        svg.text(left - 18, y + 6, str(value), fill=MUTED, font_size=15, text_anchor="end")
    for episode in EPISODES:
        x = xmap(episode)
        svg.line(x, top, x, bottom, stroke=GRID, stroke_width=1.0)
        svg.text(x, bottom + 31, str(episode), fill=MUTED, font_size=15, text_anchor="middle")
    svg.line(left, bottom, right, bottom, stroke=BLACK, stroke_width=1.5)
    svg.line(left, top, left, bottom, stroke=BLACK, stroke_width=1.5)
    svg.text((left + right) / 2, 730, "Training episode", fill=BLACK, font_size=17, text_anchor="middle")
    svg.text(85, (top + bottom) / 2, "Strict accuracy (%)  ↑", fill=BLACK, font_size=17,
             text_anchor="middle", transform=f"rotate(-90 85 {(top + bottom) / 2})")

    thinking_points = [(xmap(ep), ymap(value)) for ep, value in zip(EPISODES, thinking)]
    direct_points = [(xmap(ep), ymap(value)) for ep, value in zip(EPISODES, nonthinking)]
    svg.polyline(thinking_points, fill="none", stroke=YELLOW, stroke_width=8,
                 stroke_linecap="round", stroke_linejoin="round")
    svg.polyline(direct_points, fill="none", stroke=BLACK, stroke_width=5,
                 stroke_dasharray="14 10", stroke_linecap="round", stroke_linejoin="round")
    for x, y in thinking_points:
        svg.circle(x, y, 10, fill=YELLOW, stroke=BLACK, stroke_width=2)
    for x, y in direct_points:
        svg.circle(x, y, 9, fill=WHITE, stroke=BLACK, stroke_width=2.5)

    svg.line(183, 273, 233, 273, stroke=YELLOW, stroke_width=7, stroke_linecap="round")
    svg.text(245, 279, "Thinking", fill=BLACK, font_size=16, font_weight=720)
    svg.line(355, 273, 405, 273, stroke=BLACK, stroke_width=4, stroke_dasharray="11 8")
    svg.text(417, 279, "Non-thinking", fill=BLACK, font_size=16, font_weight=650)

    tx, ty = thinking_points[-1]
    nx, ny = direct_points[-1]
    svg.rect(tx - 180, ty - 55, 180, 37, rx=10, fill=WHITE, stroke=YELLOW, stroke_width=2)
    svg.text(tx - 90, ty - 30, f"Thinking  {thinking[-1]:.2f}%", fill=BLACK, font_size=15,
             font_weight=780, text_anchor="middle")
    svg.rect(nx - 202, ny + 18, 202, 37, rx=10, fill=WHITE, stroke=BLACK, stroke_width=1.8)
    svg.text(nx - 101, ny + 43, f"Non-thinking  {nonthinking[-1]:.2f}%", fill=BLACK, font_size=15,
             font_weight=720, text_anchor="middle")

    svg.text(60, 811, "WHY THIS PROTOCOL", fill=MUTED, font_size=14, font_weight=800,
             letter_spacing=1.6)
    explanation_card(
        svg, 55, "1", "Multi-step demand",
        ["AMC/AIME solutions require", "dependent algebra, case analysis,", "and answer verification."],
        PALE_GRAY, BLACK,
    )
    explanation_card(
        svg, 435, "2", "Reasoning workspace",
        ["Thinking mode gives Qwen3-8B", "room for intermediate steps before", "committing to a final answer."],
        PALE_BLUE, BLUE,
    )
    explanation_card(
        svg, 815, "3", "Observed consequence",
        ["Thinking rises to 66.43%;", "non-thinking falls to 24.48%", "after 64 training episodes."],
        PALE_YELLOW, YELLOW,
    )
    svg.text(1145, 1145,
             "Protocol motivation, not a causal decomposition · one matched seed · no LLM judge",
             fill=MUTED, font_size=13, text_anchor="end")
    return svg.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).resolve().parent / "thinking_mode_accuracy.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thinking, nonthinking = load_series(args.input)
    stem = "fig_qwen3_8b_why_thinking_square"
    svg_path = args.output_dir / f"{stem}.svg"
    svg_path.write_text(build(thinking, nonthinking), encoding="utf-8")
    convert(svg_path, args.output_dir / f"{stem}.png", "png", 1200, 1200)
    convert(svg_path, args.output_dir / f"{stem}.pdf", "pdf", 1200, 1200)


if __name__ == "__main__":
    main()
