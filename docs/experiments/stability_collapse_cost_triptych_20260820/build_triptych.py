#!/usr/bin/env python3
"""Render the Qwen3-8B / GPT-OSS-20B stability-cost triptych.

The renderer intentionally uses only the Python standard library.  It writes a
publication-ready SVG and, when librsvg is available, deterministic PNG/PDF copies.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


BLACK = "#111111"
BLUE = "#2563EB"
YELLOW = "#FACC15"
WHITE = "#FFFFFF"
MUTED = "#667085"
GRID = "#E5E7EB"
CARD = "#FAFAF8"

METHOD_COLOR = {"OPSD": BLUE, "LGSD": YELLOW}
MODEL_DASH = {"GPT-OSS-20B": "", "Qwen3-8B": "8 6"}

WIDTH = 1800
HEIGHT = 680


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def attrs(**kwargs: object) -> str:
    rendered = []
    for key, value in kwargs.items():
        if value is None or value == "":
            continue
        rendered.append(f'{key.replace("_", "-")}="{html.escape(str(value), quote=True)}"')
    return " ".join(rendered)


class Svg:
    def __init__(self) -> None:
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            "<defs>",
            '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">',
            '<feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#111111" flood-opacity="0.08"/>',
            "</filter>",
            '<marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">',
            f'<path d="M0,0 L8,4 L0,8 Z" fill="{BLACK}"/>',
            "</marker>",
            "</defs>",
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{WHITE}"/>',
        ]

    def add(self, element: str) -> None:
        self.parts.append(element)

    def rect(self, x: float, y: float, w: float, h: float, **kw: object) -> None:
        self.add(f'<rect {attrs(x=x, y=y, width=w, height=h, **kw)}/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, **kw: object) -> None:
        self.add(f'<line {attrs(x1=x1, y1=y1, x2=x2, y2=y2, **kw)}/>')

    def circle(self, x: float, y: float, radius: float, **kw: object) -> None:
        self.add(f'<circle {attrs(cx=x, cy=y, r=radius, **kw)}/>')

    def polyline(self, points: list[tuple[float, float]], **kw: object) -> None:
        value = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(f'<polyline {attrs(points=value, **kw)}/>')

    def text(self, x: float, y: float, value: str, **kw: object) -> None:
        self.add(f'<text {attrs(x=x, y=y, **kw)}>{html.escape(value)}</text>')

    def multiline(self, x: float, y: float, lines: list[str], line_height: float = 17, **kw: object) -> None:
        spans = []
        for index, value in enumerate(lines):
            dy = 0 if index == 0 else line_height
            spans.append(f'<tspan x="{x}" dy="{dy}">{html.escape(value)}</tspan>')
        self.add(f'<text {attrs(x=x, y=y, **kw)}>{"".join(spans)}</text>')

    def finish(self) -> str:
        return "\n".join([*self.parts, "</svg>", ""])


def panel_card(svg: Svg, x: float, title: str, letter: str) -> None:
    svg.rect(x, 122, 552, 492, rx=18, fill=CARD, stroke="#E6E3DC", stroke_width=1.2, filter="url(#shadow)")
    svg.circle(x + 28, 154, 14, fill=BLACK)
    svg.text(x + 28, 159, letter, fill=WHITE, font_size=15, font_weight=700, text_anchor="middle")
    svg.text(x + 51, 160, title, fill=BLACK, font_size=18, font_weight=700)


def grid(svg: Svg, left: float, top: float, right: float, bottom: float, xticks: list[float], yticks: list[float],
         xmap, ymap, xformat, yformat) -> None:
    for value in xticks:
        x = xmap(value)
        svg.line(x, top, x, bottom, stroke=GRID, stroke_width=1)
        svg.text(x, bottom + 23, xformat(value), fill=MUTED, font_size=12, text_anchor="middle")
    for value in yticks:
        y = ymap(value)
        svg.line(left, y, right, y, stroke=GRID, stroke_width=1)
        svg.text(left - 10, y + 4, yformat(value), fill=MUTED, font_size=12, text_anchor="end")
    svg.line(left, bottom, right, bottom, stroke=BLACK, stroke_width=1.3)
    svg.line(left, top, left, bottom, stroke=BLACK, stroke_width=1.3)


def token_radius(tokens: float) -> float:
    return 5.2 + 1.5 * math.log10(max(tokens, 1.0))


def marker(svg: Svg, model: str, x: float, y: float, radius: float, color: str) -> None:
    if model == "Qwen3-8B":
        svg.rect(x - radius, y - radius, 2 * radius, 2 * radius, rx=1.8, fill=color, stroke=BLACK, stroke_width=1.4)
    else:
        svg.circle(x, y, radius, fill=color, stroke=BLACK, stroke_width=1.4)


def draw_decoding(svg: Svg, card_x: float, rows: list[dict[str, str]]) -> None:
    panel_card(svg, card_x, "Collapse cause + decoding stability", "a")
    left, right, top, bottom = card_x + 68, card_x + 526, 202, 528
    xmap = lambda value: left + (value / 80.0) * (right - left)
    ymap = lambda value: bottom - (value / 80.0) * (bottom - top)
    grid(svg, left, top, right, bottom, [0, 20, 40, 60, 80], [0, 20, 40, 60, 80], xmap, ymap,
         lambda v: f"{int(v)}", lambda v: f"{int(v)}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["method"])].append(row)
    for (model, method), group in grouped.items():
        points = [(xmap(float(row["cap_hit_rate_pct"])), ymap(float(row["strict_accuracy_pct"]))) for row in group]
        if len(points) > 1:
            svg.polyline(points, fill="none", stroke=METHOD_COLOR[method], stroke_width=2.0,
                         stroke_opacity=0.5, stroke_dasharray=MODEL_DASH[model])
        for row, (x, y) in zip(group, points):
            marker(svg, model, x, y, token_radius(float(row["median_generated_tokens"])), METHOD_COLOR[method])

    # Short, readable callouts for the two observed GPT-OPSD failure modes.
    cap_x, cap_y = xmap(76.923), ymap(0.699)
    svg.line(cap_x - 2, cap_y - 4, xmap(63), ymap(11), stroke=BLACK, stroke_width=1.2, marker_end="url(#arrow)")
    svg.rect(xmap(48), ymap(17.5), 116, 38, rx=7, fill=WHITE, stroke=BLACK, stroke_width=0.9)
    svg.multiline(xmap(48) + 58, ymap(17.5) + 15, ["cap-hitting", "median 10,240"], 14,
                  fill=BLACK, font_size=11, text_anchor="middle", font_weight=600)
    short_x, short_y = xmap(30.7), ymap(17.3)
    svg.line(short_x + 2, short_y - 3, xmap(43), ymap(31), stroke=BLACK, stroke_width=1.2, marker_end="url(#arrow)")
    svg.rect(xmap(39), ymap(39), 128, 38, rx=7, fill=WHITE, stroke=BLACK, stroke_width=0.9)
    svg.multiline(xmap(39) + 64, ymap(39) + 15, ["premature stop", "median 86–112"], 14,
                  fill=BLACK, font_size=11, text_anchor="middle", font_weight=600)

    svg.text((left + right) / 2, 579, "Cap-hit rate (%)  → worse", fill=BLACK, font_size=13, text_anchor="middle")
    svg.text(card_x + 18, (top + bottom) / 2, "Strict accuracy (%)  ↑", fill=BLACK, font_size=13,
             text_anchor="middle", transform=f"rotate(-90 {card_x + 18} {(top + bottom) / 2})")
    svg.text(left, 599, "marker area ∝ log median tokens", fill=MUTED, font_size=10.5)

    legend_x, legend_y = card_x + 344, 177
    marker(svg, "GPT-OSS-20B", legend_x, legend_y, 6, BLUE)
    svg.text(legend_x + 12, legend_y + 4, "OPSD", fill=BLACK, font_size=11)
    marker(svg, "GPT-OSS-20B", legend_x + 78, legend_y, 6, YELLOW)
    svg.text(legend_x + 90, legend_y + 4, "LGSD", fill=BLACK, font_size=11)
    marker(svg, "GPT-OSS-20B", legend_x + 150, legend_y, 5, WHITE)
    svg.text(legend_x + 160, legend_y + 4, "GPT", fill=BLACK, font_size=11)
    marker(svg, "Qwen3-8B", legend_x + 203, legend_y, 5, WHITE)
    svg.text(legend_x + 213, legend_y + 4, "Qwen", fill=BLACK, font_size=11)


def draw_training(svg: Svg, card_x: float, rows: list[dict[str, str]]) -> None:
    panel_card(svg, card_x, "Training stability", "b")
    left, right, top, bottom = card_x + 72, card_x + 526, 202, 528
    xmap = lambda value: left + ((value - 16.0) / 48.0) * (right - left)
    ymap = lambda value: bottom - (value / 0.024) * (bottom - top)
    grid(svg, left, top, right, bottom, [16, 32, 48, 64], [0, 0.005, 0.010, 0.015, 0.020], xmap, ymap,
         lambda v: f"{int(v)}", lambda v: f"{v:.3f}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["method"])].append(row)
    for (model, method), group in grouped.items():
        group.sort(key=lambda item: int(item["episode_end"]))
        points = [(xmap(int(item["episode_end"])), ymap(float(item["mean_target_kl"]))) for item in group]
        svg.polyline(points, fill="none", stroke=METHOD_COLOR[method], stroke_width=3.2,
                     stroke_linejoin="round", stroke_linecap="round", stroke_dasharray=MODEL_DASH[model])
        for x, y in points:
            marker(svg, model, x, y, 6.5, METHOD_COLOR[method])

    gpt_end_x, gpt_end_y = xmap(64), ymap(0.021481606439)
    svg.line(xmap(49), ymap(0.0202), gpt_end_x - 3, gpt_end_y + 2, stroke=BLACK, stroke_width=1.2,
             marker_end="url(#arrow)")
    svg.rect(xmap(26), ymap(0.0223), 150, 25, rx=7, fill=WHITE, stroke=BLACK, stroke_width=0.9)
    svg.text(xmap(26) + 75, ymap(0.0223) + 17, "GPT OPSD drift: 5.5×", fill=BLACK, font_size=11,
             font_weight=600, text_anchor="middle")

    svg.rect(xmap(29), ymap(0.0080), 163, 38, rx=7, fill=WHITE, stroke=YELLOW, stroke_width=1.5)
    svg.multiline(xmap(29) + 81.5, ymap(0.0080) + 15, ["LGSD stays near 0.004", "for both models"], 14,
                  fill=BLACK, font_size=11, font_weight=600, text_anchor="middle")
    svg.line(xmap(53), ymap(0.0060), xmap(63), ymap(0.0041), stroke=BLACK, stroke_width=1.2,
             marker_end="url(#arrow)")

    svg.text((left + right) / 2, 579, "Training episode", fill=BLACK, font_size=13, text_anchor="middle")
    svg.text(card_x + 18, (top + bottom) / 2, "Mean target KL / 16 episodes  ↓", fill=BLACK, font_size=13,
             text_anchor="middle", transform=f"rotate(-90 {card_x + 18} {(top + bottom) / 2})")
    svg.text(right, 599, "solid/circle = GPT · dashed/square = Qwen", fill=MUTED, font_size=10.5, text_anchor="end")


def draw_cost(svg: Svg, card_x: float, rows: list[dict[str, str]]) -> None:
    panel_card(svg, card_x, "Time / space cost", "c")
    by_key = {(row["model"], row["method"]): row for row in rows}
    left, right, top, bottom = card_x + 143, card_x + 522, 208, 526
    xmap = lambda value: left + ((value - 0.8) / 3.25) * (right - left)

    for value in [1, 2, 3, 4]:
        x = xmap(value)
        svg.line(x, top, x, bottom, stroke=GRID if value != 1 else BLACK, stroke_width=1.1,
                 stroke_dasharray="4 4" if value == 1 else None)
        svg.text(x, bottom + 23, f"{value}×", fill=MUTED, font_size=12, text_anchor="middle")

    categories = [
        ("GPT-OSS-20B", "wall time", 246),
        ("GPT-OSS-20B", "checkpoint", 322),
        ("Qwen3-8B", "wall time", 419),
        ("Qwen3-8B", "checkpoint", 495),
    ]
    for model, metric, y in categories:
        opsd = by_key[(model, "OPSD")]
        lgsd = by_key[(model, "LGSD")]
        if metric == "wall time":
            blue_value = float(opsd["active_wall_hours"])
            yellow_value = float(lgsd["active_wall_hours"])
            actual = f"{blue_value:.2f} → {yellow_value:.2f} {opsd['accelerator']} h"
        else:
            blue_value = float(opsd["checkpoint_mib"])
            yellow_value = float(lgsd["checkpoint_mib"])
            actual = f"{blue_value:.2f} ≈ {yellow_value:.2f} MiB"
        ratio = yellow_value / blue_value
        x_blue, x_yellow = xmap(1.0), xmap(ratio)
        svg.line(x_blue, y, x_yellow, y, stroke=BLACK, stroke_width=2.0)
        svg.circle(x_blue, y, 9, fill=BLUE, stroke=BLACK, stroke_width=1.2)
        svg.circle(x_yellow, y, 6.2 if abs(ratio - 1) < 0.025 else 9, fill=YELLOW, stroke=BLACK, stroke_width=1.2)
        svg.text(card_x + 20, y - 6, model, fill=BLACK, font_size=12, font_weight=700)
        svg.text(card_x + 20, y + 12, metric, fill=MUTED, font_size=11)
        svg.text((x_blue + x_yellow) / 2, y - 14, actual, fill=BLACK, font_size=10.5, text_anchor="middle")

    ratio = float(by_key[("GPT-OSS-20B", "LGSD")]["active_wall_hours"]) / float(
        by_key[("GPT-OSS-20B", "OPSD")]["active_wall_hours"]
    )
    svg.rect(xmap(2.02), 346, 205, 45, rx=8, fill=WHITE, stroke=BLACK, stroke_width=0.9)
    svg.multiline(xmap(2.02) + 102.5, 363, ["GPT timing is not matched:", "collapse + 1→2 GPU layout"], 15,
                  fill=BLACK, font_size=10.5, text_anchor="middle", font_weight=600)
    svg.line(xmap(3.23), 346, xmap(ratio), 256, stroke=BLACK, stroke_width=1.2, marker_end="url(#arrow)")

    svg.text((left + right) / 2, 579, "Relative observed cost (OPSD = 1×)", fill=BLACK, font_size=13, text_anchor="middle")
    svg.multiline(card_x + 20, 594,
                  ["Peak allocated/device: Qwen 22.29→22.29 GiB", "GPT 63.42→40.69 GiB (1→2 GPUs)"],
                  13, fill=MUTED, font_size=10)


def render_svg(decoding: list[dict[str, str]], training: list[dict[str, str]], cost: list[dict[str, str]]) -> str:
    svg = Svg()
    svg.text(44, 56, "LGSD limits target drift and avoids GPT-OSS completion collapse", fill=BLACK, font_size=29, font_weight=750)
    svg.text(44, 88,
             "Across Qwen3-8B and GPT-OSS-20B, bounded targets stay local; GPT-OSS decoding remains stable across three seeds.",
             fill=MUTED, font_size=14)
    draw_decoding(svg, 35, decoding)
    draw_training(svg, 624, training)
    draw_cost(svg, 1213, cost)
    svg.text(1756, 653,
             "DeepMath training · AMC23/AIME24/AIME25 audit · one training run/method · point estimates",
             fill=MUTED, font_size=10.5, text_anchor="end")
    return svg.finish()


def convert(svg_path: Path, output_path: Path, file_format: str) -> None:
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        return
    command = [renderer, "--format", file_format, "--output", str(output_path)]
    if file_format == "png":
        command.extend(["--width", str(WIDTH * 2), "--height", str(HEIGHT * 2)])
    command.append(str(svg_path))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decoding = read_csv(args.input_dir / "panel_a_decoding_runs.csv")
    training = read_csv(args.input_dir / "panel_b_training_stability.csv")
    cost = read_csv(args.input_dir / "panel_c_resource_cost.csv")
    svg_path = args.output_dir / "fig_stability_collapse_cost_triptych.svg"
    svg_path.write_text(render_svg(decoding, training, cost), encoding="utf-8")
    convert(svg_path, args.output_dir / "fig_stability_collapse_cost_triptych.png", "png")
    convert(svg_path, args.output_dir / "fig_stability_collapse_cost_triptych.pdf", "pdf")


if __name__ == "__main__":
    main()
