#!/usr/bin/env python3
"""Render Qwen3-8B seed-sensitivity and thinking-mode figures as SVG/PNG/PDF."""

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
GRID = "#E6E8EC"
CARD = "#FCFCFB"
METHOD_COLOR = {"OPSD": BLUE, "LGSD": YELLOW}
PLOT_EPISODES = (16, 32, 48, 64)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def attrs(**values: object) -> str:
    rendered = []
    for key, value in values.items():
        if value is None or value == "":
            continue
        rendered.append(f'{key.replace("_", "-")}="{html.escape(str(value), quote=True)}"')
    return " ".join(rendered)


class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<defs>",
            '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">',
            '<feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#111111" flood-opacity="0.07"/>',
            "</filter>",
            "</defs>",
            f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
            '<g font-family="Inter,Arial,sans-serif">',
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def rect(self, x: float, y: float, width: float, height: float, **kw: object) -> None:
        self.add(f'<rect {attrs(x=x, y=y, width=width, height=height, **kw)}/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, **kw: object) -> None:
        self.add(f'<line {attrs(x1=x1, y1=y1, x2=x2, y2=y2, **kw)}/>')

    def circle(self, x: float, y: float, radius: float, **kw: object) -> None:
        self.add(f'<circle {attrs(cx=x, cy=y, r=radius, **kw)}/>')

    def polyline(self, points: list[tuple[float, float]], **kw: object) -> None:
        value = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(f'<polyline {attrs(points=value, **kw)}/>')

    def text(self, x: float, y: float, value: str, **kw: object) -> None:
        self.add(f'<text {attrs(x=x, y=y, **kw)}>{html.escape(value)}</text>')

    def finish(self) -> str:
        return "\n".join([*self.parts, "</g>", "</svg>", ""])


def panel(svg: Svg, x: float, width: float, letter: str, title: str, subtitle: str) -> None:
    svg.rect(x, 124, width, 486, rx=20, fill=CARD, stroke="#E4E1DA", stroke_width=1.1, filter="url(#shadow)")
    svg.circle(x + 29, 158, 15, fill=BLACK)
    svg.text(x + 29, 163, letter, fill=WHITE, font_size=15, font_weight=700, text_anchor="middle")
    svg.text(x + 53, 157, title, fill=BLACK, font_size=18, font_weight=750)
    svg.text(x + 53, 178, subtitle, fill=MUTED, font_size=11)


def accuracy_limits(*sets: list[dict[str, str]]) -> tuple[float, float, list[float]]:
    values = [float(row["strict_accuracy_pct"]) for dataset in sets for row in dataset]
    if not values:
        raise ValueError("accuracy inputs are empty")
    lower = max(0.0, math.floor((min(values) - 4.0) / 5.0) * 5.0)
    upper = min(100.0, math.ceil((max(values) + 4.0) / 5.0) * 5.0)
    if upper - lower < 20:
        lower = max(0.0, lower - 5.0)
        upper = min(100.0, upper + 5.0)
    step = 5.0 if upper - lower <= 25 else 10.0
    ticks = []
    value = math.ceil(lower / step) * step
    while value <= upper + 1e-9:
        ticks.append(value)
        value += step
    return lower, upper, ticks


def mean_by_episode(rows: list[dict[str, str]]) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["episode"]))].append(float(row["strict_accuracy_pct"]))
    output: dict[str, list[tuple[int, float]]] = {}
    for method in ("OPSD", "LGSD"):
        if all(grouped[(method, episode)] for episode in PLOT_EPISODES):
            output[method] = [
                (episode, sum(grouped[(method, episode)]) / len(grouped[(method, episode)]))
                for episode in PLOT_EPISODES
            ]
    return output


def draw_accuracy_panel(
    svg: Svg,
    card_x: float,
    rows: list[dict[str, str]],
    seed_field: str,
    letter: str,
    title: str,
    subtitle: str,
    y_limits: tuple[float, float, list[float]],
) -> None:
    rows = [row for row in rows if int(row["episode"]) in PLOT_EPISODES]
    panel(svg, card_x, 552, letter, title, subtitle)
    left, right, top, bottom = card_x + 70, card_x + 518, 214, 535
    lower, upper, yticks = y_limits
    xmap = lambda value: left + ((value - 16.0) / 48.0) * (right - left)
    ymap = lambda value: bottom - ((value - lower) / (upper - lower)) * (bottom - top)

    for episode in PLOT_EPISODES:
        x = xmap(episode)
        svg.line(x, top, x, bottom, stroke=GRID, stroke_width=1)
        svg.text(x, bottom + 24, str(episode), fill=MUTED, font_size=12, text_anchor="middle")
    for value in yticks:
        y = ymap(value)
        svg.line(left, y, right, y, stroke=GRID, stroke_width=1)
        svg.text(left - 10, y + 4, f"{value:.0f}", fill=MUTED, font_size=12, text_anchor="end")
    svg.line(left, bottom, right, bottom, stroke=BLACK, stroke_width=1.25)
    svg.line(left, top, left, bottom, stroke=BLACK, stroke_width=1.25)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row[seed_field])].append(row)
    for (method, _seed), group in grouped.items():
        group.sort(key=lambda row: int(row["episode"]))
        points = [(xmap(int(row["episode"])), ymap(float(row["strict_accuracy_pct"]))) for row in group]
        svg.polyline(points, fill="none", stroke=METHOD_COLOR[method], stroke_width=2.0, stroke_opacity=0.26,
                     stroke_linecap="round", stroke_linejoin="round")
        for x, y in points:
            svg.circle(x, y, 3.3, fill=METHOD_COLOR[method], fill_opacity=0.45, stroke=BLACK,
                       stroke_opacity=0.35, stroke_width=0.8)

    means = mean_by_episode(rows)
    for method, values in means.items():
        points = [(xmap(episode), ymap(value)) for episode, value in values]
        svg.polyline(points, fill="none", stroke=METHOD_COLOR[method], stroke_width=5.0,
                     stroke_linecap="round", stroke_linejoin="round")
        for x, y in points:
            svg.circle(x, y, 6.7, fill=METHOD_COLOR[method], stroke=BLACK, stroke_width=1.4)
        end_x, end_y = points[-1]
        label_y = end_y - 31 if method == "LGSD" else end_y + 9
        label = f"{method} {values[-1][1]:.1f}"
        svg.rect(end_x - 72, label_y, 72, 22, rx=6, fill=WHITE, stroke=METHOD_COLOR[method], stroke_width=1.3)
        svg.text(end_x - 36, label_y + 15, label, fill=BLACK, font_size=10.2, font_weight=700,
                 text_anchor="middle")

    svg.text((left + right) / 2, 579, "Training episode", fill=BLACK, font_size=13, text_anchor="middle")
    svg.text(card_x + 20, (top + bottom) / 2, "Overall strict accuracy (%)  ↑", fill=BLACK, font_size=12.5,
             text_anchor="middle", transform=f"rotate(-90 {card_x + 20} {(top + bottom) / 2})")
    svg.text(left, 598, "thin = individual seed · thick = mean", fill=MUTED, font_size=10.5)


def draw_cost_panel(svg: Svg, card_x: float, rows: list[dict[str, str]]) -> None:
    panel(svg, card_x, 552, "c", "Training time + checkpoint space", "Qwen3-8B · two new H100 training seeds")
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    metrics = [
        ("active_training_hours", "Active training time", "H100 h", 270),
        ("checkpoint_mib", "Episode-64 checkpoint", "MiB", 430),
    ]
    plot_left, plot_right = card_x + 84, card_x + 508
    for metric, label, unit, group_y in metrics:
        means = {
            method: sum(float(row[metric]) for row in by_method[method]) / len(by_method[method])
            for method in ("OPSD", "LGSD")
        }
        base = means["OPSD"]
        svg.text(card_x + 30, group_y - 49, label, fill=BLACK, font_size=14, font_weight=750)
        svg.text(card_x + 30, group_y - 30, "relative to OPSD mean", fill=MUTED, font_size=10.5)
        for index, method in enumerate(("OPSD", "LGSD")):
            y = group_y + index * 45
            ratio = means[method] / base
            width = ratio / 1.12 * (plot_right - plot_left)
            svg.text(plot_left - 12, y + 20, method, fill=BLACK, font_size=11.5, font_weight=700, text_anchor="end")
            svg.rect(plot_left, y, width, 31, rx=8, fill=METHOD_COLOR[method], stroke=BLACK, stroke_width=1.15)
            label_color = WHITE if method == "OPSD" else BLACK
            svg.text(plot_left + width - 8, y + 21, f"{means[method]:.2f} {unit}", fill=label_color,
                     font_size=11, font_weight=700, text_anchor="end")
            for row in by_method[method]:
                point_ratio = float(row[metric]) / base
                point_x = plot_left + point_ratio / 1.12 * (plot_right - plot_left)
                svg.circle(point_x, y + 15.5, 3.4, fill=WHITE, stroke=BLACK, stroke_width=1.0)
        one_x = plot_left + 1.0 / 1.12 * (plot_right - plot_left)
        svg.line(one_x, group_y - 8, one_x, group_y + 79, stroke=BLACK, stroke_width=1.0,
                 stroke_dasharray="4 4", stroke_opacity=0.65)
        svg.text(one_x, group_y + 94, "1×", fill=MUTED, font_size=10.5, text_anchor="middle")
    svg.text(card_x + 30, 585, "bars = mean · white dots = individual H100 runs", fill=MUTED, font_size=10.5)


def build_triptych(decoding: list[dict[str, str]], training: list[dict[str, str]], cost: list[dict[str, str]]) -> str:
    svg = Svg(1800, 670)
    decoding_plot = [row for row in decoding if int(row["episode"]) in PLOT_EPISODES]
    training_plot = [row for row in training if int(row["episode"]) in PLOT_EPISODES]
    limits = accuracy_limits(decoding_plot, training_plot)
    svg.text(43, 55, "Qwen3-8B accuracy across decoding and training seeds", fill=BLACK,
             font_size=28, font_weight=780)
    svg.text(43, 86, "DeepMath-64 training → AMC23/AIME24/AIME25 (143 problems) · thinking mode · no LLM judge",
             fill=MUTED, font_size=13.5)
    draw_accuracy_panel(svg, 35, decoding_plot, "decoding_seed", "a", "Decoding-seed sensitivity",
                        "fixed seed-0 training run · 3 decoding seeds", limits)
    draw_accuracy_panel(svg, 624, training_plot, "training_seed", "b", "Training-seed reproducibility",
                        "fixed decoding seed · reused seed-0 + 2 new H100 seeds", limits)
    draw_cost_panel(svg, 1213, cost)
    svg.text(1758, 651, "Yellow = LGSD · Blue = OPSD · point estimates; no error bars", fill=MUTED,
             font_size=10.5, text_anchor="end")
    return svg.finish()


def build_mode_comparison(rows: list[dict[str, str]]) -> str:
    rows = [row for row in rows if int(row["episode"]) in PLOT_EPISODES]
    svg = Svg(1320, 650)
    limits = accuracy_limits(rows)
    lower, upper, yticks = limits
    svg.text(42, 55, "Qwen3-8B: thinking vs non-thinking accuracy dynamics", fill=BLACK,
             font_size=28, font_weight=780)
    svg.text(42, 86, "Thinking: 3 matched seeds (2 new H100 + reused seed-0) · Non-thinking: one H100 seed · fixed decoding seed",
             fill=MUTED, font_size=13.5)

    for panel_index, method in enumerate(("OPSD", "LGSD")):
        card_x = 34 + panel_index * 642
        svg.rect(card_x, 124, 610, 478, rx=20, fill=CARD, stroke="#E4E1DA", stroke_width=1.1, filter="url(#shadow)")
        svg.rect(card_x, 124, 610, 7, rx=3.5, fill=METHOD_COLOR[method])
        svg.text(card_x + 28, 163, method, fill=BLACK, font_size=19, font_weight=780)
        svg.text(card_x + 28, 184, "overall strict accuracy (%)", fill=MUTED, font_size=11)
        left, right, top, bottom = card_x + 72, card_x + 570, 220, 523
        xmap = lambda value: left + ((value - 16.0) / 48.0) * (right - left)
        ymap = lambda value: bottom - ((value - lower) / (upper - lower)) * (bottom - top)
        for episode in PLOT_EPISODES:
            x = xmap(episode)
            svg.line(x, top, x, bottom, stroke=GRID, stroke_width=1)
            svg.text(x, bottom + 23, str(episode), fill=MUTED, font_size=12, text_anchor="middle")
        for value in yticks:
            y = ymap(value)
            svg.line(left, y, right, y, stroke=GRID, stroke_width=1)
            svg.text(left - 10, y + 4, f"{value:.0f}", fill=MUTED, font_size=12, text_anchor="end")
        svg.line(left, bottom, right, bottom, stroke=BLACK, stroke_width=1.25)
        svg.line(left, top, left, bottom, stroke=BLACK, stroke_width=1.25)

        method_rows = [row for row in rows if row["method"] == method]
        thinking = [row for row in method_rows if row["mode"] == "thinking"]
        nonthinking = [row for row in method_rows if row["mode"] == "non-thinking"]
        by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in thinking:
            by_seed[row["training_seed"]].append(row)
        for seed_rows in by_seed.values():
            seed_rows.sort(key=lambda row: int(row["episode"]))
            points = [(xmap(int(row["episode"])), ymap(float(row["strict_accuracy_pct"]))) for row in seed_rows]
            svg.polyline(points, fill="none", stroke=METHOD_COLOR[method], stroke_width=1.9, stroke_opacity=0.25)
        thinking_mean = mean_by_episode(thinking)[method]
        thinking_points = [(xmap(episode), ymap(value)) for episode, value in thinking_mean]
        svg.polyline(thinking_points, fill="none", stroke=METHOD_COLOR[method], stroke_width=5.0,
                     stroke_linecap="round", stroke_linejoin="round")
        for x, y in thinking_points:
            svg.circle(x, y, 6.5, fill=METHOD_COLOR[method], stroke=BLACK, stroke_width=1.3)
        nonthinking.sort(key=lambda row: int(row["episode"]))
        nonthinking_points = [(xmap(int(row["episode"])), ymap(float(row["strict_accuracy_pct"]))) for row in nonthinking]
        svg.polyline(nonthinking_points, fill="none", stroke=BLACK, stroke_width=3.2, stroke_dasharray="9 6")
        for x, y in nonthinking_points:
            svg.circle(x, y, 5.6, fill=WHITE, stroke=BLACK, stroke_width=1.7)

        legend_y = 202
        svg.line(card_x + 250, legend_y, card_x + 283, legend_y, stroke=METHOD_COLOR[method], stroke_width=4.2)
        svg.text(card_x + 291, legend_y + 4, "thinking mean (n=3)", fill=BLACK, font_size=10.5)
        svg.line(card_x + 424, legend_y, card_x + 457, legend_y, stroke=BLACK, stroke_width=2.6, stroke_dasharray="7 5")
        svg.text(card_x + 465, legend_y + 4, "non-thinking (n=1)", fill=BLACK, font_size=10.5)
        svg.text((left + right) / 2, 579, "Training episode", fill=BLACK, font_size=13, text_anchor="middle")
    svg.text(1280, 628, "Non-thinking is descriptive because only one training seed is available.", fill=MUTED,
             font_size=10.5, text_anchor="end")
    return svg.finish()


def convert(svg_path: Path, output_path: Path, file_format: str, width: int, height: int) -> None:
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        return
    command = [renderer, "--format", file_format, "--output", str(output_path)]
    if file_format == "png":
        command.extend(["--width", str(width * 2), "--height", str(height * 2)])
    command.append(str(svg_path))
    subprocess.run(command, check=True)


def save_bundle(output_dir: Path, stem: str, content: str, width: int, height: int) -> None:
    svg_path = output_dir / f"{stem}.svg"
    svg_path.write_text(content, encoding="utf-8")
    convert(svg_path, output_dir / f"{stem}.png", "png", width, height)
    convert(svg_path, output_dir / f"{stem}.pdf", "pdf", width, height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoding = read_csv(args.input_dir / "decoding_seed_accuracy.csv")
    training = read_csv(args.input_dir / "training_seed_accuracy.csv")
    mode = read_csv(args.input_dir / "thinking_mode_accuracy.csv")
    cost = read_csv(args.input_dir / "resource_cost.csv")
    save_bundle(args.output_dir, "fig_qwen3_8b_seed_stability_cost", build_triptych(decoding, training, cost), 1800, 670)
    save_bundle(args.output_dir, "fig_qwen3_8b_thinking_mode_comparison", build_mode_comparison(mode), 1320, 650)


if __name__ == "__main__":
    main()
