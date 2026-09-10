#!/usr/bin/env python3
"""Render the context-to-supervision mechanism, with an explicitly synthetic example.

Requires matplotlib and numpy. Run from any directory; writes beside this file.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as PlotPath
import numpy as np


OUT = Path(__file__).resolve().parent
INK, MUTED, LINE = "#22333B", "#61717A", "#C8D2D6"
TEAL, RUST = "#187A80", "#B96845"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def main():
    fig = plt.figure(figsize=(12, 7.4), facecolor="white")
    top = fig.add_axes([.045, .59, .91, .35])
    top.set(xlim=(0, 1), ylim=(0, 1))
    top.axis("off")
    top.text(0, 1.04, "A   Context preferences become training targets", fontsize=14, weight="bold")
    xs, width, bottom, height = [0, .258, .516, .774], .226, .36, .47
    titles = ["Teacher context", "Dense supervision", "Local projection", "Shared model update"]
    equations = [r"$q_k=\pi_{\theta_k}(\cdot\mid x,c,y_{<t})$",
                 r"$u_k=\log q_k-\log p_k$",
                 r"$r_\alpha\propto p_k^{1-\alpha}q_k^\alpha$",
                 r"$\theta_k\ \longrightarrow\ \theta_{k+1}$"]
    notes = ["Additional reasoning instruction\nNo reference answer",
             "Context-induced preference\nNo outcome-based sign check",
             "Controls target distance\nNo correctness selector",
             "Student and teacher\nuse the updated parameters"]
    for i, x in enumerate(xs):
        top.add_patch(Rectangle((x, bottom), width, height, fc="white", ec=LINE, lw=1.15))
        top.text(x+width/2, .745, titles[i], ha="center", va="center", fontsize=11, weight="bold", color=TEAL if i==2 else INK)
        top.text(x+width/2, .602, equations[i], ha="center", va="center", fontsize=12 if i else 11)
        top.text(x+width/2, .455, notes[i], ha="center", va="center", fontsize=9.2, color=MUTED, linespacing=1.55)
        if i < 3:
            top.add_patch(FancyArrowPatch((x+width+.003,.6),(xs[i+1]-.006,.6),arrowstyle="-|>",mutation_scale=12,lw=1.1,color=MUTED))
    vertices = [(xs[-1]+width/2,.35),(xs[-1]+width/2,.15),(xs[0]+width/2,.15),(xs[0]+width/2,.35)]
    path = PlotPath(vertices,[PlotPath.MOVETO,PlotPath.LINETO,PlotPath.LINETO,PlotPath.LINETO])
    top.add_patch(FancyArrowPatch(path=path,arrowstyle="-|>",mutation_scale=12,lw=1.05,color=MUTED))
    top.text(.5,.195,"Next round: changed teacher parameters and student prefixes",ha="center",fontsize=9.8,color=MUTED,
             bbox=dict(facecolor="white",edgecolor="none",pad=3))
    top.text(.5,-.015,"Arrows show implemented dependencies; harmful feedback is a hypothesis about this loop.",ha="center",fontsize=9,color=MUTED)

    ax = fig.add_axes([.085,.105,.395,.365])
    ax.set_title("B   Shrinking a harmful preference",loc="left",fontsize=13,weight="bold",pad=24)
    alpha=np.linspace(0,1,101)
    p,q=.7,.4
    z=(1-alpha)*np.log(p/(1-p))+alpha*np.log(q/(1-q))
    r=1/(1+np.exp(-z))
    a=.25
    selected=1/(1+np.exp(-((1-a)*np.log(p/(1-p))+a*np.log(q/(1-q)))))
    ax.axhline(p,color=LINE,lw=1.2,ls=(0,(4,3)))
    ax.plot(alpha,r,color=RUST,lw=2.4)
    ax.scatter([0,1],[p,q],color=[INK,RUST],s=36,zorder=5)
    ax.scatter([a],[selected],color=TEAL,s=55,zorder=6)
    ax.annotate("Projected target\nα = 0.25;  0.6304",xy=(a,selected),xytext=(.04,.435),fontsize=10,color=TEAL,
                arrowprops=dict(arrowstyle="-",connectionstyle="arc3,rad=-0.15",color=TEAL,lw=1.1),linespacing=1.4)
    ax.text(.99,.711,"Current policy: 0.70",ha="right",va="bottom",fontsize=9.7,color=MUTED)
    ax.text(.97,.38,"Teacher: 0.40",ha="right",va="top",fontsize=9.7,color=RUST)
    ax.set(xlim=(-.015,1.015),ylim=(.33,.76),xticks=[0,.25,.5,.75,1],yticks=[.4,.5,.6,.7],
           xlabel="Projection strength  α",ylabel="Probability of a useful action")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(LINE)
    ax.grid(axis="y",color="#EDF0F2",lw=.8)
    ax.set_axisbelow(True)
    ax.text(0,1.025,"Synthetic example · not GPT-OSS measurements",transform=ax.transAxes,fontsize=9,color=MUTED)

    proof=fig.add_axes([.555,.105,.4,.365])
    proof.set(xlim=(0,1),ylim=(0,1));proof.axis("off")
    proof.text(0,1.10,"C   Local control preserves gradient direction",fontsize=13,weight="bold")
    proof.text(0,.80,r"$\nabla L_{\mathrm{LGSD}}(\theta_k)=\alpha_k\,\nabla L_{\mathrm{raw}}(\theta_k)$",fontsize=20,color=TEAL)
    proof.text(0,.64,"At an identical state: detached exponential target,\nunclipped reverse KL, one shared α.",fontsize=10,color=MUTED,linespacing=1.55,va="top")
    proof.text(0,.38,"A positive α preserves the local gradient direction.\nIt supplies no test of whether that direction\nimproves task correctness.",fontsize=11,linespacing=1.65,va="top")
    proof.text(0,.025,"Established: objective and parameter-sharing loop.\nUnresolved: the harmful GPT-OSS token preference.",fontsize=9.8,color=MUTED,linespacing=1.65,va="top")
    for extension in ("svg","pdf","png"):
        fig.savefig(OUT/f"privileged_context_mechanism.{extension}",dpi=155,bbox_inches="tight",facecolor="white")
    svg_path = OUT/"privileged_context_mechanism.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    with (OUT/"illustrative_target_path.csv").open("w") as handle:
        writer=csv.writer(handle, lineterminator="\n")
        writer.writerow(["alpha","student_useful_probability","teacher_useful_probability","target_useful_probability","data_type"])
        for strength,probability in zip(alpha,r):writer.writerow([float(strength),p,q,float(probability),"synthetic illustration"])
    print(f"Rendered figure and synthetic source data in {OUT}")


if __name__ == "__main__":
    main()
