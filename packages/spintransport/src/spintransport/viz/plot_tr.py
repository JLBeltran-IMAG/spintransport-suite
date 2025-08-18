# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.viz.plot_tr
# Brief : Plot T(E) and/or R(E) per component, optionally restricting energy indices.
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
---------------------------------------------------------------------------------------------
| T/R plotting CLI: modes T | R | TR, uses E_eV.npy if available or energy index otherwise. |
---------------------------------------------------------------------------------------------

Plot T(E) and/or R(E) per component from a simulation result folder.

Behavior
--------
- CLI:
    --dir           : results folder (must contain psi.npy and meta.json)
    --mode          : one of {T, R, TR}
    --eindex-range  : optional inclusive energy index range "I0 I1"
- If T.npy/R.npy exist, they are loaded.
  Otherwise they are computed from psi.npy using yL/yR (or a centered fallback)
  and saved back to disk (T.npy, R.npy, Ttot.npy, Rtot.npy) — totals are not plotted.
- X-axis prefers E_eV.npy when present and consistent; otherwise uses energy index.

Modes
-----
T  : plot T(E) only
R  : plot R(E) only
TR : plot both T(E) and R(E) on the same axis (T solid, R dashed)

Component order is [HH↑, LH↑, LH↓, HH↓] with consistent colors/styles.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from spintransport.io.simio import read_sim
from spintransport.analysis.tr_curves import (
    compute_or_load_TR,
    COMP_LABELS,
    COMP_COLORS,
    COMP_STYLES
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot T(E)/R(E) per component (no totals).")
    p.add_argument("--dir", type=Path, required=True, help="Folder with exported results.")
    p.add_argument("--mode", choices=["T", "R", "TR"], default="TR",
                   help="T: Transmission only | R: Reflection only | TR: both overlaid.")
    p.add_argument("--eindex-range", nargs=2, type=int, metavar=("I0", "I1"),
                   help="Inclusive energy-index range to plot (e.g., 10 60).")
    return p


def _normalize_range(nE: int, i0: int, i1: int) -> Tuple[int, int]:
    """Clamp to [0, nE-1] and ensure i0 <= i1."""
    a = max(0, min(int(i0), nE - 1))
    b = max(0, min(int(i1), nE - 1))
    if a > b:
        a, b = b, a
    return a, b


def main():
    args = build_parser().parse_args()
    sim = read_sim(args.dir, mmap=True)  # validates psi.npy + meta.json

    # Load or compute TR arrays
    T, R, _Ttot, _Rtot, x_axis, source = compute_or_load_TR(sim)  # totals ignored in plotting
    nE = T.shape[1]
    print(f"[i] TR data source: {source} ({sim.base_dir})")

    # Optional energy-index window
    if args.eindex_range is not None:
        i0, i1 = _normalize_range(nE, args.eindex_range[0], args.eindex_range[1])
        sl = slice(i0, i1 + 1)  # inclusive
        T = T[:, sl]
        R = R[:, sl]
        x_axis = x_axis[sl]
        xlab_suffix = f" (indices {i0}–{i1})"
    else:
        xlab_suffix = ""

    xlabel = ("Energy (eV)" if (sim.base_dir / "E_eV.npy").exists() and len(x_axis) == T.shape[1]
              else "Energy index") + xlab_suffix

    plt.style.use("default")

    if args.mode == "T":
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.6))
        for i in range(4):
            ax.plot(x_axis, T[i], label=f"T {COMP_LABELS[i]}",
                    color=COMP_COLORS[i], ls=COMP_STYLES[i], lw=1.8)
        ax.set_xlabel(xlabel); ax.set_ylabel("T(E)")
        ax.set_title("Transmission per component")
        ax.grid(True); ax.legend(loc="best")
        plt.tight_layout(); plt.show()
        return

    if args.mode == "R":
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.6))
        for i in range(4):
            ax.plot(x_axis, R[i], label=f"R {COMP_LABELS[i]}",
                    color=COMP_COLORS[i], ls=COMP_STYLES[i], lw=1.8)
        ax.set_xlabel(xlabel); ax.set_ylabel("R(E)")
        ax.set_title("Reflection per component")
        ax.grid(True); ax.legend(loc="best")
        plt.tight_layout(); plt.show()
        return

    # args.mode == "TR" → overlay T and R on the same axis
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.0))
    for i in range(4):
        # T: solid
        ax.plot(x_axis, T[i], label=f"T {COMP_LABELS[i]}",
                color=COMP_COLORS[i], ls="-", lw=1.8)
        # R: dashed (same color)
        ax.plot(x_axis, R[i], label=f"R {COMP_LABELS[i]}",
                color=COMP_COLORS[i], ls="--", lw=1.6, alpha=0.9)

    ax.set_xlabel(xlabel); ax.set_ylabel("Coefficient")
    ax.set_title("T(E) and R(E) per component")
    ax.grid(True); ax.legend(loc="best", ncols=2)
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
