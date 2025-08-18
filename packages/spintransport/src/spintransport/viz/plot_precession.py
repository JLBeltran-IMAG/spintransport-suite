# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.viz.plot_precession
# Brief : Plot ⟨σx,y,z⟩(t) in left/barrier/right/full regions for HH/LH, abs/cond modes.
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
---------------------------------------------------------------------------------------------
| Precession plotting CLI with energy selection (mid/avg/i0:i1), arrival mark, and styling. |
---------------------------------------------------------------------------------------------

Thin CLI wrapper for spin precession plotting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spintransport.io.simio import read_sim
from spintransport.analysis.precession import run_precession


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spin precession in a region (minimal CLI).")
    p.add_argument("--dir", type=Path, required=True, help="Results folder (psi.npy, meta.json).")
    p.add_argument("--pair", choices=["hh", "lh"], default="hh", help="Band pair to analyze.")
    p.add_argument("--region", choices=["left", "barrier", "right", "full"], default="right",
                   help="Spatial region to integrate.")
    p.add_argument("--mode", choices=["abs", "cond"], default="abs",
                   help="abs: ∫ψ†σψ/||ψ(0)||²; cond: ∫ψ†σψ/∫|ψ|².")
    p.add_argument("--energy", default="mid",
                   help="Energy selector: 'mid' | 'avg' | '<i>' | '<i0>:<i1>' (index range).")
    p.add_argument("--no-mark", action="store_true", help="Do not draw the arrival marker.")
    p.add_argument("--style", default="default", help="Matplotlib style (e.g., 'seaborn-v0_8-paper').")
    p.add_argument("--save", type=Path, default=None, help="Optional PNG output path.")
    return p


def main():
    args = build_parser().parse_args()
    sim = read_sim(args.dir, mmap=True)  # validates presence of psi.npy and meta.json
    run_precession(
        sim=sim,
        pair=args.pair,
        region=args.region,
        mode=args.mode,
        energy_arg=args.energy,
        mark_arrival=(not args.no_mark),
        style=args.style,
        save=args.save,
    )


if __name__ == "__main__":
    main()
