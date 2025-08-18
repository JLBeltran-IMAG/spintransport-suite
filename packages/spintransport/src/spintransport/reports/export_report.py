# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.reports.export_report
# Brief : Export JSON report with transport metrics and precession highlights.
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
-----------------------------------------------------------------------------------
| Writes metrics_report.json from computed T/R and precession-derived quantities. |
-----------------------------------------------------------------------------------

Export a JSON report with key metrics for device design.

Minimal CLI:
    --dir : results folder (psi.npy, meta.json)

Optional (kept minimal; sensible defaults used if omitted):
    --Ef   : Fermi level in eV (default: mid of E grid if available)
    --T    : Temperature in K (default: 300)
    --spin-deg : use 2e^2/h instead of e^2/h (default: False)

Outputs:
    <dir>/metrics_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spintransport.io.simio import read_sim
from spintransport.analysis.metrics import device_metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export device metrics to JSON.")
    p.add_argument("--dir", type=Path, required=True, help="Results folder.")
    p.add_argument("--Ef", type=float, default=None, help="Fermi energy (eV). Default: mid of E grid if available.")
    p.add_argument("--T", type=float, default=300.0, help="Temperature (K). Default: 300.")
    p.add_argument("--spin-deg", action="store_true", help="Use 2e^2/h instead of e^2/h.")
    p.add_argument("--out", type=Path, default=None, help="Output JSON file (default: <dir>/metrics_report.json).")
    return p


def main():
    args = build_parser().parse_args()
    sim = read_sim(args.dir, mmap=True)

    report = device_metrics(
        sim,
        Ef_eV=args.Ef,
        T_K=args.T,
        spin_degeneracy=bool(args.spin_deg),
        precession_pair="hh",          # defaults consistent with prior usage
        precession_region="right",
        precession_mode="abs",
        precession_energy="mid",
    )

    out = args.out or (sim.base_dir / "metrics_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[✓] Report written to: {out}")


if __name__ == "__main__":
    main()
