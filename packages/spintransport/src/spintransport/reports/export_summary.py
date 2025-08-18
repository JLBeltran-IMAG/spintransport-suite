# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.reports.export_summary
# Brief : Export concise Markdown summary for device design docs.
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
----------------------------------------------------------------------------------------------------
| Generates summary.md: grid/barrier params, T/R highlights, conductance, and precession snapshot. |
----------------------------------------------------------------------------------------------------

Export a concise, human-readable design summary (Markdown).

Minimal CLI:
    --dir : results folder (psi.npy, meta.json)

Optional:
    --Ef, --T, --spin-deg   (same semantics as export_report)
    --out : output .md file (default: <dir>/summary.md)

Content:
- Key sim settings (from meta).
- TR highlights (Ttot stats & on/off ratio).
- Landauer conductance at (Ef, T).
- Spin precession summary (arrival time, final in-plane angle & amplitude).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spintransport.io.simio import read_sim
from spintransport.analysis.metrics import device_metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a concise design summary (Markdown).")
    p.add_argument("--dir", type=Path, required=True, help="Results folder.")
    p.add_argument("--Ef", type=float, default=None, help="Fermi energy (eV). Default: mid of E grid if available.")
    p.add_argument("--T", type=float, default=300.0, help="Temperature (K).")
    p.add_argument("--spin-deg", action="store_true", help="Use 2e^2/h instead of e^2/h.")
    p.add_argument("--out", type=Path, default=None, help="Output .md path (default: <dir>/summary.md).")
    return p


def main():
    args = build_parser().parse_args()
    sim = read_sim(args.dir, mmap=True)

    d = device_metrics(
        sim,
        Ef_eV=args.Ef,
        T_K=args.T,
        spin_degeneracy=bool(args.spin_deg),
    )
    meta = sim.meta

    # Pretty strings
    Ef_str = f"{d['conductance']['Ef_eV_used']:.4f} eV" if d['conductance']['Ef_eV_used'] is not None else "mid-index"
    G_uS = d["conductance"]["G_uS"]
    T_onoff = d["tr_stats"]["T_on_off_ratio"]
    T_mean = d["tr_stats"]["Ttot"]["mean"]; T_max = d["tr_stats"]["Ttot"]["max"]
    R_mean = d["tr_stats"]["Rtot"]["mean"]

    arrival = d["precession"]["arrival_time"]
    phi = d["precession"]["phi_final_rad"]
    amp = d["precession"]["inplane_amplitude"]

    md = []
    md.append(f"# SpinFET Design Summary\n")
    md.append(f"**Folder:** `{sim.base_dir}`\n")
    md.append(f"**Grid:** ny={sim.ny}, nt={sim.nt}, nE={sim.nE}, dy_A={meta.get('dy_A')}, dt_fs={meta.get('dt_fs')}\n")
    md.append(f"**Barrier:** yL={sim.yL}, yR={sim.yR}, Lb_A={meta.get('Lb_A')}, Vb_eV={meta.get('Vb_eV')}, beta_eff_eVA={meta.get('beta_eff_eVA')}\n")
    md.append("\n---\n")
    md.append("## Transport (T/R)\n")
    md.append(f"- ⟨T_tot⟩ = {T_mean:.3f},  T_max = {T_max:.3f},  ⟨R_tot⟩ = {R_mean:.3f}\n")
    md.append(f"- On/Off ratio (max/min T_tot) = **{T_onoff:.2f}**\n")
    md.append("\n## Conductance (Landauer)\n")
    md.append(f"- Temperature: **{d['conductance']['T_K_used']} K**\n")
    md.append(f"- Fermi level: **{Ef_str}**\n")
    md.append(f"- G ≈ **{G_uS:.2f} μS**  ({d['conductance']['G_e2_over_h']:.3f} × e²/h)\n")
    md.append("\n## Spin Precession\n")
    md.append(f"- Region: **{d['precession']['region']}**, Pair: **{d['precession']['pair']}**, Mode: **{d['precession']['mode']}**\n")
    md.append(f"- Energy selection: {d['precession']['energy_label']}\n")
    md.append(f"- Arrival time ≈ **{arrival:.3f}** {'fs' if sim.dt_fs is not None else 'steps'}\n")
    md.append(f"- Final in-plane angle φ ≈ **{phi:.3f} rad**, amplitude |S⊥| ≈ **{amp:.3f}**\n")

    out = args.out or (sim.base_dir / "summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(f"[✓] Summary written to: {out}")


if __name__ == "__main__":
    main()
