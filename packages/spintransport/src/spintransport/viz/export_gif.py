# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.viz.export_gif
# Brief : Render |psi|^2 time evolution to animated GIF (Pillow writer).
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
--------------------------------------------------------------------------------
| GIF exporter; no ffmpeg required. Mirrors MP4 options (fps, stride, layout). |
--------------------------------------------------------------------------------

Export time evolution of |psi(y,t;E)|^2 to an animated GIF.

- Validates required files via `read_sim`.
- Layouts:
  * 'shared': HH & LH on the same (positive) y-axis.
  * 'split' : HH positive; LH negative (mirror of their positive magnitudes).
- Uses Matplotlib + PillowWriter (no ffmpeg needed).

Example:
    spintransport-export-gif --dir rashba_out --layout shared
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from spintransport.io.simio import read_sim
from spintransport.io.simio import read_sim


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export time evolution of |psi|^2 to GIF.")
    p.add_argument("--dir", type=Path, required=True, help="Folder with exported results (psi.npy, meta.json).")
    p.add_argument("--energy-index", type=int, default=None, help="Energy index to render (default: middle).")
    p.add_argument("--layout", choices=["shared", "split"], default="shared",
                   help="shared: HH/LH on same y-axis; split: HH positive, LH negative (mirror).")
    p.add_argument("--fps", type=int, default=20, help="Frames per second for the GIF.")
    p.add_argument("--stride", type=int, default=1, help="Use every STRIDE-th frame (>=1).")
    p.add_argument("--dpi", type=int, default=120, help="Figure DPI for the GIF.")
    p.add_argument("--style", default="default", help="Matplotlib style (e.g., 'seaborn-v0_8-paper').")
    p.add_argument("--outfile", type=Path, default=None, help="Output GIF filename. Default auto in --dir.")
    p.add_argument("--no-mmap", action="store_true", help="Disable numpy memmap when loading psi.npy.")
    return p


def _barrier(ax, yL: int | None, yR: int | None, dy_A: float):
    """Draw barrier span if indices are provided."""
    if yL is None or yR is None:
        return None
    xL, xR = yL * dy_A, yR * dy_A
    return ax.axvspan(xL, xR, alpha=0.15, color="green", lw=0, label="barrier")


def _compute_ylim(psi_E: np.ndarray, layout: str) -> Tuple[float, float]:
    """Return (ymin, ymax) for one energy slice psi_E with shape (4*ny, nt)."""
    ny = psi_E.shape[0] // 4
    hh_up = np.abs(psi_E[0 * ny:1 * ny, :]) ** 2
    hh_dn = np.abs(psi_E[3 * ny:4 * ny, :]) ** 2
    lh_up = np.abs(psi_E[1 * ny:2 * ny, :]) ** 2
    lh_dn = np.abs(psi_E[2 * ny:3 * ny, :]) ** 2
    peak = float(max(hh_up.max(), hh_dn.max(), lh_up.max(), lh_dn.max()) + 1e-30)
    if layout == "split":
        return (-1.1 * peak, 1.1 * peak)
    return (0.0, 1.1 * peak)


def main():
    args = build_parser().parse_args()
    sim = read_sim(args.dir, mmap=not args.no_mmap)
    E = sim.nE // 2 if args.energy_index is None else max(0, min(int(args.energy_index), sim.nE - 1))
    stride = max(1, int(args.stride))

    # X-axis in Angstrom
    x_A = np.arange(sim.ny) * float(sim.dy_A)
    psi_E = sim.psi[:, :, E]  # (4*ny, nt)

    # Frames
    frames = list(range(0, sim.nt, stride))
    if frames[-1] != sim.nt - 1:
        frames.append(sim.nt - 1)

    # Style & figure
    plt.style.use(args.style)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    _barrier(ax, sim.yL, sim.yR, sim.dy_A)

    # Limits
    ylo, yhi = _compute_ylim(psi_E, args.layout)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(0.0, x_A[-1] if len(x_A) else 1.0)

    # Initial data (t=0)
    ny = sim.ny
    hh_up = np.abs(sim.psi[0 * ny:1 * ny, 0, E]) ** 2
    hh_dn = np.abs(sim.psi[3 * ny:4 * ny, 0, E]) ** 2
    lh_up = np.abs(sim.psi[1 * ny:2 * ny, 0, E]) ** 2
    lh_dn = np.abs(sim.psi[2 * ny:3 * ny, 0, E]) ** 2

    if args.layout == "split":
        y_init = (hh_up, hh_dn, -lh_up, -lh_dn)
        labels = ("HH↑", "HH↓", "LH↑ (neg)", "LH↓ (neg)")
    else:
        y_init = (hh_up, hh_dn, lh_up, lh_dn)
        labels = ("HH↑", "HH↓", "LH↑", "LH↓")

    l1, = ax.plot(x_A, y_init[0], lw=1.6, color="tab:red",   label=labels[0])
    l2, = ax.plot(x_A, y_init[1], lw=1.6, color="tab:blue",  label=labels[1], ls="--")
    l3, = ax.plot(x_A, y_init[2], lw=1.2, color="tab:green", label=labels[2])
    l4, = ax.plot(x_A, y_init[3], lw=1.2, color="tab:purple",label=labels[3], ls="--")
    lines = [l1, l2, l3, l4]

    ax.set_xlabel(r"$L\ (\AA)$", fontsize=12)
    ax.set_ylabel(r"$|\psi|^2$", fontsize=12)

    title_e = f"E index {E}/{sim.nE-1}"
    if sim.energies_eV is not None and 0 <= E < len(sim.energies_eV):
        title_e += f"  (E ≈ {float(sim.energies_eV[E]):.3f} eV)"
    ax.set_title(f"Time evolution — {title_e}")
    ax.grid(True)
    ax.legend(loc="best")

    # Time textbox
    time_txt = ax.text(0.02, 0.92,
                       f"t = 0.00 fs" if sim.dt_fs is not None else "t index = 0",
                       transform=ax.transAxes)

    def update(tidx: int):
        hh_up = np.abs(sim.psi[0 * ny:1 * ny, tidx, E]) ** 2
        hh_dn = np.abs(sim.psi[3 * ny:4 * ny, tidx, E]) ** 2
        lh_up = np.abs(sim.psi[1 * ny:2 * ny, tidx, E]) ** 2
        lh_dn = np.abs(sim.psi[2 * ny:3 * ny, tidx, E]) ** 2
        if args.layout == "split":
            y = (hh_up, hh_dn, -lh_up, -lh_dn)
        else:
            y = (hh_up, hh_dn, lh_up, lh_dn)
        for ln, ydata in zip(lines, y):
            ln.set_ydata(ydata)

        time_txt.set_text(
            f"t = {tidx * sim.dt_fs:.2f} fs" if sim.dt_fs is not None else f"t index = {tidx}"
        )
        return (*lines, time_txt)

    writer = animation.PillowWriter(fps=args.fps)
    tag = f"{args.layout}"
    outfile = args.outfile or (sim.base_dir / f"evolution_E{E}_{tag}.gif")
    outfile.parent.mkdir(parents=True, exist_ok=True)

    print(f"[i] Writing GIF to: {outfile}")
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False, interval=1000 / max(1, args.fps))
    ani.save(str(outfile), writer=writer, dpi=args.dpi)
    print("[✓] Done.")


if __name__ == "__main__":
    main()
