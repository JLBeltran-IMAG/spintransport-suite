# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.viz.time_evolution
# Brief : Interactive viewer (Matplotlib slider) for |psi|^2 time evolution by component.
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
Interactive time-evolution viewer for |psi(y,t;E)|^2 with a Matplotlib Slider,
plus optional PNG snapshot export.

Layouts:
- 'shared': HH and LH plotted on the same y-axis (all positive).
- 'split' : HH plotted positive; LH plotted negative (mirror of positive values).

Snapshots:
- Use --snap <t> (repeatable) to export PNG snapshots at specific time indices.
  Files are saved as: snapshot_E{E}_t{t}_{layout}.png in --snap-outdir (default: results dir).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Iterable, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from spintransport.io.simio import read_sim


def _build_barrier(ax, yL: int | None, yR: int | None, dy_A: float):
    """Draw a translucent span indicating the barrier region, if indices are provided."""
    if yL is None or yR is None:
        return None
    xL, xR = yL * dy_A, yR * dy_A
    return ax.axvspan(xL, xR, alpha=0.15, color="green", lw=0, label="barrier")


def _compute_ylim_for_energy(psi_E: np.ndarray, layout: str) -> Tuple[float, float]:
    """
    Compute a sensible (ymin, ymax) for one energy slice: psi_E shape (4*ny, nt).

    - 'shared': all curves are >= 0 → use (0, 1.1*max).
    - 'split' : LH curves are negative mirror → use symmetric limits.
    """
    ny = psi_E.shape[0] // 4
    hh_up = np.abs(psi_E[0 * ny:1 * ny, :]) ** 2
    hh_dn = np.abs(psi_E[3 * ny:4 * ny, :]) ** 2
    lh_up = np.abs(psi_E[1 * ny:2 * ny, :]) ** 2
    lh_dn = np.abs(psi_E[2 * ny:3 * ny, :]) ** 2
    peak = float(max(hh_up.max(), hh_dn.max(), lh_up.max(), lh_dn.max()) + 1e-30)
    if layout == "split":
        return (-1.1 * peak, 1.1 * peak)
    return (0.0, 1.1 * peak)


def _prepare_lines(ax,
                   x_A: np.ndarray,
                   psi: np.ndarray,
                   ny: int,
                   t0: int,
                   E: int,
                   layout: str):
    """
    Create line objects for the initial time slice t0. Returns the list of lines and a closure updater.
    In 'split' layout, LH curves are plotted as the negative mirror of their positive magnitudes.
    """
    # initial data
    hh_up = np.abs(psi[0 * ny:1 * ny, t0, E]) ** 2
    hh_dn = np.abs(psi[3 * ny:4 * ny, t0, E]) ** 2
    lh_up = np.abs(psi[1 * ny:2 * ny, t0, E]) ** 2
    lh_dn = np.abs(psi[2 * ny:3 * ny, t0, E]) ** 2

    if layout == "split":
        # HH positive, LH negative (mirror)
        y_hh_up = hh_up
        y_hh_dn = hh_dn
        y_lh_up = -lh_up
        y_lh_dn = -lh_dn
        lh_labels = ("LH↑ (neg)", "LH↓ (neg)")
    else:
        # Shared axis: all positive
        y_hh_up = hh_up
        y_hh_dn = hh_dn
        y_lh_up = lh_up
        y_lh_dn = lh_dn
        lh_labels = ("LH↑", "LH↓")

    l1, = ax.plot(x_A, y_hh_up, lw=1.6, color="tab:red",   label="HH↑")
    l2, = ax.plot(x_A, y_hh_dn, lw=1.6, color="tab:blue",  label="HH↓", ls="--")
    l3, = ax.plot(x_A, y_lh_up, lw=1.2, color="tab:green", label=lh_labels[0])
    l4, = ax.plot(x_A, y_lh_dn, lw=1.2, color="tab:purple",label=lh_labels[1], ls="--")
    lines = [l1, l2, l3, l4]

    def _update(time_idx: int):
        hh_up = np.abs(psi[0 * ny:1 * ny, time_idx, E]) ** 2
        hh_dn = np.abs(psi[3 * ny:4 * ny, time_idx, E]) ** 2
        lh_up = np.abs(psi[1 * ny:2 * ny, time_idx, E]) ** 2
        lh_dn = np.abs(psi[2 * ny:3 * ny, time_idx, E]) ** 2

        if layout == "split":
            y = (hh_up, hh_dn, -lh_up, -lh_dn)
        else:
            y = (hh_up, hh_dn, lh_up, lh_dn)

        for ln, ydata in zip(lines, y):
            ln.set_ydata(ydata)
        return lines  # useful for blitting

    return lines, _update


def _save_snapshot(sim, E: int, t_idx: int, layout: str, style: str, outdir: Path, dpi: int) -> Path:
    """
    Render and save a static PNG snapshot at time index t_idx for energy E.
    """
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    # Axis and barrier
    x_A = np.arange(sim.ny) * float(sim.dy_A)
    _build_barrier(ax, sim.yL, sim.yR, sim.dy_A)

    # Limits
    psi_E = sim.psi[:, :, E]
    ylo, yhi = _compute_ylim_for_energy(psi_E, layout)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(0.0, x_A[-1] if len(x_A) else 1.0)

    # Plot lines for this t
    ny = sim.ny
    hh_up = np.abs(sim.psi[0 * ny:1 * ny, t_idx, E]) ** 2
    hh_dn = np.abs(sim.psi[3 * ny:4 * ny, t_idx, E]) ** 2
    lh_up = np.abs(sim.psi[1 * ny:2 * ny, t_idx, E]) ** 2
    lh_dn = np.abs(sim.psi[2 * ny:3 * ny, t_idx, E]) ** 2

    if layout == "split":
        y = (hh_up, hh_dn, -lh_up, -lh_dn)
        labels = ("HH↑", "HH↓", "LH↑ (neg)", "LH↓ (neg)")
    else:
        y = (hh_up, hh_dn, lh_up, lh_dn)
        labels = ("HH↑", "HH↓", "LH↑", "LH↓")

    ax.plot(x_A, y[0], lw=1.6, color="tab:red",   label=labels[0])
    ax.plot(x_A, y[1], lw=1.6, color="tab:blue",  label=labels[1], ls="--")
    ax.plot(x_A, y[2], lw=1.2, color="tab:green", label=labels[2])
    ax.plot(x_A, y[3], lw=1.2, color="tab:purple",label=labels[3], ls="--")

    # Labels & title
    ax.set_xlabel(r"$L\ (\AA)$", fontsize=12)
    ax.set_ylabel(r"$|\psi|^2$", fontsize=12)
    title_e = f"E index {E}/{sim.nE-1}"
    if sim.energies_eV is not None and 0 <= E < len(sim.energies_eV):
        title_e += f"  (E ≈ {float(sim.energies_eV[E]):.3f} eV)"
    if sim.dt_fs is not None:
        ax.set_title(f"Snapshot — {title_e} — t = {t_idx * sim.dt_fs:.2f} fs")
    else:
        ax.set_title(f"Snapshot — {title_e} — t index = {t_idx}")
    ax.grid(True)
    ax.legend(loc="best")

    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"snapshot_E{E}_t{t_idx}_{layout}.png"
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outfile


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive time-evolution viewer for |psi|^2 (+ PNG snapshots).")
    p.add_argument("--dir", type=Path, required=True, help="Folder with exported results (psi.npy, meta.json).")
    p.add_argument("--energy-index", type=int, default=None, help="Energy index to visualize (default: middle).")
    p.add_argument("--layout", choices=["shared", "split"], default="shared",
                   help="shared: HH/LH on same y-axis; split: HH positive, LH negative (mirror).")
    p.add_argument("--style", default="default", help="Matplotlib style (e.g., 'seaborn-v0_8-paper').")
    p.add_argument("--no-mmap", action="store_true", help="Disable numpy memmap when loading psi.npy.")
    # Snapshots
    p.add_argument("--snap", type=int, action="append", default=[],
                   help="Export a PNG snapshot at this time index (repeatable).")
    p.add_argument("--snap-outdir", type=Path, default=None,
                   help="Directory to write snapshots (default: simulation folder).")
    p.add_argument("--png-dpi", type=int, default=140, help="DPI for PNG snapshots.")
    return p


def main():
    args = build_parser().parse_args()
    sim = read_sim(args.dir, mmap=not args.no_mmap)

    # Choose energy index
    E = sim.nE // 2 if args.energy_index is None else max(0, min(int(args.energy_index), sim.nE - 1))

    # ---------- Optional snapshot exports ----------
    if args.snap:
        outdir = args.snap_outdir or sim.base_dir
        for t_idx in args.snap:
            if not (0 <= t_idx < sim.nt):
                raise ValueError(f"--snap {t_idx} is out of range [0, {sim.nt-1}].")
            outfile = _save_snapshot(sim, E, t_idx, args.layout, args.style, outdir, dpi=args.png_dpi)
            print(f"[✓] Snapshot saved: {outfile}")

    # ---------- Interactive viewer ----------
    # Prepare x-axis in Angstrom
    x_A = np.arange(sim.ny) * float(sim.dy_A)

    # Style
    plt.style.use(args.style)

    # Figure layout: main axis + slider axis
    fig = plt.figure(figsize=(10.5, 5.5))
    ax = fig.add_axes((0.08, 0.18, 0.88, 0.75))   # left, bottom, width, height
    ax_slider = fig.add_axes((0.08, 0.08, 0.88, 0.05))

    # Barrier overlay (if available in meta)
    _build_barrier(ax, sim.yL, sim.yR, sim.dy_A)

    # Y-limits
    psi_E = sim.psi[:, :, E]  # (4*ny, nt)
    ymin, ymax = _compute_ylim_for_energy(psi_E, layout=args.layout)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0.0, x_A[-1] if len(x_A) else 1.0)

    # Lines and updater
    lines, updater = _prepare_lines(ax, x_A, sim.psi, sim.ny, t0=0, E=E, layout=args.layout)

    # Labels & title
    ax.set_xlabel(r"$L\ (\AA)$", fontsize=12)
    ax.set_ylabel(r"$|\psi|^2$", fontsize=12)
    title_e = f"E index {E}/{sim.nE-1}"
    if sim.energies_eV is not None and 0 <= E < len(sim.energies_eV):
        title_e += f"  (E ≈ {float(sim.energies_eV[E]):.3f} eV)"
    ax.set_title(f"Time evolution — {title_e}")
    ax.grid(True)
    ax.legend(loc="best")

    # Time label (fs if known)
    if sim.dt_fs is not None:
        time_txt = ax.text(0.02, 0.94, f"t = 0.00 fs", transform=ax.transAxes)
    else:
        time_txt = ax.text(0.02, 0.94, f"t index = 0", transform=ax.transAxes)

    # Slider (integer ticks over [0, nt-1])
    s_time = Slider(
        ax=ax_slider,
        label="time index",
        valmin=0,
        valmax=sim.nt - 1,
        valinit=0,
        valstep=1,
    )

    def _on_change(val):
        t_idx = int(s_time.val)
        updater(t_idx)
        if sim.dt_fs is not None:
            time_txt.set_text(f"t = {t_idx * sim.dt_fs:.2f} fs")
        else:
            time_txt.set_text(f"t index = {t_idx}")
        fig.canvas.draw_idle()

    s_time.on_changed(_on_change)

    # Keyboard shortcuts for convenience
    def _on_key(event):
        if event.key in ("left", "right"):
            cur = int(s_time.val)
            step = -1 if event.key == "left" else 1
            new = max(0, min(sim.nt - 1, cur + step))
            if new != cur:
                s_time.set_val(new)

    fig.canvas.mpl_connect("key_press_event", _on_key)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
