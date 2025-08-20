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
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import FuncFormatter, ScalarFormatter


from spintransport.io.simio import read_sim, SimData


def _build_barrier(ax, yL: int | None, yR: int | None, dy_A: float):
    """Draw a translucent span indicating the barrier region, if indices are provided."""
    if yL is None or yR is None:
        return None
    xL, xR = yL * dy_A, yR * dy_A
    return ax.axvspan(xL, xR, alpha=0.5, color="green", lw=0, label="barrier")


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
                   E0: int,
                   layout: str):
    """
    Create line objects for the initial time slice t0. Returns the list of lines and a closure updater.
    In 'split' layout, LH curves are plotted as the negative mirror of their positive magnitudes.
    """
    # initial data
    hh_up = np.abs(psi[0 * ny:1 * ny, t0, E0]) ** 2
    hh_dn = np.abs(psi[3 * ny:4 * ny, t0, E0]) ** 2
    lh_up = np.abs(psi[1 * ny:2 * ny, t0, E0]) ** 2
    lh_dn = np.abs(psi[2 * ny:3 * ny, t0, E0]) ** 2

    if layout == "split":
        # HH positive, LH negative (mirror)
        y_hh_up = hh_up
        y_hh_dn = hh_dn
        y_lh_up = -lh_up
        y_lh_dn = -lh_dn
    else:
        # Shared axis: all positive
        y_hh_up = hh_up
        y_hh_dn = hh_dn
        y_lh_up = lh_up
        y_lh_dn = lh_dn

    l1, = ax.plot(x_A, y_hh_up, lw=1.6, color="tab:red",   label="HH↑")
    l2, = ax.plot(x_A, y_hh_dn, lw=1.6, color="tab:blue",  label="HH↓", ls="--")
    l3, = ax.plot(x_A, y_lh_up, lw=1.2, color="tab:green", label="LH↑")
    l4, = ax.plot(x_A, y_lh_dn, lw=1.2, color="tab:purple",label="LH↓", ls="--")
    lines = [l1, l2, l3, l4]

    def _update(time_idx: int, E_idx: int):
        hh_up = np.abs(psi[0 * ny:1 * ny, time_idx, E_idx]) ** 2
        hh_dn = np.abs(psi[3 * ny:4 * ny, time_idx, E_idx]) ** 2
        lh_up = np.abs(psi[1 * ny:2 * ny, time_idx, E_idx]) ** 2
        lh_dn = np.abs(psi[2 * ny:3 * ny, time_idx, E_idx]) ** 2

        if layout == "split":
            y = (hh_up, hh_dn, -lh_up, -lh_dn)
        else:
            y = (hh_up, hh_dn, lh_up, lh_dn)

        for ln, ydata in zip(lines, y):
            ln.set_ydata(ydata)
        return lines  # useful for blitting

    return lines, _update


def _save_snapshot(sim: SimData, E: int, t_idx: int, layout: str, style: str, outdir: Path, dpi: int) -> Path:
    """
    Render and save a static PNG snapshot at time index t_idx for energy index E.
    """
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(4, 3.2), layout="constrained")

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)

    # --- Axis & barrier
    x_A = np.arange(sim.ny) * float(sim.dy_A)
    _build_barrier(ax, sim.yL, sim.yR, sim.dy_A)

    # --- Limits
    psi_E = sim.psi[:, :, E]
    ylo, yhi = _compute_ylim_for_energy(psi_E, layout)
    ax.set_ylim(ylo, yhi)

    # Show positive labels on Y in split layout
    if layout == "split":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{abs(y):.3g}"))
    ax.set_xlim(0.0, x_A[-1] if len(x_A) else 1.0)

    # --- Data at t_idx
    ny = sim.ny
    hh_up = np.abs(sim.psi[0*ny:1*ny, t_idx, E])**2
    hh_dn = np.abs(sim.psi[3*ny:4*ny, t_idx, E])**2
    lh_up = np.abs(sim.psi[1*ny:2*ny, t_idx, E])**2
    lh_dn = np.abs(sim.psi[2*ny:3*ny, t_idx, E])**2

    labels = ("HH↑", "HH↓", "LH↑", "LH↓")
    if layout == "split":
        yvals  = (hh_up, hh_dn, -lh_up, -lh_dn)
    else:
        yvals  = (hh_up, hh_dn,  lh_up,  lh_dn)

    ax.plot(x_A, yvals[0], lw=1.6, color="tab:red",    label=labels[0])
    ax.plot(x_A, yvals[1], lw=1.6, color="tab:blue",   label=labels[1], ls="--")
    ax.plot(x_A, yvals[2], lw=1.2, color="tab:green",  label=labels[2])
    ax.plot(x_A, yvals[3], lw=1.2, color="tab:purple", label=labels[3], ls="--")

    ax.set_xlabel(r"$L\ (\AA)$", fontsize=12)
    ax.set_ylabel(r"$|\psi|^2$", fontsize=12)
    ax.grid(True)
    ax.legend(loc="best")

    # ---- Gather meta
    energies_eV = getattr(sim, "energies_eV", None)
    dt_fs       = getattr(sim, "dt_fs", None)
    dy_A        = float(getattr(sim, "dy_A", 1.0))
    nt          = int(getattr(sim, "nt", sim.psi.shape[1]))
    yL, yR      = getattr(sim, "yL", None), getattr(sim, "yR", None)

    # E
    line1_parts = []
    if energies_eV is not None and 0 <= E < len(energies_eV):
        line1_parts.append(rf"$E \approx {float(energies_eV[E]):.3f}\ \mathrm{{eV}}$")

    # time (fs o index)
    if dt_fs is not None:
        line1_parts.append(rf"$t = {t_idx*float(dt_fs):.2f}\ \mathrm{{fs}}$")
    else:
        line1_parts.append(rf"$t_\mathrm{{idx}} = {t_idx}$")
    line1 = " \u2014 ".join(line1_parts)  # em dash entre E y t

    # Lb, L, T
    L_total_A = sim.ny * dy_A if hasattr(sim, "ny") else None
    T_total_fs = float(dt_fs)*(nt-1) if dt_fs is not None else None

    # Lb from meta or from yR-yL
    Lb_A = None
    if hasattr(sim, "meta") and isinstance(sim.meta, dict):
        for k in ("Lb_A", "Lb_Å", "Lb_angstrom"):
            if k in sim.meta:
                try: Lb_A = float(sim.meta[k]); break
                except Exception: pass
    if Lb_A is None and (yL is not None and yR is not None):
        Lb_A = float(max(0, int(yR) - int(yL))) * dy_A

    line2_parts = []
    if Lb_A is not None:   line2_parts.append(rf"$L_b = {Lb_A:.1f}\ \AA$")
    if L_total_A is not None: line2_parts.append(rf"$L = {L_total_A:.0f}\ \AA$")
    if T_total_fs is not None: line2_parts.append(rf"$T = {T_total_fs:.1f}\ \mathrm{{fs}}$")
    line2 = " \u2014 ".join(line2_parts)

    # gamma_1,2,3 and beta_eff
    def _get_meta(key):
        if hasattr(sim, key): return getattr(sim, key)
        if hasattr(sim, "meta") and isinstance(sim.meta, dict): return sim.meta.get(key, None)
        return None

    gamma1 = _get_meta("gamma1")
    gamma2 = _get_meta("gamma2")
    gamma3 = _get_meta("gamma3")

    beta_eff = _get_meta("beta_eff_eVA") or _get_meta("beta_eff") or _get_meta("beta_eVA")

    line3_parts = []
    if gamma1 is not None and gamma2 is not None and gamma3 is not None:
        line3_parts.append(rf"$\gamma_1={float(gamma1):.3f},\ \gamma_2={float(gamma2):.3f},\ \gamma_3={float(gamma3):.3f}$")

    if beta_eff is not None:
        line3_parts.append(rf"$\beta_\mathrm{{eff}}={float(beta_eff):.3f}\ \mathrm{{eV}}\cdot\AA$")

    line3 = " \u2014 ".join(line3_parts)


    title_lines = [l for l in (line1, line2, line3) if l]
    if title_lines:
        ax.set_title("\n".join(title_lines))

    # ---- Save
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
    p.add_argument("--style", default="seaborn-v0_8-paper", help="Matplotlib style (e.g., 'seaborn-v0_8-paper').")
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
    fig = plt.figure(figsize=(10.5, 5.8))
    ax        = fig.add_axes((0.08, 0.24, 0.88, 0.70))  # main plot
    ax_time   = fig.add_axes((0.08, 0.13, 0.88, 0.05))  # time slider
    ax_energy = fig.add_axes((0.08, 0.06, 0.88, 0.05))  # energy slider

    # Barrier overlay (if available in meta)
    _build_barrier(ax, sim.yL, sim.yR, sim.dy_A)

    # Y-limits
    psi_E = sim.psi[:, :, E]  # (4*ny, nt)
    ymin, ymax = _compute_ylim_for_energy(psi_E, layout=args.layout)
    ax.set_ylim(ymin, ymax)

    if args.layout == "split":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{abs(y):.3g}"))

    ax.set_xlim(0.0, x_A[-1] if len(x_A) else 1.0)

    # Lines and updater
    lines, updater = _prepare_lines(ax, x_A, sim.psi, sim.ny, t0=0, E0=E, layout=args.layout)

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
        ax=ax_time,
        label="time",
        valmin=0,
        valmax=sim.nt - 1,
        valinit=0,
        valstep=1,
    )

    s_energy = Slider(
        ax=ax_energy,
        label="Energy",
        valmin=0,
        valmax=sim.nE - 1,
        valinit=E,
        valstep=1,
    )

    # Drawing the position of Barrier potential value on the energy slider
    # --------------------------------------------------------------------
    Vb: float | None = None
    if isinstance(sim.meta, dict):
        vb_val = sim.meta.get("Vb_eV")
        if vb_val is not None:
            try:
                Vb = float(vb_val)
            except (TypeError, ValueError):
                Vb = None

    if (Vb is not None) and (sim.energies_eV is not None) and (sim.nE > 0):
        e = np.asarray(sim.energies_eV, dtype=float)
        idx_vb = int(np.argmin(np.abs(e - Vb)))
        idx_vb = max(0, min(sim.nE - 1, idx_vb))
        ax_energy.axvline(idx_vb, color="green", lw=1.6, zorder=5)
    # --------------------------------------------------------------------

    def _update_all(_=None):
        t_idx = int(s_time.val)
        E_idx = int(s_energy.val)

        # actualizar curvas
        updater(t_idx, E_idx)

        # actualizar límites Y según energía
        ymin, ymax = _compute_ylim_for_energy(sim.psi[:, :, E_idx], layout=args.layout)
        ax.set_ylim(ymin, ymax)
        if args.layout == "split":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{abs(y):.3g}"))

        # actualizar título y etiqueta de tiempo
        title_e = f"E index {E_idx}/{sim.nE-1}"
        if sim.energies_eV is not None and 0 <= E_idx < len(sim.energies_eV):
            title_e += f"  (E ≈ {float(sim.energies_eV[E_idx]):.3f} eV)"
        ax.set_title(f"Time evolution — {title_e}")

        if sim.dt_fs is not None:
            time_txt.set_text(f"t = {t_idx * sim.dt_fs:.2f} fs")
        else:
            time_txt.set_text(f"t index = {t_idx}")

        # etiqueta del slider de energía
        fig.canvas.draw_idle()

    s_time.on_changed(_update_all)
    s_energy.on_changed(_update_all)

    # Keyboard shortcuts for convenience
    def _on_key(event):
        if event.key in ("left", "right"):
            cur = int(s_time.val)
            step = -1 if event.key == "left" else 1
            new = max(0, min(sim.nt - 1, cur + step))
            if new != cur:
                s_time.set_val(new)
        elif event.key in ("down", "up"):
            cur = int(s_energy.val)
            step = -1 if event.key == "down" else 1
            new = max(0, min(sim.nE - 1, cur + step))
            if new != cur:
                s_energy.set_val(new)

    fig.canvas.mpl_connect("key_press_event", _on_key)

    plt.show()


if __name__ == "__main__":
    main()
