# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.analysis.precession
# Brief : Spin precession expectations ⟨σx,y,z⟩ in spatial regions for HH/LH pairs.
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
Spin precession (〈σx〉, 〈σy〉, 〈σz〉) over a spatial region for HH or LH pairs.

Minimal CLI:
    --dir      : results folder (psi.npy, meta.json required)
    --pair     : hh | lh  (default: hh)
    --region   : left | barrier | right | full  (default: right)
    --mode     : abs | cond  (default: abs)
    --energy   : energy selector (default: mid)
                 formats:
                   - "mid"      → middle energy index
                   - "avg"      → average over all energies
                   - "37"       → a single energy index
                   - "10:30"    → inclusive index range to average

Behavior:
- ABS mode:   〈σ〉_abs(t)   = ∫_reg ψ†σψ / ||ψ(0)||²  → flat ~0 before wave arrival
- COND mode:  〈σ〉_cond(t)  = ∫_reg ψ†σψ / ∫_reg |ψ|² → local orientation (NaN before arrival)

Arrival detection:
- Uses a simple occupancy gate: occ_frac(t) = ∫_reg |ψ|² / ||ψ(0)||² averaged over selected energies.
- Threshold and run-length are fixed to sensible defaults (see constants).

This module uses `read_sim()` for IO validation and fails with clear errors if files are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from spintransport.io.simio import read_sim, SimData


# ---- Constants (tuned, not exposed as CLI) ----
EPS: float = 1e-30
ARRIVAL_FRAC: float = 1e-3    # occupancy fraction threshold
ARRIVAL_RUN: int = 3          # min consecutive frames above threshold to "arrive"


# ---- Core utilities ---------------------------------------------------------

def _choose_region(region: str, yL: int, yR: int, ny: int) -> slice:
    """Return a slice for the requested region."""
    region = region.lower()
    if region == "left":
        return slice(0, max(0, int(yL)))
    if region == "barrier":
        return slice(max(0, int(yL)), min(ny, int(yR)))
    if region == "right":
        return slice(min(ny, int(yR)), ny)
    if region == "full":
        return slice(0, ny)
    raise ValueError("region must be one of: left | barrier | right | full")


def _extract_pair(psi: np.ndarray, ny: int, pair: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (up, down) components for HH or LH pair as views: shape (ny, nt, nE)."""
    pair = pair.lower()
    if pair == "hh":
        return psi[0 * ny:1 * ny, :, :], psi[3 * ny:4 * ny, :, :]
    if pair == "lh":
        return psi[1 * ny:2 * ny, :, :], psi[2 * ny:3 * ny, :, :]
    raise ValueError("pair must be 'hh' or 'lh'")


def _numerators_and_dens(psi: np.ndarray, reg: slice, ny: int, pair: str):
    """
    Compute 〈σ〉 numerators and regional density.

    Returns
    -------
    sx_num, sy_num, sz_num, dens_reg : np.ndarray
        Each has shape (nt, nE).
    """
    u, d = _extract_pair(psi, ny, pair)  # (ny, nt, nE)
    u = u[reg, :, :]
    d = d[reg, :, :]

    dens_reg = (np.abs(u) ** 2 + np.abs(d) ** 2).sum(axis=0) + EPS  # (nt, nE)
    sx_num = (np.conj(u) * d + np.conj(d) * u).sum(axis=0)          # (nt, nE)
    sy_num = (-1j * np.conj(u) * d + 1j * np.conj(d) * u).sum(axis=0)
    sz_num = (np.abs(u) ** 2 - np.abs(d) ** 2).sum(axis=0)
    return sx_num, sy_num, sz_num, dens_reg


def _pair_norm_t0(psi: np.ndarray, ny: int, pair: str) -> np.ndarray:
    """Return ||ψ_pair(0)||² per energy: shape (nE,)."""
    u, d = _extract_pair(psi, ny, pair)
    return (np.abs(u[:, 0, :]) ** 2 + np.abs(d[:, 0, :]) ** 2).sum(axis=0) + EPS


def _first_run_over_threshold(x: np.ndarray, thr: float, run: int) -> int:
    """Return first t such that x[t:t+run] ≥ thr (simple hysteresis)."""
    n = x.size
    r = max(1, int(run))
    if r == 1:
        return int(np.argmax(x >= thr)) if np.any(x >= thr) else n - 1
    over = (x >= thr).astype(int)
    counts = np.convolve(over, np.ones(r, dtype=int), mode="same")
    idx = np.where(counts >= r)[0]
    return int(idx[0]) if idx.size else n - 1


# ---- Energy selection -------------------------------------------------------

@dataclass(frozen=True)
class EnergySelection:
    kind: str  # "single" | "avg" | "range" | "mid"
    indices: np.ndarray
    label: str


def _parse_energy_arg(nE: int, arg: str | None) -> EnergySelection:
    """
    Parse the --energy selector into a set of indices and a label.

    Accepts:
      - None or "mid"        → middle index
      - "avg"                → average over all
      - "37"                 → single index
      - "10:30"              → inclusive index range (clamped)
    """
    if arg is None or arg.lower() == "mid":
        idx = np.array([nE // 2], dtype=int)
        return EnergySelection("mid", idx, f"E_idx={idx[0]}")
    a = arg.strip().lower()
    if a == "avg" or a == "all":
        return EnergySelection("avg", np.arange(nE, dtype=int), "E: avg(all)")
    if ":" in a:
        s0, s1 = a.split(":", 1)
        try:
            i0 = max(0, min(int(s0), nE - 1))
            i1 = max(0, min(int(s1), nE - 1))
            if i0 > i1:
                i0, i1 = i1, i0
            idx = np.arange(i0, i1 + 1, dtype=int)
            return EnergySelection("range", idx, f"E_idx in [{i0},{i1}]")
        except Exception:
            pass
    # fallback: try single integer
    try:
        i = max(0, min(int(a), nE - 1))
        return EnergySelection("single", np.array([i], dtype=int), f"E_idx={i}")
    except Exception:
        # last resort: mid
        i = nE // 2
        return EnergySelection("mid", np.array([i], dtype=int), f"E_idx={i}")


# ---- Public API -------------------------------------------------------------

def run_precession(sim: SimData, pair: str, region: str, mode: str, energy_arg: Optional[str],
                   mark_arrival: bool = True, style: str = "default", save: Optional[Path] = None) -> None:
    """
    Compute and plot spin precession in a chosen region.

    Parameters
    ----------
    sim : SimData
        Loaded simulation data (from read_sim()).
    pair : str
        "hh" or "lh".
    region : str
        "left" | "barrier" | "right" | "full".
    mode : str
        "abs" or "cond" (see module docstring).
    energy_arg : Optional[str]
        Energy selector string, see _parse_energy_arg().
    mark_arrival : bool
        Whether to draw a vertical line at the detected arrival time.
    style : str
        Matplotlib style (default: "default").
    save : Optional[Path]
        If provided, save a PNG of the figure to this path.
    """
    if sim.yL is None or sim.yR is None:
        raise RuntimeError("meta.json must provide 'yL' and 'yR' for region-based precession.")

    # Region
    reg = _choose_region(region, sim.yL, sim.yR, sim.ny)

    # Numerators and density in region
    sx_num, sy_num, sz_num, dens_reg = _numerators_and_dens(sim.psi, reg, sim.ny, pair=pair)
    N0_pair = _pair_norm_t0(sim.psi, sim.ny, pair)

    # Energies selection
    sel = _parse_energy_arg(sim.nE, energy_arg)

    # Arrival (based on occupancy fraction averaged over selected energies)
    occ_frac_tE = dens_reg[:, sel.indices] / (N0_pair[None, sel.indices] + EPS)  # (nt, m)
    occ_frac = np.mean(occ_frac_tE, axis=1)                                      # (nt,)
    t0 = _first_run_over_threshold(occ_frac, ARRIVAL_FRAC, ARRIVAL_RUN)

    # Expectations vs time
    if mode.lower() == "abs":
        Sx = np.real(np.mean(sx_num[:, sel.indices] / (N0_pair[None, sel.indices] + EPS), axis=1))
        Sy = np.real(np.mean(sy_num[:, sel.indices] / (N0_pair[None, sel.indices] + EPS), axis=1))
        Sz = np.real(np.mean(sz_num[:, sel.indices] / (N0_pair[None, sel.indices] + EPS), axis=1))
        # flat baseline before arrival
        Sx[:t0] = 0.0; Sy[:t0] = 0.0; Sz[:t0] = 0.0
    elif mode.lower() == "cond":
        Sx = np.real(np.mean(sx_num[:, sel.indices] / (dens_reg[:, sel.indices] + EPS), axis=1))
        Sy = np.real(np.mean(sy_num[:, sel.indices] / (dens_reg[:, sel.indices] + EPS), axis=1))
        Sz = np.real(np.mean(sz_num[:, sel.indices] / (dens_reg[:, sel.indices] + EPS), axis=1))
        # masked before arrival (NaN gaps)
        Sx[:t0] = np.nan; Sy[:t0] = np.nan; Sz[:t0] = np.nan
    else:
        raise ValueError("mode must be 'abs' or 'cond'.")

    # Time axis (fs if available)
    if sim.dt_fs is not None:
        t = np.arange(sim.nt) * float(sim.dt_fs)
        txlab = "t (fs)"
        t_arrival_val = t[t0]
    else:
        t = np.arange(sim.nt)
        txlab = "t index"
        t_arrival_val = t0

    # Plot
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    ax.plot(t, Sx, label="⟨σx⟩")
    ax.plot(t, Sy, label="⟨σy⟩")
    ax.plot(t, Sz, label="⟨σz⟩")
    if mark_arrival:
        ax.axvline(t_arrival_val, color="k", ls=":", lw=1.2, label="arrival")

    ax.set_xlabel(txlab)
    ax.set_ylabel("expectation")
    ttl_pair = "HH" if pair.lower() == "hh" else "LH"
    ax.set_title(
        f"Spin precession ({ttl_pair}) — region={region}, {sel.label}, mode={mode}\n"
        f"occ_frac≥{ARRIVAL_FRAC:g} (run={ARRIVAL_RUN}), t_arrival≈{t_arrival_val:.3g}"
    )
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[✓] Saved: {save}")

    plt.show()
