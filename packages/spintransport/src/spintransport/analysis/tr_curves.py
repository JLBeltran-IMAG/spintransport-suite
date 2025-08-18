# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.analysis.tr_curves
# Brief : Compute T(E) and R(E) per component (and totals) from psi(y,t;E).
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
Transmission/Reflection post-processing utilities.
"""


from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np

from spintransport.io.simio import read_sim, SimData


# Component labels/colors/styles match package basis: [hh↑, lh↑, lh↓, hh↓]
COMP_LABELS = ["HH↑", "LH↑", "LH↓", "HH↓"]
COMP_COLORS = ["tab:red", "tab:green", "tab:purple", "tab:blue"]
COMP_STYLES = ["-", "-", "--", "--"]


def choose_regions(ny: int, yL: Optional[int], yR: Optional[int]) -> Tuple[int, int, bool]:
    """
    Decide left/right integration regions for R/T based on barrier indices.

    Parameters
    ----------
    ny : int
        Grid size per component.
    yL, yR : Optional[int]
        Barrier left (inclusive) and right (exclusive) indices from meta.

    Returns
    -------
    left_end, right_start, used_meta : (int, int, bool)
        `left_end`  : slice end for reflection integration (exclusive).
        `right_start`: slice start for transmission integration (inclusive).
        `used_meta` : True if yL/yR came from meta, False if we used a fallback.
    """
    if yL is not None and yR is not None:
        return int(yL), int(yR), True

    # Fallback: centered barrier of ~5% domain width
    width = max(2, int(0.05 * ny))
    mid = ny // 2
    left = max(0, mid - width // 2)
    right = min(ny, mid + width // 2)
    return left, right, False


def compute_TR_per_component(psi: np.ndarray, left_end: int, right_start: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute T(E) and R(E) per component from `psi(y,t,E)`.

    Parameters
    ----------
    psi : np.ndarray
        Complex array of shape (4*ny, nt, nE). Components stacked as [hh↑, lh↑, lh↓, hh↓].
    left_end : int
        Left boundary index (exclusive) for reflection integration.
    right_start : int
        Right boundary index (inclusive) for transmission integration.

    Returns
    -------
    T, R, Ttot, Rtot : (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        T, R have shape (4, nE).
        Ttot, Rtot have shape (nE,).
    """
    if psi.ndim != 3 or (psi.shape[0] % 4) != 0:
        raise ValueError("psi must have shape (4*ny, nt, nE).")

    four_ny, nt, nE = psi.shape
    ny = four_ny // 4

    T = np.zeros((4, nE), dtype=float)
    R = np.zeros((4, nE), dtype=float)

    for b in range(4):
        psi_b_t0 = psi[b * ny:(b + 1) * ny, 0, :]   # (ny, nE) at t=0
        psi_b_tf = psi[b * ny:(b + 1) * ny, -1, :]  # (ny, nE) at t_final

        norm0 = (np.abs(psi_b_t0) ** 2).sum(axis=0) + 1e-30
        dens_f = np.abs(psi_b_tf) ** 2

        R[b] = dens_f[:left_end, :].sum(axis=0) / norm0
        T[b] = dens_f[right_start:, :].sum(axis=0) / norm0

    Ttot = T.sum(axis=0)
    Rtot = R.sum(axis=0)
    return T, R, Ttot, Rtot


def load_TR_if_present(base_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Try to load precomputed TR arrays from disk.

    Returns
    -------
    (T, R, Ttot, Rtot) or None
        T, R with shape (4, nE). Totals may be None if not present.
    """
    T_p = base_dir / "T.npy"
    R_p = base_dir / "R.npy"
    if not (T_p.exists() and R_p.exists()):
        return None
    T = np.load(T_p)
    R = np.load(R_p)

    Ttot = np.load(base_dir / "Ttot.npy") if (base_dir / "Ttot.npy").exists() else None
    Rtot = np.load(base_dir / "Rtot.npy") if (base_dir / "Rtot.npy").exists() else None
    return T, R, Ttot, Rtot


def save_TR(base_dir: Path, T: np.ndarray, R: np.ndarray, Ttot: np.ndarray, Rtot: np.ndarray) -> None:
    """Persist TR arrays to the result folder."""
    np.save(base_dir / "T.npy", T)
    np.save(base_dir / "R.npy", R)
    np.save(base_dir / "Ttot.npy", Ttot)
    np.save(base_dir / "Rtot.npy", Rtot)


def energy_axis(base_dir: Path, nE: int) -> Tuple[np.ndarray, str]:
    """
    Build the x-axis for plotting. Prefer E_eV.npy if present and consistent.

    Returns
    -------
    x, xlabel : (np.ndarray, str)
    """
    efile = base_dir / "E_eV.npy"
    if efile.exists():
        x = np.load(efile)
        if x.shape[0] == nE:
            return x, "Energy (eV)"
        print("[WARN] E_eV.npy length does not match nE; falling back to energy index.")
    return np.arange(nE), "Energy index"


def compute_or_load_TR(sim: SimData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load T/R if present; otherwise compute from psi and save results.

    Returns
    -------
    T, R, Ttot, Rtot, x_axis, source : (arrays..., str)
        `source` is "loaded" or "computed".
    """
    maybe = load_TR_if_present(sim.base_dir)
    if maybe is not None:
        T, R, Ttot, Rtot = maybe
        nE = T.shape[1]
        x, _ = energy_axis(sim.base_dir, nE)
        if Ttot is None or Rtot is None:
            # compute totals if missing
            Ttot = T.sum(axis=0)
            Rtot = R.sum(axis=0)
        return T, R, Ttot, Rtot, x, "loaded"

    # Compute from psi
    left_end, right_start, used_meta = choose_regions(sim.ny, sim.yL, sim.yR)
    if not used_meta:
        print("[WARN] meta.json lacks yL/yR; using a centered barrier (≈5% width) as fallback.")

    T, R, Ttot, Rtot = compute_TR_per_component(sim.psi, left_end, right_start)
    save_TR(sim.base_dir, T, R, Ttot, Rtot)
    x, _ = energy_axis(sim.base_dir, sim.nE)
    return T, R, Ttot, Rtot, x, "computed"

