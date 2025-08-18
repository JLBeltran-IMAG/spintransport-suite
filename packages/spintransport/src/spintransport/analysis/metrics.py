# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.analysis.metrics
# Brief : Device-level figures of merit from T/R and spin precession (e.g., conductance).
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
Metrics layer: roll-ups, Landauer conductance, on/off ratios, and precession summaries.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

from spintransport.io.simio import read_sim, SimData
from spintransport.analysis.tr_curves import compute_or_load_TR

# ---------- Physical constants ----------
e_C = 1.602_176_634e-19           # Coulomb
h_Js = 6.626_070_15e-34           # Joule·s
kB_eV_per_K = 8.617_333_262e-5    # eV/K
G0_e2_over_h_S = (e_C**2) / h_Js  # ≈ 3.874e-5 S (= 38.74 μS)
G0_2e2_over_h_S = 2.0 * G0_e2_over_h_S

# ---------- Convenience containers ----------
@dataclass(frozen=True)
class TRData:
    E: np.ndarray              # energy axis (eV) or indices
    T: np.ndarray              # shape (4, nE)
    R: np.ndarray              # shape (4, nE)
    Ttot: np.ndarray           # (nE,)
    Rtot: np.ndarray           # (nE,)
    x_is_eV: bool


@dataclass(frozen=True)
class PrecessionSeries:
    t: np.ndarray              # time axis, fs or index
    Sx: np.ndarray             # (nt,)
    Sy: np.ndarray             # (nt,)
    Sz: np.ndarray             # (nt,)
    arrival_index: int         # first arrival time index
    arrival_time: float        # fs or index (same units as t)
    label_energy: str          # description of energy selection
    pair: str                  # 'hh' or 'lh'
    region: str                # region string
    mode: str                  # 'abs' or 'cond'

# ---------- T/R helpers ----------
def load_TR(sim: SimData) -> TRData:
    """Load or compute T/R and build an energy axis."""
    T, R, Ttot, Rtot, x, src = compute_or_load_TR(sim)
    x_is_eV = (sim.base_dir / "E_eV.npy").exists() and (x.shape[0] == T.shape[1])
    return TRData(E=x, T=T, R=R, Ttot=Ttot, Rtot=Rtot, x_is_eV=x_is_eV)

def tr_basic_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Basic stats over an array y(x):
    - mean, std, min, max, argmax (value & x location)
    """
    out: Dict[str, Any] = {}
    out["mean"] = float(np.mean(y))
    out["std"]  = float(np.std(y))
    out["min"]  = float(np.min(y))
    out["max"]  = float(np.max(y))
    imax = int(np.argmax(y))
    out["argmax_index"] = imax
    out["x_at_max"] = float(x[imax]) if x.size == y.size else float(imax)
    return out

def on_off_ratio(y: np.ndarray, eps: float = 1e-12) -> float:
    """Return max(y)/min(y+eps)."""
    return float((np.max(y) + eps) / (np.min(y) + eps))

# ---------- Conductance (Landauer) ----------
def fermi_derivative_kernel(E_eV: np.ndarray, Ef_eV: float, T_K: float) -> np.ndarray:
    """
    Return -df/dE(E) at temperature T (K) in eV^-1 units.
    -df/dE = 1/(4 k_B T) * sech^2((E - Ef)/(2 k_B T))
    """
    if T_K <= 0:
        # Dirac delta in the discrete limit: we'll sample by nearest E to Ef
        ker = np.zeros_like(E_eV, dtype=float)
        idx = int(np.argmin(np.abs(E_eV - Ef_eV)))
        ker[idx] = 1.0  # will be used as discrete pick
        return ker
    beta = 1.0 / (kB_eV_per_K * T_K)
    x = 0.5 * (E_eV - Ef_eV) * beta
    sech2 = 1.0 / (np.cosh(x) ** 2)
    return 0.5 * beta * sech2  # equals 1/(4 kBT) * sech^2(...)

def landauer_conductance(Ttot: np.ndarray,
                         E_eV: Optional[np.ndarray],
                         Ef_eV: Optional[float] = None,
                         T_K: float = 300.0,
                         spin_degeneracy: bool = False) -> Dict[str, Any]:
    """
    Compute Landauer conductance:
      G(T) = (e^2/h) * ∫ dE T(E) (-df/dE)
    If E_eV is None, fallback to a discrete pick at the mid index:
      G ≈ (e^2/h) * T(E_mid)

    Returns dict with:
      - G_e2_over_h : dimensionless
      - G_S         : Siemens
      - G_uS        : microSiemens
      - Ef_eV_used, T_K_used
    """
    G0 = G0_2e2_over_h_S if spin_degeneracy else G0_e2_over_h_S

    if E_eV is None:
        # No energy axis → choose middle index
        Emid_idx = int(Ttot.size // 2)
        G_e2h = float(Ttot[Emid_idx])  # in units of (e^2/h)
        return {
            "G_e2_over_h": G_e2h,
            "G_S": G_e2h * G0,
            "G_uS": G_e2h * G0 * 1e6,
            "Ef_eV_used": None,
            "T_K_used": T_K,
            "note": "No E_eV grid; used mid-index transmission."
        }

    # With energy grid: integrate with -df/dE
    E = np.asarray(E_eV, dtype=float)
    if Ef_eV is None:
        Ef_eV = float(E[len(E)//2])
    kernel = fermi_derivative_kernel(E, Ef_eV, T_K)  # eV^-1
    # Normalize kernel for discrete trapz to ~1
    kernel = kernel / np.trapz(kernel, E)

    G_e2h = float(np.trapz(Ttot * kernel, E))
    return {
        "G_e2_over_h": G_e2h,
        "G_S": G_e2h * G0,
        "G_uS": G_e2h * G0 * 1e6,
        "Ef_eV_used": Ef_eV,
        "T_K_used": T_K,
    }

# ---------- Spin precession (no plotting) ----------
def _choose_region(region: str, yL: int, yR: int, ny: int) -> slice:
    region = region.lower()
    if region == "left":
        return slice(0, max(0, int(yL)))
    if region == "barrier":
        return slice(max(0, int(yL)), min(ny, int(yR)))
    if region == "right":
        return slice(min(ny, int(yR)), ny)
    if region == "full":
        return slice(0, ny)
    raise ValueError("region must be: left | barrier | right | full")

def _extract_pair(psi: np.ndarray, ny: int, pair: str):
    pair = pair.lower()
    if pair == "hh":
        return psi[0*ny:1*ny, :, :], psi[3*ny:4*ny, :, :]
    if pair == "lh":
        return psi[1*ny:2*ny, :, :], psi[2*ny:3*ny, :, :]
    raise ValueError("pair must be 'hh' or 'lh'")

def _series_precession(sim: SimData, pair: str, region: str, mode: str,
                       energy: str = "mid") -> PrecessionSeries:
    """
    Compute time series Sx,Sy,Sz with a minimal energy selector:
      energy = "mid" | "avg" | "<i>" | "<i0>:<i1>"
    """
    if sim.yL is None or sim.yR is None:
        raise RuntimeError("meta.json must include yL and yR.")

    # Parse energy selector
    nE = sim.nE
    sel_label = ""
    if energy is None or energy.lower() == "mid":
        idx = np.array([nE // 2], dtype=int); sel_label = f"E_idx={idx[0]}"
    elif energy.lower() in ("avg", "all"):
        idx = np.arange(nE, dtype=int); sel_label = "E: avg(all)"
    elif ":" in energy:
        s0, s1 = energy.split(":", 1)
        i0 = max(0, min(int(s0), nE - 1)); i1 = max(0, min(int(s1), nE - 1))
        if i0 > i1: i0, i1 = i1, i0
        idx = np.arange(i0, i1 + 1, dtype=int); sel_label = f"E_idx in [{i0},{i1}]"
    else:
        i = max(0, min(int(energy), nE - 1))
        idx = np.array([i], dtype=int); sel_label = f"E_idx={i}"

    reg = _choose_region(region, sim.yL, sim.yR, sim.ny)

    # Build numerators and regional density
    u, d = _extract_pair(sim.psi, sim.ny, pair)  # (ny, nt, nE)
    u = u[reg, :, :]; d = d[reg, :, :]
    dens_reg = (np.abs(u)**2 + np.abs(d)**2).sum(axis=0) + 1e-30  # (nt, nE)
    sx_num = (np.conj(u)*d + np.conj(d)*u).sum(axis=0)
    sy_num = (-1j*np.conj(u)*d + 1j*np.conj(d)*u).sum(axis=0)
    sz_num = (np.abs(u)**2 - np.abs(d)**2).sum(axis=0)

    # Norm at t0 for the pair
    N0_pair = (np.abs(_extract_pair(sim.psi, sim.ny, pair)[0][:, 0, :])**2 +
               np.abs(_extract_pair(sim.psi, sim.ny, pair)[1][:, 0, :])**2).sum(axis=0) + 1e-30

    # Arrival detection
    occ_frac_tE = dens_reg[:, idx] / (N0_pair[None, idx] + 1e-30)
    occ_frac = np.mean(occ_frac_tE, axis=1)
    # same hysteresis as earlier
    r = 3; thr = 1e-3
    over = (occ_frac >= thr).astype(int)
    counts = np.convolve(over, np.ones(r, dtype=int), mode="same")
    arr_candidates = np.where(counts >= r)[0]
    t0_idx = int(arr_candidates[0]) if arr_candidates.size else sim.nt - 1

    # Expectations
    if mode.lower() == "abs":
        Sx = np.real(np.mean(sx_num[:, idx] / (N0_pair[None, idx] + 1e-30), axis=1))
        Sy = np.real(np.mean(sy_num[:, idx] / (N0_pair[None, idx] + 1e-30), axis=1))
        Sz = np.real(np.mean(sz_num[:, idx] / (N0_pair[None, idx] + 1e-30), axis=1))
        Sx[:t0_idx] = 0.0; Sy[:t0_idx] = 0.0; Sz[:t0_idx] = 0.0
    elif mode.lower() == "cond":
        Sx = np.real(np.mean(sx_num[:, idx] / (dens_reg[:, idx] + 1e-30), axis=1))
        Sy = np.real(np.mean(sy_num[:, idx] / (dens_reg[:, idx] + 1e-30), axis=1))
        Sz = np.real(np.mean(sz_num[:, idx] / (dens_reg[:, idx] + 1e-30), axis=1))
        Sx[:t0_idx] = np.nan; Sy[:t0_idx] = np.nan; Sz[:t0_idx] = np.nan
    else:
        raise ValueError("mode must be 'abs' or 'cond'")

    # Time axis
    if sim.dt_fs is not None:
        t = np.arange(sim.nt) * float(sim.dt_fs)
        t_arr = float(t[t0_idx])
    else:
        t = np.arange(sim.nt); t_arr = float(t0_idx)

    return PrecessionSeries(
        t=t, Sx=Sx, Sy=Sy, Sz=Sz,
        arrival_index=t0_idx, arrival_time=t_arr,
        label_energy=sel_label, pair=pair, region=region, mode=mode
    )

def precession_metrics(series: PrecessionSeries) -> Dict[str, Any]:
    """Compute a small set of scalar metrics from a precession time-series."""
    # Use last defined point after arrival (ignore NaNs in cond)
    valid = np.isfinite(series.Sx) & np.isfinite(series.Sy) & np.isfinite(series.Sz)
    valid[ : series.arrival_index] = False
    if not np.any(valid):
        # fallback: try final sample (even if NaN)
        phi_final = float("nan"); amp_xy = float("nan")
        S_final = {"Sx": float(series.Sx[-1]), "Sy": float(series.Sy[-1]), "Sz": float(series.Sz[-1])}
    else:
        j = int(np.where(valid)[0][-1])
        phi_final = float(np.arctan2(series.Sy[j], series.Sx[j]))
        amp_xy = float(np.hypot(series.Sx[j], series.Sy[j]))
        S_final = {"Sx": float(series.Sx[j]), "Sy": float(series.Sy[j]), "Sz": float(series.Sz[j])}

    return {
        "arrival_time": series.arrival_time,
        "phi_final_rad": phi_final,
        "inplane_amplitude": amp_xy,
        "S_final": S_final,
        "energy_label": series.label_energy,
        "pair": series.pair,
        "region": series.region,
        "mode": series.mode,
    }

# ---------- Device-level rollup ----------
def device_metrics(sim: SimData,
                   Ef_eV: Optional[float] = None,
                   T_K: float = 300.0,
                   spin_degeneracy: bool = False,
                   precession_pair: str = "hh",
                   precession_region: str = "right",
                   precession_mode: str = "abs",
                   precession_energy: str = "mid") -> Dict[str, Any]:
    """
    Assemble a dict with key metrics for a Datta-Das-like spin FET design:
    - TR statistics (per total T(E), R(E))
    - Landauer conductance at (Ef, T)
    - Spin precession (arrival time, final angle/amplitude)
    """
    tr = load_TR(sim)
    # Roll-up over TOTAL T/R
    Tstats = tr_basic_stats(tr.E, tr.Ttot)
    Rstats = tr_basic_stats(tr.E, tr.Rtot)
    T_onoff = on_off_ratio(tr.Ttot)

    # Conductance
    Eaxis = tr.E if tr.x_is_eV else None
    cond = landauer_conductance(tr.Ttot, Eaxis, Ef_eV=Ef_eV, T_K=T_K, spin_degeneracy=spin_degeneracy)

    # Precession (single region/pair)
    series = _series_precession(sim, pair=precession_pair, region=precession_region,
                                mode=precession_mode, energy=precession_energy)
    pmetrics = precession_metrics(series)

    return {
        "tr_stats": {"Ttot": Tstats, "Rtot": Rstats, "T_on_off_ratio": T_onoff},
        "conductance": cond,
        "precession": pmetrics,
    }
