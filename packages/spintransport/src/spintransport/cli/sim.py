# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.cli.sim
# Brief : CLI to run the Kohn–Luttinger + Rashba time-domain simulator and save outputs.
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
Simulation CLI: builds the Hamiltonian, evolves 4-component wavepackets, and writes psi/meta/energy axis.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import numpy as np

from spintransport.physics import st


# =================
# Unit helpers (au)
# =================
HBAR = st.HBAR_AU
def eV(x): return st.eV(x)
def A(x):  return st.angstrom(x)
def fs(x): return st.femtosecond(x)

def alpha_eVA_to_au(alpha_eVA: float) -> float:
    """Convert α (or β ε_z) from eV·Å to Hartree·Bohr (au)."""
    return alpha_eVA * st.EV_TO_AU * st.ANGSTROM_TO_BOHR


# ===========================
# Grid masks and potentials
# ===========================
def barrier_mask(ny: int, dy_au: float, barrier_thickness_A: float, center_index: int | None = None):
    """
    Build a 1D mask equal to 1 inside the barrier and 0 outside.

    Returns
    -------
    mask : np.ndarray
        Array of shape (ny,) with 0/1 values.
    yL : int
        Left index of the barrier (inclusive).
    yR : int
        Right index of the barrier (exclusive).
    """
    Lb = A(barrier_thickness_A)
    half = int(0.5 * Lb / dy_au)
    mid = ny // 2 if center_index is None else int(center_index)
    mask = np.zeros(ny, dtype=float)
    left = max(0, mid - half)
    right = min(ny, mid + half)
    mask[left:right] = 1.0
    return mask, left, right


def potential_matrix(ny: int, Vb_eV: float, mask1d: np.ndarray) -> np.ndarray:
    """
    Scalar barrier potential replicated on the 4-component basis (block diagonal).
    """
    V_diag = np.diag(eV(Vb_eV) * mask1d.astype(float))
    H = np.zeros((4 * ny, 4 * ny), dtype=complex)
    for b in range(4):
        H[b * ny:(b + 1) * ny, b * ny:(b + 1) * ny] = V_diag
    return H


# =========================
# Differential operators
# =========================
def KY(ny: int, dy_au: float) -> np.ndarray:
    """
    Hermitian k_y operator. Note: st.partial_derivative_operator(ny, dy, 1)
    equals 2*k_y by construction; we multiply by 1/2.
    """
    return 0.5 * st.partial_derivative_operator(ny, dy_au, order=1)


def KY2(ny: int, dy_au: float) -> np.ndarray:
    """
    k_y^2 operator (already positive by construction).
    """
    return st.partial_derivative_operator(ny, dy_au, order=2)


def symmetrize(A: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Hermitian symmetrization ½ (A D + D A), useful when coefficients depend on y.
    """
    return 0.5 * (A @ D + D @ A)


# =========================
# Hamiltonian components
# =========================
def kohn_luttinger_hamiltonian(ny: int, dy_au: float, gamma1: float, gamma2: float, gamma3: float,
                               kz_expect: float = 0.0) -> np.ndarray:
    """
    Reduced 4×4 Kohn–Luttinger Hamiltonian in the basis [hh↑, lh↑, lh↓, hh↓],
    discretized on a 1D grid along y with operators KY, KY2.

    Blocks pattern (schematic):
      H11 ~ (ħ² γ_+)/2 * k_y²,   H22 ~ (ħ² γ_-)/2 * k_y²
      H12 ~ (ħ² √3 γ_3 ⟨kz⟩)/2 * k_y
      H13 ~ (ħ² √3 γ_2)/2 * k_y²
      Remaining entries by symmetry/pattern for 4×4 structure.

    Parameters
    ----------
    ny, dy_au : grid size and spacing (atomic units).
    gamma1, gamma2, gamma3 : Luttinger parameters.
    kz_expect : float
        Expectation of k_z (dimension of inverse length, in the same au system).
        For symmetric ground subband, it is ~0.

    Returns
    -------
    np.ndarray
        (4*ny, 4*ny) complex Hermitian matrix.
    """
    gamma_p = gamma1 + gamma2
    gamma_m = gamma1 - gamma2

    ky = KY(ny, dy_au)
    ky2 = KY2(ny, dy_au)

    H11 = (HBAR ** 2) * 0.5 * gamma_p * ky2
    H22 = (HBAR ** 2) * 0.5 * gamma_m * ky2
    H13 = (HBAR ** 2) * 0.5 * np.sqrt(3.0) * gamma2 * ky2
    H12 = (HBAR ** 2) * 0.5 * np.sqrt(3.0) * gamma3 * kz_expect * ky

    H = np.zeros((4 * ny, 4 * ny), dtype=complex)

    # Diagonal blocks
    H[0 * ny:1 * ny, 0 * ny:1 * ny] = H11
    H[1 * ny:2 * ny, 1 * ny:2 * ny] = H22
    H[2 * ny:3 * ny, 2 * ny:3 * ny] = H22
    H[3 * ny:4 * ny, 3 * ny:4 * ny] = H11

    # Off-diagonal pattern
    H[0 * ny:1 * ny, 1 * ny:2 * ny] = H12
    H[1 * ny:2 * ny, 0 * ny:1 * ny] = H12

    H[0 * ny:1 * ny, 2 * ny:3 * ny] = H13
    H[2 * ny:3 * ny, 0 * ny:1 * ny] = H13

    H[1 * ny:2 * ny, 3 * ny:4 * ny] = H13
    H[3 * ny:4 * ny, 1 * ny:2 * ny] = H13

    H[2 * ny:3 * ny, 3 * ny:4 * ny] = -H12
    H[3 * ny:4 * ny, 2 * ny:3 * ny] = -H12

    return H


def rashba_hamiltonian(ny: int, dy_au: float, beta_eff_eVA: float, mask1d: np.ndarray) -> np.ndarray:
    """
    Rashba-like effective Hamiltonian active only where `mask1d == 1`.

    Implementation detail:
    - Use a y-dependent coefficient β_eff (in au) via diagonal Beta,
      and Hermitian symmetrization with k_y: K = ½ (Beta k_y + k_y Beta).
    """
    ky = KY(ny, dy_au)
    Beta = np.diag(alpha_eVA_to_au(beta_eff_eVA) * mask1d.astype(float))  # Ha·Bohr
    K = symmetrize(Beta, ky)

    i = 1j
    s3 = np.sqrt(3.0) / 2.0
    H = np.zeros((4 * ny, 4 * ny), dtype=complex)

    H[0 * ny:1 * ny, 1 * ny:2 * ny] = -i * s3 * K
    H[1 * ny:2 * ny, 0 * ny:1 * ny] = +i * s3 * K

    H[1 * ny:2 * ny, 2 * ny:3 * ny] = -i * K
    H[2 * ny:3 * ny, 1 * ny:2 * ny] = +i * K

    H[2 * ny:3 * ny, 3 * ny:4 * ny] = -i * s3 * K
    H[3 * ny:4 * ny, 2 * ny:3 * ny] = +i * s3 * K
    return H


# =========================
# Initial conditions
# =========================
def initial_wavepacket_bandmix(
    ny: int,
    y0: float,
    sigma: float,
    energies_au: np.ndarray,
    m_eff_hh: np.ndarray | float,
    m_eff_lh: np.ndarray | float,
    lead_fraction: float = 0.2,
) -> np.ndarray:
    """
    Build a 4-component spinor wave packet at t=0 with energy-resolved columns.

    Components order: [hh↑, lh↑, lh↓, hh↓], each of length ny stacked vertically.
    The injection wavenumbers are taken energy-by-energy and assumed y-independent.

    Parameters
    ----------
    ny : int
        Number of grid points per component.
    y0 : float
        Packet center (grid units, consistent with how you build `y = arange(ny)`).
    sigma : float
        Packet width (grid units).
    energies_au : np.ndarray
        Injection energies in atomic units, shape (nE,).
    m_eff_hh, m_eff_lh : np.ndarray | float
        Effective masses. If arrays (shape (ny,)), a scalar reference mass is built
        by averaging over the left `lead_fraction` of the domain (lead injection).
        If scalars, they are used directly.
    lead_fraction : float
        Fraction of the left domain used to average masses when arrays are provided.

    Returns
    -------
    np.ndarray
        ψ0 with shape (4*ny, nE), complex.
    """
    def _mass_ref(m):
        if np.ndim(m) == 0:
            return float(m)
        # 1D profile → average over left lead
        n_left = max(1, int(ny * lead_fraction))
        return float(np.mean(np.asarray(m)[:n_left]))

    m_hh_ref = _mass_ref(m_eff_hh)
    m_lh_ref = _mass_ref(m_eff_lh)

    # k(E) = sqrt(2 m* E)  → vectors of shape (nE,)
    k_hh = np.sqrt(2.0 * m_hh_ref * energies_au)
    k_lh = np.sqrt(2.0 * m_lh_ref * energies_au)

    y = np.arange(ny)
    hh_up = st.gaussian_packet(y, y0, sigma, k_hh, conj=False)  # (ny, nE)
    lh_up = st.gaussian_packet(y, y0, sigma, k_lh, conj=False)  # (ny, nE)
    lh_dn = st.gaussian_packet(y, y0, sigma, k_lh, conj=True)   # (ny, nE)
    hh_dn = st.gaussian_packet(y, y0, sigma, k_hh, conj=True)   # (ny, nE)

    # Gentle normalization for a localized packet, equally split across components
    norm_total = 2.0 / (np.pi * sigma ** 2) ** 0.25
    norm_spinor = 0.25

    hh_up *= norm_total * norm_spinor
    lh_up *= norm_total * norm_spinor
    lh_dn *= norm_total * norm_spinor
    hh_dn *= norm_total * norm_spinor

    psi0 = np.zeros((4 * ny, energies_au.size), dtype=complex)
    psi0[0 * ny:1 * ny, :] = hh_up
    psi0[1 * ny:2 * ny, :] = lh_up
    psi0[2 * ny:3 * ny, :] = lh_dn
    psi0[3 * ny:4 * ny, :] = hh_dn
    return psi0


# =========================
# Observables
# =========================
def transmission_reflection(psi: np.ndarray, left_end: int, right_start: int, ny: int):
    """
    Compute per-component and total transmission/reflection from densities
    at final time step (last t index) relative to initial normalization.

    Parameters
    ----------
    psi : np.ndarray
        Shape (4*ny, nt, nE)
    left_end : int
        Left index up to which reflection is integrated (exclusive).
    right_start : int
        Right index from which transmission is integrated (inclusive).
    ny : int
        Grid size per component.

    Returns
    -------
    T, R, Ttot, Rtot
        T, R have shape (4, nE). Ttot, Rtot have shape (nE,).
    """
    _, nt, nE = psi.shape
    T = np.zeros((4, nE))
    R = np.zeros((4, nE))

    def comp_block(b):  # last time slice
        return psi[b * ny:(b + 1) * ny, nt - 1, :]

    for b in range(4):
        dens_final = np.abs(comp_block(b)) ** 2
        dens_init = np.abs(psi[b * ny:(b + 1) * ny, 0, :]) ** 2
        norm0 = dens_init.sum(axis=0) + 1e-30
        T[b] = dens_final[right_start:, :].sum(axis=0) / norm0
        R[b] = dens_final[:left_end, :].sum(axis=0) / norm0

    return T, R, T.sum(axis=0), R.sum(axis=0)


# =========================
# CLI and simulation driver
# =========================
@dataclass
class SimConfig:
    L_A: float
    T_fs: float
    dy_A: float
    dt_fs: float
    E_min_eV: float
    E_max_eV: float
    nE: int
    gamma1: float
    gamma2: float
    gamma3: float
    Lb_A: float
    Vb_eV: float
    beta_eff_eVA: float
    outdir: Path
    kz_expect: float = 0.0


def parse_args() -> SimConfig:
    p = argparse.ArgumentParser(description="spintransport simulation (KL + Rashba, CN time evolution)")

    p.add_argument("--L_A", type=float, required=True, help="Total device length (Å).")
    p.add_argument("--T_fs", type=float, required=True, help="Total simulated time (fs).")
    p.add_argument("--dy_A", type=float, required=True, help="Grid spacing (Å).")
    p.add_argument("--dt_fs", type=float, required=True, help="Time step (fs).")

    p.add_argument("--E_min", type=float, required=True, help="Minimum injection energy (eV).")
    p.add_argument("--E_max", type=float, required=True, help="Maximum injection energy (eV).")
    p.add_argument("--nE", type=int, required=True, help="Number of energies for linspace [E_min, E_max].")

    p.add_argument("--gamma1", type=float, required=True, help="Luttinger parameter γ1.")
    p.add_argument("--gamma2", type=float, required=True, help="Luttinger parameter γ2.")
    p.add_argument("--gamma3", type=float, required=True, help="Luttinger parameter γ3.")

    p.add_argument("--Lb_A", type=float, required=True, help="Barrier thickness (Å).")
    p.add_argument("--Vb_eV", type=float, required=True, help="Barrier height (eV).")
    p.add_argument("--beta_eff_eVA", type=float, required=True, help="Effective Rashba β ε_z (eV·Å).")

    p.add_argument("--outdir", type=Path, required=True, help="Output directory path.")
    p.add_argument("--kz_expect", type=float, default=0.0, help="⟨kz⟩ (au, typically 0.0).")

    a = p.parse_args()

    if a.E_max < a.E_min:
        p.error("E_max must be >= E_min")
    if a.nE < 1:
        p.error("nE must be >= 1")

    return SimConfig(
        L_A=a.L_A,
        T_fs=a.T_fs,
        dy_A=a.dy_A,
        dt_fs=a.dt_fs,
        E_min_eV=a.E_min,
        E_max_eV=a.E_max,
        nE=int(a.nE),
        gamma1=a.gamma1,
        gamma2=a.gamma2,
        gamma3=a.gamma3,
        Lb_A=a.Lb_A,
        Vb_eV=a.Vb_eV,
        beta_eff_eVA=a.beta_eff_eVA,
        outdir=a.outdir,
        kz_expect=a.kz_expect,
    )


def run_simulation(cfg: SimConfig) -> None:
    """
    Full simulation pipeline:
    1) Build grid/time / energies.
    2) Build Hamiltonian H = H_KL + H_R + V(y).
    3) CN precompute (L, R, LU(L)).
    4) Build ψ(t=0; E).
    5) Time-evolve all energies at once (multi-RHS solve per step).
    6) Compute T(E), R(E); export arrays and meta.
    """
    # --- Grid & time ---
    dy = A(cfg.dy_A)
    dt = fs(cfg.dt_fs)
    ny = int(A(cfg.L_A) / dy)
    nt = int(fs(cfg.T_fs) / dt)

    # Center and width of the packet in *grid indices*
    y0 = ny / 4.0
    sigma = ny * 0.05

    # Energies (eV & au)
    E_eV = np.linspace(cfg.E_min_eV, cfg.E_max_eV, cfg.nE)
    E_au = eV(E_eV)

    # --- Materials (effective masses via γ1, γ2) ---
    # m* in the usual KL reduced form along y for hh/lh
    m_eff_hh = np.full(ny, 1.0 / (cfg.gamma1 + cfg.gamma2), dtype=float)
    m_eff_lh = np.full(ny, 1.0 / (cfg.gamma1 - cfg.gamma2), dtype=float)

    # --- Barrier mask and potential ---
    mask_b, yL, yR = barrier_mask(ny, dy, cfg.Lb_A)
    H_V = potential_matrix(ny, cfg.Vb_eV, mask_b)

    # --- Hamiltonian pieces ---
    H_KL = kohn_luttinger_hamiltonian(ny, dy, cfg.gamma1, cfg.gamma2, cfg.gamma3, kz_expect=cfg.kz_expect)
    H_R = rashba_hamiltonian(ny, dy, cfg.beta_eff_eVA, mask_b)

    H = H_KL + H_R + H_V

    # --- Crank–Nicolson operators ---
    _, R, LU = st.build_crank_nicolson_system(H, dt)

    # --- Initial condition for all energies ---
    psi0 = initial_wavepacket_bandmix(ny, y0, sigma, E_au, m_eff_hh, m_eff_lh)  # (4ny, nE)

    # --- Time evolution (full history) ---
    psi = st.evolve_crank_nicolson(LU, R, psi0, n_time_steps=nt, show_progress=True)

    # --- Observables ---
    T, Rcoef, Ttot, Rtot = transmission_reflection(psi, yL, yR, ny)

    # --- Export ---
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    meta = {
        "L_A": float(cfg.L_A),
        "T_fs": float(cfg.T_fs),
        "dy_A": float(cfg.dy_A),
        "dt_fs": float(cfg.dt_fs),
        "E_min_eV": float(cfg.E_min_eV),
        "E_max_eV": float(cfg.E_max_eV),
        "nE": int(cfg.nE),
        "gamma1": float(cfg.gamma1),
        "gamma2": float(cfg.gamma2),
        "gamma3": float(cfg.gamma3),
        "Lb_A": float(cfg.Lb_A),
        "Vb_eV": float(cfg.Vb_eV),
        "beta_eff_eVA": float(cfg.beta_eff_eVA),
        "kz_expect": float(cfg.kz_expect),
        "yL": int(yL),
        "yR": int(yR),
        "notes": "Crank-Nicolson with LU(left), multi-RHS solve; basis [hh↑, lh↑, lh↓, hh↓].",
    }
    (cfg.outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    np.save(cfg.outdir / "psi.npy", psi)          # (4*ny, nt, nE)
    np.save(cfg.outdir / "T.npy", T)              # (4, nE)
    np.save(cfg.outdir / "R.npy", Rcoef)          # (4, nE)
    np.save(cfg.outdir / "Ttot.npy", Ttot)        # (nE,)
    np.save(cfg.outdir / "Rtot.npy", Rtot)        # (nE,)
    np.save(cfg.outdir / "E_eV.npy", E_eV)

    print(f"\n[OK] Results written to: {cfg.outdir.resolve()}")


def main():
    cfg = parse_args()
    run_simulation(cfg)


if __name__ == "__main__":
    main()
