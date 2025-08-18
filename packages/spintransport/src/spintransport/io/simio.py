# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.io.simio
# Brief : I/O helpers to read and validate simulation folders (psi.npy, E_eV.npy, meta.json, ...).
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
I/O utilities: read_sim(), file presence checks, and structured accessors for results.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any

import json
import numpy as np


@dataclass(frozen=True)
class SimData:
    """
    Container with loaded simulation results.

    Attributes
    ----------
    base_dir : Path
        Folder containing exported results.
    psi : np.ndarray
        Array with shape (4*ny, nt, nE). May be a memmap (read-only).
    ny : int
        Grid size per component (ny).
    nt : int
        Number of time steps.
    nE : int
        Number of energies.
    meta : Dict[str, Any]
        Parsed meta.json content.
    dy_A : float
        Spatial step in Angstrom (from meta).
    dt_fs : Optional[float]
        Time step in femtoseconds (from meta); may be None if not present.
    yL : Optional[int]
        Barrier left index (inclusive), or None.
    yR : Optional[int]
        Barrier right index (exclusive), or None.
    energies_eV : Optional[np.ndarray]
        Energy axis in eV loaded from E_eV.npy if present, else None.
    """
    base_dir: Path
    psi: np.ndarray
    ny: int
    nt: int
    nE: int
    meta: Dict[str, Any]
    dy_A: float
    dt_fs: Optional[float]
    yL: Optional[int]
    yR: Optional[int]
    energies_eV: Optional[np.ndarray]


def _require_files(base_dir: Path, required: Iterable[str]) -> None:
    """Raise FileNotFoundError listing all missing required files."""
    missing = [name for name in required if not (base_dir / name).exists()]
    if missing:
        miss_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Required files not found in '{base_dir}': {miss_list}"
        )


def _load_meta(base_dir: Path) -> Dict[str, Any]:
    """Load and sanitize meta.json."""
    meta_path = base_dir / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # Normalize keys we care about; keep others as-is
    dy_A = float(meta.get("dy_A") or meta.get("dy_Å") or 1.0)
    dt_fs_raw = meta.get("dt_fs")
    try:
        dt_fs = None if dt_fs_raw is None else float(dt_fs_raw)
    except Exception:
        dt_fs = None

    meta["dy_A"] = dy_A
    meta["dt_fs"] = dt_fs
    return meta


def extract_components_view(psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return views of the 4 components stacked along the first axis in `psi`.

    Parameters
    ----------
    psi : np.ndarray
        Array with shape (4*ny, nt, nE).

    Returns
    -------
    (hh_up, lh_up, lh_dn, hh_dn) : tuple of np.ndarray
        Each view has shape (ny, nt, nE). No copy is made.
    """
    if psi.ndim != 3 or (psi.shape[0] % 4) != 0:
        raise ValueError("psi must have shape (4*ny, nt, nE).")
    ny = psi.shape[0] // 4
    hh_up = psi[0 * ny:1 * ny, :, :]
    lh_up = psi[1 * ny:2 * ny, :, :]
    lh_dn = psi[2 * ny:3 * ny, :, :]
    hh_dn = psi[3 * ny:4 * ny, :, :]
    return hh_up, lh_up, lh_dn, hh_dn


def read_sim(base_dir: str | Path, mmap: bool = True) -> SimData:
    """
    Read a simulation folder exported by `spintransport-sim` and validate files.

    Required files for time-evolution visualization:
    - `psi.npy`
    - `meta.json`

    Optional:
    - `E_eV.npy` (energy axis)

    Parameters
    ----------
    base_dir : str | Path
        Folder with exported results.
    mmap : bool
        If True, open `psi.npy` with numpy memmap (read-only). Useful for large arrays.

    Returns
    -------
    SimData
        Structured access to results and metadata.

    Raises
    ------
    FileNotFoundError
        If any required file is missing (prints which one).
    ValueError
        If array shapes are inconsistent.
    """
    base = Path(base_dir).resolve()
    _require_files(base, required=("psi.npy", "meta.json"))

    # Load metadata
    meta = _load_meta(base)
    dy_A: float = float(meta["dy_A"])
    dt_fs: Optional[float] = meta["dt_fs"]
    yL = meta.get("yL")
    yR = meta.get("yR")

    # Load arrays
    psi_path = base / "psi.npy"
    psi = np.load(psi_path, mmap_mode="r" if mmap else None)  # (4*ny, nt, nE)
    if psi.ndim != 3 or (psi.shape[0] % 4) != 0:
        raise ValueError(f"psi.npy must have shape (4*ny, nt, nE); got {psi.shape}.")

    ny = psi.shape[0] // 4
    nt = psi.shape[1]
    nE = psi.shape[2]

    energies = None
    efile = base / "E_eV.npy"
    if efile.exists():
        energies = np.load(efile)

    return SimData(
        base_dir=base,
        psi=psi,
        ny=ny,
        nt=nt,
        nE=nE,
        meta=meta,
        dy_A=dy_A,
        dt_fs=dt_fs,
        yL=None if yL is None else int(yL),
        yR=None if yR is None else int(yR),
        energies_eV=energies,
    )
