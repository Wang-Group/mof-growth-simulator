#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a 10×10×10 superlattice from a P1 UiO-66 CIF, truncate to {111} octahedron,
preserving connectivity by inferring bonds in the unit cell under PBC and
replicating them with image offsets. Output: bonded MOL2 in Å.
"""

import re
import math
import argparse
from typing import List, Tuple, Dict
import numpy as np

# ---------------------------- CIF parsing (P1) ----------------------------

def _strip_paren_uncert(x: str) -> str:
    return re.sub(r"\(.*\)", "", x)

def read_cif_p1(path: str) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
    """Parse a P1 CIF's cell and atom loop (type_symbol + fract x/y/z)."""
    with open(path, "r") as f:
        lines = f.readlines()

    params = {}
    for key in ["_cell_length_a","_cell_length_b","_cell_length_c",
                "_cell_angle_alpha","_cell_angle_beta","_cell_angle_gamma"]:
        for ln in lines:
            if ln.strip().startswith(key):
                parts = ln.strip().split()
                if len(parts) >= 2:
                    params[key] = float(_strip_paren_uncert(parts[1]))
                break

    atoms_frac = []
    atoms_type = []
    i = 0
    while i < len(lines):
        if lines[i].strip().lower().startswith("loop_"):
            i += 1
            headers = []
            while i < len(lines) and lines[i].strip().startswith("_"):
                headers.append(lines[i].strip())
                i += 1
            if any(h.startswith("_atom_site_") for h in headers):
                def idx(opts):
                    for o in opts:
                        if o in headers: return headers.index(o)
                    return None
                idx_type  = idx(["_atom_site_type_symbol","_atom_site_type"])
                idx_fx    = idx(["_atom_site_fract_x"])
                idx_fy    = idx(["_atom_site_fract_y"])
                idx_fz    = idx(["_atom_site_fract_z"])
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("loop_","_")):
                    row = re.split(r"\s+", lines[i].strip())
                    if idx_fx is not None and len(row) > max(idx_fx, idx_fy, idx_fz):
                        t  = row[idx_type] if idx_type is not None else "Xx"
                        fx = float(_strip_paren_uncert(row[idx_fx]))
                        fy = float(_strip_paren_uncert(row[idx_fy]))
                        fz = float(_strip_paren_uncert(row[idx_fz]))
                        atoms_frac.append((fx, fy, fz))
                        atoms_type.append(t)
                    i += 1
                break
        else:
            i += 1

    return params, np.array(atoms_frac, float), atoms_type

# ---------------------------- Lattice helpers ----------------------------

def lattice_vectors(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = params["_cell_length_a"]
    b = params["_cell_length_b"]
    c = params["_cell_length_c"]
    alpha = math.radians(params.get("_cell_angle_alpha", 90.0))
    beta  = math.radians(params.get("_cell_angle_beta",  90.0))
    gamma = math.radians(params.get("_cell_angle_gamma", 90.0))

    va = np.array([a, 0.0, 0.0], float)
    vb = np.array([b*math.cos(gamma), b*math.sin(gamma), 0.0], float)
    cx = c*math.cos(beta)
    cy = c*(math.cos(alpha) - math.cos(beta)*math.cos(gamma)) / (math.sin(gamma) if abs(math.sin(gamma))>1e-12 else 1.0)
    cz = math.sqrt(max(0.0, c*c - cx*cx - cy*cy))
    vc = np.array([cx, cy, cz], float)
    A = np.column_stack([va, vb, vc])  # 3×3
    return va, vb, vc, A

def elem_from_type(t: str) -> str:
    m = re.match(r"([A-Za-z]+)", t)
    e = m.group(1).capitalize() if m else t.capitalize()
    if e.lower() == "zr": e = "Zr"
    if e.lower() == "cl": e = "Cl"
    return e

# ---------------------------- Bond inference (PBC) ----------------------------

COV_RADII = {"H":0.31, "C":0.76, "N":0.71, "O":0.66, "F":0.57, "Cl":1.02, "Zr":1.48}
def rad(e): return COV_RADII.get(e, 0.80)

def infer_pbc_bonds(frac_uc: np.ndarray, types_uc: List[str], va, vb, vc, scale=1.20):
    """
    Infer bonds in the *unit cell* under PBC. Return list of (i, j, (tx,ty,tz)),
    meaning atom i in the reference cell bonds to atom j in the image cell
    shifted by (tx,ty,tz) in lattice units. i<j by convention.
    """
    cart_uc = (np.column_stack([va, vb, vc]) @ frac_uc.T).T  # N×3
    N = len(frac_uc)
    images = [(ix,iy,iz) for ix in (-1,0,1) for iy in (-1,0,1) for iz in (-1,0,1)]
    bonds = []
    for i in range(N):
        ri = cart_uc[i]; ei = elem_from_type(types_uc[i]); ri_cov = rad(ei)
        for j in range(i+1, N):
            ej = elem_from_type(types_uc[j]); rj_cov = rad(ej)
            cutoff2 = (scale*(ri_cov + rj_cov))**2
            best_d2 = None; best_t = None
            for (tx,ty,tz) in images:
                rj_img = cart_uc[j] + tx*va + ty*vb + tz*vc
                d2 = float(np.dot(rj_img - ri, rj_img - ri))
                if (best_d2 is None) or (d2 < best_d2):
                    best_d2 = d2; best_t = (tx,ty,tz)
            if best_d2 is not None and best_d2 <= cutoff2:
                bonds.append((i, j, best_t))
    return bonds

# ---------------------------- Replicate & octa mask ----------------------------

def build_super_octa_bonded(frac_uc: np.ndarray,
                            types_uc: List[str],
                            bonds_uc: List[Tuple[int,int,Tuple[int,int,int]]],
                            va, vb, vc, n=10):
    """
    Replicate the unit cell n×n×n, place atoms, apply {111} octahedral mask at atom level,
    and keep only bonds whose endpoints remain. Return (coords, elems, bonds).
    """
    center = (n - 1)/2.0
    R = n/2.0
    tol = 1e-8

    # Place atoms & index map
    index_map = {}  # (ci,cj,ck, ai) -> global id (1-based)
    coords = []
    elems = []
    gid = 0
    for ci in range(n):
        for cj in range(n):
            for ck in range(n):
                T = ci*va + cj*vb + ck*vc
                for ai, f in enumerate(frac_uc):
                    ux, uy, uz = ci + f[0], cj + f[1], ck + f[2]
                    if abs(ux-center) + abs(uy-center) + abs(uz-center) <= R + tol:
                        gid += 1
                        index_map[(ci,cj,ck,ai)] = gid
                        coords.append(T + f[0]*va + f[1]*vb + f[2]*vc)
                        elems.append(elem_from_type(types_uc[ai]))
    coords = np.array(coords, float)
    Natoms = len(coords)

    # Lift bonds to supercell
    bonds = []  # (ga, gb), 1-based
    for ci in range(n):
        for cj in range(n):
            for ck in range(n):
                for (i, j, (tx,ty,tz)) in bonds_uc:
                    di, dj, dk = ci + tx, cj + ty, ck + tz
                    if (0 <= di < n) and (0 <= dj < n) and (0 <= dk < n):
                        ka = (ci, cj, ck, i)
                        kb = (di, dj, dk, j)
                        ga = index_map.get(ka)
                        gb = index_map.get(kb)
                        if (ga is not None) and (gb is not None):
                            if ga < gb: bonds.append((ga, gb))
                            elif gb < ga: bonds.append((gb, ga))
    # Deduplicate
    bonds = sorted(set(bonds))
    return coords, elems, bonds

# ---------------------------- MOL2 writer ----------------------------

def write_mol2(path: str, coords: np.ndarray, elems: List[str], bonds: List[Tuple[int,int]]):
    with open(path, "w") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write("UiO-66_10x10x10_octahedron_bonded_from_CIF\n")
        f.write(f"{len(coords)} {len(bonds)} 0 0 0\n")
        f.write("SMALL\n")
        f.write("USER_CHARGES\n")
        f.write("@<TRIPOS>ATOM\n")
        for i, (xyz, elem) in enumerate(zip(coords, elems), start=1):
            x, y, z = xyz
            f.write(f"{i} {elem}{i} {x:.4f} {y:.4f} {z:.4f} {elem} 1 U66 0.0000\n")
        f.write("@<TRIPOS>BOND\n")
        for k, (a, b) in enumerate(bonds, start=1):
            f.write(f"{k} {a} {b} 1\n")

# ---------------------------- CLI ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Build 10×10×10 {111} octahedron from P1 CIF preserving connectivity.")
    ap.add_argument("--cif", required=True, help="Input P1 CIF file")
    ap.add_argument("--n", type=int, default=10, help="Replication per axis (default 10)")
    ap.add_argument("--scale", type=float, default=1.20, help="Bond cutoff factor ×(ri+rj), default 1.20")
    ap.add_argument("--out", default="UiO-66_10x10x10_octahedron_bonded_from_CIF.mol2", help="Output MOL2")
    args = ap.parse_args()

    # 1) Read CIF (P1)
    params, frac_uc, types_uc_raw = read_cif_p1(args.cif)
    va, vb, vc, A = lattice_vectors(params)
    types_uc = [elem_from_type(t) for t in types_uc_raw]

    # 2) Infer PBC bonds in unit cell
    bonds_uc = infer_pbc_bonds(frac_uc, types_uc, va, vb, vc, scale=args.scale)

    # 3) Replicate, mask to {111} octahedron, and lift bonds
    coords, elems, bonds = build_super_octa_bonded(frac_uc, types_uc, bonds_uc, va, vb, vc, n=args.n)

    # 4) Write MOL2
    write_mol2(args.out, coords, elems, bonds)

    print(f"Wrote: {args.out}")
    print(f"Atoms: {len(coords)} | Bonds: {len(bonds)}")

if __name__ == "__main__":
    main()