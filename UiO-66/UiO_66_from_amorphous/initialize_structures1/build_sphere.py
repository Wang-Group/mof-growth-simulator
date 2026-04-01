#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a 15×15×15 superlattice from a P1 UiO-66 CIF,
keep only atoms within ±6 lattice units of the center (a sphere),
and preserve all periodic bonds inside that sphere.
Outputs: bonded MOL2 in Å.
"""

import re, math, argparse
import numpy as np
from typing import List, Tuple, Dict

# ---------- CIF parsing ----------
def _strip_paren_uncert(x): return re.sub(r"\(.*\)", "", x)
def read_cif_p1(path):
    with open(path) as f: lines = f.readlines()
    params = {}
    for key in ["_cell_length_a","_cell_length_b","_cell_length_c",
                "_cell_angle_alpha","_cell_angle_beta","_cell_angle_gamma"]:
        for ln in lines:
            if ln.strip().startswith(key):
                params[key] = float(_strip_paren_uncert(ln.split()[1]))
                break
    atoms_frac, atoms_type = [], []
    i = 0
    while i < len(lines):
        if lines[i].strip().lower().startswith("loop_"):
            i += 1; headers=[]
            while i < len(lines) and lines[i].strip().startswith("_"):
                headers.append(lines[i].strip()); i+=1
            if any(h.startswith("_atom_site_") for h in headers):
                def idx(opts):
                    for o in opts:
                        if o in headers: return headers.index(o)
                    return None
                ix_t = idx(["_atom_site_type_symbol","_atom_site_type"])
                ix_x = idx(["_atom_site_fract_x"]); ix_y = idx(["_atom_site_fract_y"]); ix_z = idx(["_atom_site_fract_z"])
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("loop_","_")):
                    row = re.split(r"\s+", lines[i].strip())
                    t = row[ix_t] if ix_t is not None else "Xx"
                    fx,fy,fz = [float(_strip_paren_uncert(row[ix])) for ix in (ix_x,ix_y,ix_z)]
                    atoms_frac.append((fx,fy,fz)); atoms_type.append(t)
                    i+=1
                break
        else: i+=1
    return params, np.array(atoms_frac,float), atoms_type

# ---------- lattice ----------
def lattice_vectors(p):
    a,b,c = p["_cell_length_a"],p["_cell_length_b"],p["_cell_length_c"]
    α,β,γ = map(math.radians,[p.get("_cell_angle_alpha",90),p.get("_cell_angle_beta",90),p.get("_cell_angle_gamma",90)])
    va = np.array([a,0,0]); vb = np.array([b*math.cos(γ), b*math.sin(γ),0])
    cx = c*math.cos(β); cy = c*(math.cos(α)-math.cos(β)*math.cos(γ))/math.sin(γ)
    cz = math.sqrt(max(0.0,c*c-cx*cx-cy*cy))
    vc = np.array([cx,cy,cz])
    return va,vb,vc,np.column_stack([va,vb,vc])

def elem(t):
    m=re.match(r"([A-Za-z]+)",t)
    e=(m.group(1) if m else t).capitalize()
    return {"zr":"Zr","cl":"Cl"}.get(e.lower(),e)

# ---------- PBC bonds ----------
COV={"H":0.31,"C":0.76,"N":0.71,"O":0.66,"F":0.57,"Cl":1.02,"Zr":1.48}
def rad(e): return COV.get(e,0.8)
def infer_pbc_bonds(frac,types,va,vb,vc,scale=1.2):
    A=np.column_stack([va,vb,vc]); cart=(A@frac.T).T; N=len(frac)
    bonds=[]; imgs=[(i,j,k) for i in(-1,0,1) for j in(-1,0,1) for k in(-1,0,1)]
    for i in range(N):
        ri=cart[i]; ri_r=rad(elem(types[i]))
        for j in range(i+1,N):
            rj_r=rad(elem(types[j])); cutoff=(scale*(ri_r+rj_r))**2
            best=None; best_t=None
            for (tx,ty,tz) in imgs:
                rj=cart[j]+tx*va+ty*vb+tz*vc; d2=np.dot(rj-ri,rj-ri)
                if best is None or d2<best: best,best_t=d2,(tx,ty,tz)
            if best<=cutoff: bonds.append((i,j,best_t))
    return bonds

# ---------- replicate + spherical mask ----------
def super_sphere(frac,types,bonds_uc,va,vb,vc,n=15,R=6):
    ctr=(n-1)/2; index={}; coords=[]; elems=[]; gid=0
    for ci in range(n):
        for cj in range(n):
            for ck in range(n):
                T=ci*va+cj*vb+ck*vc
                for ai,f in enumerate(frac):
                    dx,dy,dz=ci+f[0]-ctr, cj+f[1]-ctr, ck+f[2]-ctr
                    if dx*dx+dy*dy+dz*dz<=R*R:
                        gid+=1; index[(ci,cj,ck,ai)]=gid
                        coords.append(T+f[0]*va+f[1]*vb+f[2]*vc)
                        elems.append(elem(types[ai]))
    coords=np.array(coords)
    bonds=[]
    for ci in range(n):
        for cj in range(n):
            for ck in range(n):
                for (i,j,(tx,ty,tz)) in bonds_uc:
                    di,dj,dk=ci+tx,cj+ty,ck+tz
                    if not(0<=di<n and 0<=dj<n and 0<=dk<n): continue
                    ka=(ci,cj,ck,i); kb=(di,dj,dk,j)
                    if ka in index and kb in index:
                        a,b=index[ka],index[kb]
                        bonds.append((min(a,b),max(a,b)))
    bonds=sorted(set(bonds))
    return coords,elems,bonds

# ---------- MOL2 writer ----------
def write_mol2(path,coords,elems,bonds):
    with open(path,"w") as f:
        f.write("@<TRIPOS>MOLECULE\nUiO-66_15x15x15_sphere_R6\n")
        f.write(f"{len(coords)} {len(bonds)} 0 0 0\nSMALL\nUSER_CHARGES\n@<TRIPOS>ATOM\n")
        for i,(xyz,el) in enumerate(zip(coords,elems),1):
            x,y,z=xyz; f.write(f"{i} {el}{i} {x:.4f} {y:.4f} {z:.4f} {el} 1 U66 0.0000\n")
        f.write("@<TRIPOS>BOND\n")
        for k,(a,b) in enumerate(bonds,1): f.write(f"{k} {a} {b} 1\n")

# ---------- main ----------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cif",required=True); ap.add_argument("--out",default="UiO-66_15x15x15_sphere_R6.mol2")
    ap.add_argument("--scale",type=float,default=1.2)
    ap.add_argument("--R",type=float,default=6.0)
    args=ap.parse_args()

    params,frac,types=read_cif_p1(args.cif)
    va,vb,vc,A=lattice_vectors(params)
    bonds_uc=infer_pbc_bonds(frac,types,va,vb,vc,scale=args.scale)
    coords,elems,bonds=super_sphere(frac,types,bonds_uc,va,vb,vc,n=15,R=args.R)
    write_mol2(args.out,coords,elems,bonds)
    print(f"Wrote {args.out}\nAtoms: {len(coords)}  Bonds: {len(bonds)}")