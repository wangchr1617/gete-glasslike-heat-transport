"""Paths, table I/O and HNEMD estimators shared by reproduction commands."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "minimal_reproduction" / "outputs"
TEMPERATURES = (300,400,500,550,585,600,650,700,800)
ALL_TEMPERATURES = (300,400,500,550,560,570,580,585,590,600,610,620,630,640,650,700,800)
IMPOSED = ((550,"cubic_metric"),(585,"rhombo_metric"),(585,"cubic_metric"),(650,"rhombo_metric"))

def read_csv(path):
    with Path(path).open(encoding="utf8",newline="") as stream:
        return list(csv.DictReader(stream))

def write_csv(path, rows):
    rows=list(rows)
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def output_dir(path=None):
    path=Path(path or DEFAULT_OUTPUT).resolve()
    # Reproduction output must never replace archived inputs or data.
    if path == ROOT or (path.is_relative_to(ROOT) and not path.is_relative_to(DEFAULT_OUTPUT)):
        raise ValueError("Inside the repository, use minimal_reproduction/outputs/ for generated files")
    path.mkdir(parents=True,exist_ok=True)
    return path

def native_kappa_path(t,direction):
    if t in (300,585,800):
        return ROOT/f"05_supplemental/S1_hnemd_convergence/{direction}/{t}K/kappa.out"
    return ROOT/f"01_A_tensor_transport/raw/{t}K/{direction}/kappa.out"

def hnemd(path,direction):
    data=np.loadtxt(path,ndmin=2)
    if data.shape != (10000,5) or not np.isfinite(data).all():
        raise ValueError(f"Expected 10000 finite 5-column HNEMD samples: {path}")
    values=data[:,4] if direction=="para" else data[:,0]+data[:,1]
    blocks=values.reshape(10,1000).mean(axis=1)
    return float(values.mean()),float(blocks.std(ddof=1)/np.sqrt(10))

def tensor_row(t,parallel,perpendicular):
    p,sp=parallel
    x,sx=perpendicular
    if p<=0 or x<=0:
        raise ValueError("Positive directional conductivities required for ratio propagation")
    r=p/x
    return dict(temperature_K=t,kappa_parallel_W_mK=p,kappa_parallel_sem=sp,
                kappa_perpendicular_W_mK=x,kappa_perpendicular_sem=sx,
                kappa_trace_W_mK=(p+2*x)/3,kappa_trace_sem=np.hypot(sp,2*sx)/3,
                anisotropy_ratio=r,anisotropy_ratio_sem=r*np.hypot(sp/p,sx/x))

def positive_integral(t,c,cap=None):
    t,c=np.asarray(t),np.asarray(c)
    if t.ndim!=1 or c.shape!=t.shape or not np.isfinite(c).all() or not np.all(np.diff(t)>0):
        raise ValueError("Finite correlations on a strictly increasing time grid required")
    crossing=np.flatnonzero(c[1:]<=0)
    stop=int(crossing[0]+1) if len(crossing) else len(c)
    if cap is not None:
        stop=min(stop,int(np.searchsorted(t,cap,side="right")))
    return float(np.trapezoid(c[:stop],t[:stop]))
