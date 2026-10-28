"""Replot the four main and six supplemental figures from this repository."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from common import ROOT, output_dir

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--only",nargs="+",help="Figure IDs: 1 2 3 4 S1 S2 S3 S4 S5 S6")
    p.add_argument("--formats",nargs="+",choices=["png","pdf","svg"],default=["png"],
                   help="PNG is always written; optionally add editable PDF/SVG")
    args=p.parse_args()
    destination=output_dir(args.output_dir)/"figures"
    destination.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parent/"figures"
    files=sorted(source.glob("fig_*.py"))
    selected={x.upper() for x in args.only} if args.only else None
    available={str(int(f.name.split('_')[1])) if not f.name.split('_')[1].startswith('s') else
               'S'+str(int(f.name.split('_')[1][1:])) for f in files}
    if selected and not selected<=available:
        p.error(f"Unknown figure IDs: {selected-available}")
    env=dict(os.environ,GETE_FIGURE_OUTPUT=str(destination),
             GETE_FIGURE_FORMATS=','.join(args.formats),PYTHONUTF8="1",
             PYTHONDONTWRITEBYTECODE="1",MPLCONFIGDIR=str(output_dir(args.output_dir)/"_mplconfig"),
             OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    for file in files:
        part=file.name.split('_')[1]
        label=('S'+str(int(part[1:]))) if part.startswith('s') else str(int(part))
        if selected is not None and label not in selected:continue
        print(f"Rendering Fig. {label}",flush=True)
        subprocess.run([sys.executable,"-B",str(file)],cwd=ROOT,env=env,check=True)

if __name__=="__main__":main()
