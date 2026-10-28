"""Prepare 500 displaced structures and their GPUMD force-evaluation inputs."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temperature",type=int,required=True,choices=[300,400,500,550,585,600,650,700,800])
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    here=Path(__file__).resolve().parent
    root=here.parents[1]
    dest=args.output_dir.resolve()
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    source=root/f"04_D_modal_transport/inputs/{args.temperature}K"
    shutil.copyfile(source/"model.xyz",dest/"model.xyz")
    # Preserve the temperature-specific primitive used by the archived fit.
    from ase.io import read,write
    write(dest/"primitive.xyz",read(source/"primitive.vasp"),format="extxyz")
    subprocess.run([sys.executable,str(here/"generate_displacements.py"),str(dest)],check=True)
    template=("potential nep.txt\nvelocity 300\ntime_step 1\nensemble nve\ndump_force 1\nrun 1\n")
    for structure in sorted((dest/"displacements").glob("structure_*.xyz")):
        case=dest/"force_eval"/structure.stem
        case.mkdir(parents=True)
        shutil.copyfile(structure,case/"model.xyz")
        shutil.copyfile(root/"inputs/nep.txt",case/"nep.txt")
        (case/"run.in").write_text(template,encoding="utf8")
    print(f"Prepared 500 force-evaluation cases in {dest/'force_eval'}")

if __name__=="__main__":main()
