"""Rebuild a volume-matched imposed metric from archived native structures."""
import argparse
from pathlib import Path
import re
import numpy as np

def read(path):
    with Path(path).open(encoding="utf8") as stream:
        n=int(stream.readline());header=stream.readline()
        cell=np.fromstring(re.search(r'Lattice="([^"]+)"',header).group(1),sep=" ").reshape(3,3)
        rows=[stream.readline().split() for _ in range(n)]
    return [r[0] for r in rows],np.array([r[1:4] for r in rows],dtype=float),cell

def transform(native,template):
    species,positions,cell=read(native)
    _,_,metric=read(template)
    new=metric*(abs(np.linalg.det(cell))/abs(np.linalg.det(metric)))**(1/3)
    fractional=positions@np.linalg.inv(cell)
    return species,(fractional%1.0)@new,new

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("native",type=Path);p.add_argument("template",type=Path);p.add_argument("output",type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    sp,pos,cell=transform(args.native,args.template)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf8",newline="\n") as out:
        out.write(f'{len(sp)}\npbc="T T T" Lattice="'+" ".join(f"{v:.10f}" for v in cell.ravel())+'" Properties=species:S:1:pos:R:3\n')
        for symbol,x in zip(sp,pos):out.write(f"{symbol} {x[0]:.10f} {x[1]:.10f} {x[2]:.10f}\n")

if __name__=="__main__":main()
