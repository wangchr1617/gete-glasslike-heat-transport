"""Assemble an archived GPUMD case in a new output directory."""
import argparse
import json
from pathlib import Path
import shutil
from common import ROOT, DEFAULT_OUTPUT, output_dir, sha256

def cases():
    result={}
    groups={"A":"01_A_tensor_transport","B":"02_B_pjt_dynamics",
            "C":"03_C_fixed_metric","D":"04_D_modal_transport",
            "VACF":"05_supplemental/S2_vacf"}
    for family,folder in groups.items():
        base=ROOT/folder/"inputs"
        for run in sorted(base.rglob("run.in")):
            rel=run.parent.relative_to(base)
            model=run.parent/"model.xyz"
            if not model.exists():
                model=next((q/"model.xyz" for q in run.parents if (q/"model.xyz").is_file()),None)
            if family in ("B","VACF") and model is None:
                model=ROOT/"01_A_tensor_transport/inputs"/rel.parts[0]/"model.xyz"
            if model is None or not model.is_file():
                raise FileNotFoundError(f"Model missing for {run.relative_to(ROOT)}")
            files={"run.in":run,"model.xyz":model,"nep.txt":ROOT/"inputs/nep.txt"}
            if family=="D":
                files["eigenvector.in"]=base/rel.parts[0]/"eigenvector.in"
            result[f"{family}/{rel.as_posix()}"]=files
    return result

def prepare(case,destination):
    files=cases()[case]
    destination=Path(destination)
    # Refuse existing targets, including partially completed simulation runs.
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.mkdir(parents=True)
    meta={}
    for name,src in files.items():
        shutil.copyfile(src,destination/name)
        meta[name]={"source":src.relative_to(ROOT).as_posix(),"sha256":sha256(src)}
    (destination/"input_sources.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf8")
    return destination

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list",action="store_true")
    p.add_argument("--case",help="For example A/585K/para, B/585K/traj_01, D/585K/traj_01; or all")
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT/"runs")
    args=p.parse_args()
    available=cases()
    if args.list:
        print("\n".join(available));return
    if args.case not in available and args.case!="all":
        p.error("Choose --case from --list")
    dest=output_dir(args.output_dir)
    selected=list(available) if args.case=="all" else [args.case]
    if any((dest/name).exists() for name in selected):
        raise FileExistsError("One or more destinations already exist; choose a new output directory")
    for name in selected:
        print(prepare(name,dest/name))

if __name__=="__main__":main()
