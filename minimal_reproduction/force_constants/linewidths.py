"""Run or check a 15^3 tetrahedron phono3py half-linewidth calculation."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temperature',type=int,required=True,
                        choices=[300,400,500,550,585,600,650,700,800])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--check-inputs',action='store_true',
                        help='Load the archived force constants and mesh without calculating scattering')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    output=args.output_dir.resolve()
    if output.exists():raise FileExistsError(output)
    import phono3py
    import phonopy
    ph3=phono3py.load(
        unitcell_filename=root/f'05_supplemental/S3_effective_modes/{args.temperature}K/POSCAR',
        supercell_matrix=[7,7,7],primitive_matrix=np.eye(3),is_nac=False,
        fc2_filename=root/f'05_supplemental/S3_effective_modes/{args.temperature}K/fc2.hdf5',
        fc3_filename=root/f'05_supplemental/modal_descriptors/{args.temperature}K/fc3.hdf5',
        produce_fc=False,symmetrize_fc=False,lang='C',log_level=1)
    ph3.mesh_numbers=[15,15,15]
    ph3.sigmas=[None]  # tetrahedron integration
    metadata=dict(temperature_K=args.temperature,mesh=[15,15,15],sigmas=[None],
                  phono3py_version=phono3py.__version__,phonopy_version=phonopy.__version__,
                  fc2_shape=list(ph3.fc2.shape),fc3_shape=list(ph3.fc3.shape),
                  backend='C',scattering_calculated=False)
    output.mkdir(parents=True)
    if not args.check_inputs:
        os.chdir(output)
        ph3.init_phph_interaction()
        ph3.run_thermal_conductivity(temperatures=[args.temperature],
                                    write_gamma=True,write_kappa=True)
        metadata['scattering_calculated']=True
    (output/'calculation.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
