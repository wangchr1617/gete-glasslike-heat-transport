"""Rebuild the three effective-frequency centroid volume derivatives in SM S2."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,ALL_TEMPERATURES,output_dir,write_csv

def rebuild(destination):
    rows=[]
    for temperature in ALL_TEMPERATURES:
        for lower,upper in [(3.0,6.0),(6.0,12.0),(12.0,20.0)]:
            centroids=[]
            for volume in ('Vm1','V0','Vp1'):
                file=ROOT/f'05_supplemental/S2_vacf/volume_outputs/{temperature}K/{volume}/dos.out'
                raw=np.loadtxt(file,ndmin=2)
                omega=raw[:,0];density=raw[:,1:4].mean(axis=1)
                use=(omega>=lower)&(omega<=upper)
                weight=np.trapezoid(density[use],omega[use])
                if not np.isfinite(weight) or weight<=1e-10:
                    raise ValueError(f'Insufficient centroid weight: {file}, {lower}--{upper}')
                centroids.append(float(np.trapezoid(omega[use]*density[use],omega[use])/weight))
            gamma=-float(np.polyfit(np.log([0.99,1,1.01]),np.log(centroids),1)[0])
            rows.append(dict(temperature_K=temperature,band_lower_omega_ps_inv=lower,
                             band_upper_omega_ps_inv=upper,band_lower_frequency_THz=lower/(2*np.pi),
                             band_upper_frequency_THz=upper/(2*np.pi),
                             centroid_Vm1_THz=centroids[0]/(2*np.pi),centroid_V0_THz=centroids[1]/(2*np.pi),
                             centroid_Vp1_THz=centroids[2]/(2*np.pi),effective_centroid_gruneisen=gamma))
    write_csv(output_dir(destination)/'tables/vacf_centroid_volume_derivatives.csv',rows)
    print('Rebuilt 51 temperature/band centroid derivatives from the 51 DOS files')
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    rebuild(parser.parse_args().output_dir)
