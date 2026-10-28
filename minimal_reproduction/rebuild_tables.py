"""Rebuild tensor, PJT and GKMA tables from archived numerical outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import (ROOT,ALL_TEMPERATURES,TEMPERATURES,IMPOSED,output_dir,read_csv,
                    write_csv,native_kappa_path,hnemd,tensor_row,positive_integral)

def native():
    return [tensor_row(t,hnemd(native_kappa_path(t,"para"),"para"),
                       hnemd(native_kappa_path(t,"perp"),"perp")) for t in ALL_TEMPERATURES]

def fixed():
    rows=[]
    for t,metric in sorted(list(IMPOSED)+[(t,"native_metric") for t in (550,585,650)]):
        paths=[native_kappa_path(t,d) if metric=="native_metric" else
               ROOT/f"03_C_fixed_metric/raw/{t}K/{metric}/{d}/kappa.out" for d in ("para","perp")]
        r=tensor_row(t,hnemd(paths[0],"para"),hnemd(paths[1],"perp"))
        rows.append(dict(temperature_K=t,metric=metric,
                         source="native_HNEMD" if metric=="native_metric" else "imposed_metric_HNEMD",
                         kappa_para=r["kappa_parallel_W_mK"],kappa_para_sem=r["kappa_parallel_sem"],
                         kappa_perp=r["kappa_perpendicular_W_mK"],kappa_perp_sem=r["kappa_perpendicular_sem"],
                         kappa_trace=r["kappa_trace_W_mK"],kappa_trace_sem=r["kappa_trace_sem"],
                         anisotropy_ratio=r["anisotropy_ratio"],anisotropy_ratio_sem=r["anisotropy_ratio_sem"]))
    return rows

def pjt_groups():
    for t,n in [(550,3),(585,6),(650,3)]:
        yield t,"native_metric",[ROOT/f"02_B_pjt_dynamics/replicas/{t}K/traj_{i:02d}" for i in range(1,n+1)]
    for t,metric in IMPOSED:
        yield t,metric,[ROOT/f"03_C_fixed_metric/pjt_replicas/{t}K/{metric}/traj_{i:02d}" for i in range(1,4)]

def pjt():
    grouped,replicas=[],[]
    keys=["mean_local_pjt_A","mean_collective_order_ratio","connected_relaxation_time_ps",
          "collective_relaxation_time_ps","low_frequency_weight_0_0p5_THz"]
    for t,metric,paths in pjt_groups():
        summaries=[]
        for path in paths:
            s=json.loads((path/"summary.json" if metric=="native_metric" else path.with_suffix(".json")).read_text())
            c=np.genfromtxt(path/"correlation.csv",delimiter=",",names=True)
            tau=positive_integral(c["time_ps"],c["C_collective_connected"])
            np.testing.assert_allclose(tau,s["collective_relaxation_time_ps"],rtol=1e-8,atol=1e-8)
            replicas.append(dict(temperature_K=t,metric=metric,replica=s["replica"],
                                 collective_tau_ps=tau,collective_tau_capped_10ps_ps=positive_integral(
                                     c["time_ps"],c["C_collective_connected"],cap=10)))
            summaries.append(s)
        r=dict(temperature_K=t,metric=metric,n_replicas=len(summaries))
        for key in keys:
            x=np.array([s[key] for s in summaries])
            r[key]=float(x.mean());r[key+"_sem"]=float(x.std(ddof=1)/np.sqrt(len(x)))
        grouped.append(r)
    return grouped,replicas

def modal():
    rows,modes,bins=[],[],[]
    edges=np.arange(-1.8,5.6+0.10,0.10)
    for t in TEMPERATURES:
        with np.load(ROOT/f"04_D_modal_transport/processed/{t}K/modal_results.npz",allow_pickle=False) as z:
            eig=np.fromfile(ROOT/f"04_D_modal_transport/inputs/{t}K/eigenvector.in",dtype=np.float32,count=2058)
            f=np.sign(eig)*np.sqrt(np.abs(eig))/(2*np.pi)
            k=z["kappa_mode_total_xyz_mean"].mean(axis=1)
            values=z["kappa_total_scalar_values"]
            run=z["running_kappa_total_xyz_mean"].mean(axis=0)
            time=np.arange(1,len(run)+1)*json.loads(str(z["metadata_json"]))["sample_dt_fs"]/1000
            masks=[f<0,(f>=0)&(f<2),f>=2]
            sector=[float(k[m].sum()) for m in masks]
            rows.append(dict(temperature_K=t,kappa_negative_curvature_W_mK=sector[0],
                             kappa_0_to_2_THz_W_mK=sector[1],kappa_2_THz_and_above_W_mK=sector[2],
                             kappa_gkma_total_W_mK=float(values.mean()),
                             kappa_gkma_sem=float(values.std(ddof=1)/np.sqrt(len(values))),
                             kappa_30_50ps_mean_W_mK=float(run[(time>=30)&(time<=50)].mean())))
            modes.extend(dict(temperature_K=t,mode_index_1based=i+1,signed_frequency_THz=float(f[i]),
                              kappa_mode_W_mK=float(k[i])) for i in range(len(f)))
            density=np.histogram(f,bins=edges,weights=k)[0]/0.10
            bins.extend(dict(temperature_K=t,signed_frequency_THz=float((edges[i]+edges[i+1])/2),
                             kappa_density_W_mK_per_THz=float(v)) for i,v in enumerate(density))
    return rows,modes,bins

def rebuild(output):
    output=output_dir(output)/"tables"
    write_csv(output/"tensor_transport.csv",native())
    write_csv(output/"fixed_metric_conductivity_sample_sem.csv",fixed())
    grouped,replicas=pjt()
    write_csv(output/"pjt_summary.csv",grouped)
    write_csv(output/"pjt_relaxation_sensitivity.csv",replicas)
    rows,modes,bins=modal()
    write_csv(output/"modal_summary.csv",rows)
    write_csv(output/"modal_contributions.csv",modes)
    write_csv(output/"signed_frequency_density.csv",bins)
    from postprocess.vacf_centroids import rebuild as rebuild_centroids
    rebuild_centroids(output.parent)
    print(f"Rebuilt tables: {output}")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path)
    rebuild(p.parse_args().output_dir)
