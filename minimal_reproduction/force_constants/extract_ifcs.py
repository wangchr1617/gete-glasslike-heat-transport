#!/usr/bin/env python3
"""Joint second/third-order HiPhive fit with the archived LASSO/RFE/OLS procedure."""
import argparse
import contextlib
import gc
from pathlib import Path
import sys
import time
import numpy as np

def compute_displacements_mic(atoms, ref_atoms):
    cell = ref_atoms.cell.array
    ref_scaled = ref_atoms.get_scaled_positions()
    disp_scaled = atoms.get_scaled_positions()
    diff = disp_scaled - ref_scaled
    diff -= np.round(diff)
    return diff @ cell

def build_fit_data(fcm, ideal_supercell, structures, cs, batch_size=50):
    n_structures = len(structures)
    n_atoms = len(ideal_supercell)
    n_fp = n_atoms * 3
    n_params = cs.n_dofs
    M = np.zeros((n_structures * n_fp, n_params), dtype=np.float32)
    F = np.zeros(n_structures * n_fp, dtype=np.float32)
    t0 = time.time()
    for start in range(0, n_structures, batch_size):
        end = min(start + batch_size, n_structures)
        for i in range(start, end):
            d = compute_displacements_mic(structures[i], ideal_supercell)
            M_i = fcm.get_fit_matrix(d).astype(np.float32)
            f_i = structures[i].get_forces().ravel().astype(np.float32)
            r = i * n_fp
            M[r:r + n_fp] = M_i
            F[r:r + n_fp] = f_i
        elapsed = time.time() - t0
        print(f"  [{end}/{n_structures}] ~{end/(elapsed+.001):.0f} str/s, "
              f"~{end*n_fp*n_params*4/1e9:.1f} GB")
    gc.collect()
    return M, F

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except ValueError:
                pass

def lasso_rfe_ols(M, F_target, n_dofs, n_structures, n_atoms,
                  alpha_range, cv_folds, rfe_steps, label="", n_jobs=1):
    """Joint LASSO + RFE + OLS pipeline. Returns (params, r2, rmse_ols)."""
    from sklearn.linear_model import LassoCV
    from sklearn.metrics import mean_squared_error

    t0 = time.time()
    lasso = LassoCV(alphas=alpha_range, cv=cv_folds, max_iter=10000,
                    tol=1e-3, n_jobs=n_jobs, selection='random',
                    random_state=42, fit_intercept=False,
                    precompute=True, copy_X=False)
    lasso.fit(M, F_target)
    rmse = np.sqrt(mean_squared_error(F_target, lasso.predict(M)))
    print(f"  {label} LASSO: alpha={lasso.alpha_:.4e}, R2={lasso.score(M,F_target):.6f}, "
          f"RMSE={rmse:.6f} eV/A")

    params = lasso.coef_.copy()
    best_score = lasso.score(M, F_target)
    best_coef = lasso.coef_.copy()

    if rfe_steps > 0:
        coef = lasso.coef_.copy()
        support = np.abs(coef) > 1e-10
        n_feat = support.sum()
        print(f"  {label} RFE initial: {n_feat}/{n_dofs} non-zero")
        n_elim = max(1, n_feat // rfe_steps)

        for step in range(rfe_steps):
            if support.sum() <= 1:
                break
            n_kill = min(n_elim, support.sum() - 1)
            nz = np.where(support)[0]
            kill = nz[np.argsort(np.abs(coef[support]))[:n_kill]]
            support[kill] = False
            if support.sum() == 0:
                break

            ls = LassoCV(alphas=alpha_range, cv=min(cv_folds, 3),
                         max_iter=10000, tol=1e-3, n_jobs=n_jobs,
                         selection='random', random_state=42,
                         fit_intercept=False,
                         precompute=True, copy_X=False)
            ls.fit(M[:, support], F_target)
            cn = np.zeros(n_dofs)
            cn[support] = ls.coef_
            sc = ls.score(M[:, support], F_target)
            if sc > best_score:
                best_score, best_coef = sc, cn.copy()
            coef = cn
            if (step+1) % 5 == 0 or step == rfe_steps-1:
                print(f"    {label} Step {step+1}: {support.sum()} params, "
                      f"alpha={ls.alpha_:.2e}, R2={sc:.6f}")

        params = best_coef
        n_final = np.sum(np.abs(params) > 1e-10)
        print(f"  {label} RFE final: {n_final}/{n_dofs} non-zero, R2={best_score:.6f}")

        # OLS polishing
        sf = np.abs(params) > 1e-10
        if 0 < sf.sum() < M.shape[0]:
            M_sel = M[:, sf].astype(np.float64)
            ols_coef = np.linalg.lstsq(M_sel, F_target.astype(np.float64),
                                       rcond=None)[0]
            pp = np.zeros(n_dofs)
            pp[sf] = ols_coef

            # Validate OLS on a subset (use design matrix directly)
            n_check = min(10, n_structures)
            n_fp_check = n_check * n_atoms * 3
            f_pred = (M[:n_fp_check].astype(np.float64) @ pp.astype(np.float64)).ravel()
            f_true = F_target[:n_fp_check].ravel()

            rmse_p = np.sqrt(np.mean((f_pred - f_true) ** 2))
            print(f"    {label} OLS RMSE: {rmse_p:.6f} eV/A "
                  f"({'accepted' if rmse_p < rmse * 1.1 else 'rejected'})")
            if rmse_p < rmse * 1.1:
                params = pp
                best_score = 1 - np.sum((f_pred - f_true) ** 2) / np.sum((f_true - f_true.mean()) ** 2)

    return params, best_score, rmse

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir",type=Path)
    parser.add_argument("--n-jobs",type=int,default=1)
    args=parser.parse_args()
    root=args.workdir.resolve()
    required=[root/name for name in ["model.xyz","primitive.xyz","displaced_structures_with_forces.extxyz"]]
    for path in required:
        if not path.is_file():raise FileNotFoundError(path)
    outputs=[root/name for name in ["fcm.pickle","FORCE_CONSTANTS","FORCE_CONSTANTS_3RD","extract_ifcs.log"]]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Fit outputs already exist; use a fresh calculation directory")
    from ase.io import read
    from hiphive import ClusterSpace
    from hiphive.cutoffs import CutoffMaximumBody
    from hiphive.force_constant_model import ForceConstantModel
    primitive=read(required[1]);ideal=read(required[0])
    structures=read(required[2],index=":")
    if len(structures)!=500 or any(len(at)!=len(ideal) for at in structures):
        raise ValueError("Expected all 500 force-evaluation configurations with matching atom counts")
    with outputs[3].open("w",encoding="utf8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)):
            cs=ClusterSpace(primitive,CutoffMaximumBody([8.0,8.0],3))
            model=ForceConstantModel(ideal,cs)
            matrix,force=build_fit_data(model,ideal,structures,cs)
            parameters,r2,rmse=lasso_rfe_ols(
                matrix,force,cs.n_dofs,len(structures),len(ideal),
                np.logspace(-6,-1,20),5,20,n_jobs=args.n_jobs)
            model.parameters=parameters
            constants=model.get_force_constants()
            model.write(str(outputs[0]))
            constants.write_to_phonopy(str(outputs[1]))
            constants.write_to_shengBTE(str(outputs[2]),primitive)
            print(f"Exported joint 2nd/3rd-order fit: R2={r2:.6f}, initial LASSO RMSE={rmse:.6f} eV/A")

if __name__=="__main__":main()
