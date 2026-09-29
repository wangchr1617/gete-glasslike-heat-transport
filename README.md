# GeTe glasslike heat transport | GeTe 玻璃态热输运

**中文。** 本仓库对应 PRB 手稿 *Glasslike Heat Transport and Collective Orientational Decorrelation in GeTe*（2026 年 9 月 27 日版本）。目录按正文 Results A–D、补充材料 S1–S5 排列，提供数值数据、图像和模拟输入，便于复现关键结论并进一步分析。

**English.** This repository accompanies the 27 September 2026 PRB manuscript *Glasslike Heat Transport and Collective Orientational Decorrelation in GeTe*. It follows Results A–D and Supplemental Material S1–S5, providing numerical data, figures, and simulation inputs for reproducing key conclusions and further analysis.

## 目录 / Contents

| 顺序 / Order | 结果数据 / Results | 输入 / Inputs |
| --- | --- | --- |
| A · Fig. 1 | [`tensor_transport.csv`](01_A_tensor_transport/tensor_transport.csv): HNEMD 张量分量、迹均值与各向异性 / tensor components, trace mean, and anisotropy | [`01_A_tensor_transport/inputs/`](01_A_tensor_transport/inputs/): 17 个温度的结构，平行与垂直方向的 `run.in` / structures at 17 temperatures and both directional runs |
| B · Fig. 2 | [`native_pjt_summary.csv`](02_B_pjt_dynamics/native_pjt_summary.csv) 与 [`replicas/`](02_B_pjt_dynamics/replicas/): 局域和集体 PJT 动力学 / local and collective PJT dynamics | [`02_B_pjt_dynamics/inputs/`](02_B_pjt_dynamics/inputs/): 12 条原生晶胞轨迹的 `run.in` / 12 native-cell run controls |
| C · Fig. 3 | [`fixed_metric_conductivity.csv`](03_C_fixed_metric/fixed_metric_conductivity.csv) 与 [`fixed_metric_pjt_summary.csv`](03_C_fixed_metric/fixed_metric_pjt_summary.csv): 固定度量的热输运与取向有序 / fixed-metric transport and orientational order | [`03_C_fixed_metric/inputs/`](03_C_fixed_metric/inputs/): 四种温度–度量条件的结构、8 条 HNEMD 与 12 条 PJT `run.in` / four model structures and 20 run controls |
| D · Fig. 4 | [`modal_summary.csv`](04_D_modal_transport/modal_summary.csv) 与 [`signed_frequency_density.csv`](04_D_modal_transport/signed_frequency_density.csv): 有效频率与 GKMA 投影 / effective frequencies and GKMA projections | [`04_D_modal_transport/inputs/`](04_D_modal_transport/inputs/): 九温度的 686 原子结构、`eigenvector.in`、原胞与 90 条 GKMA `run.in` / nine temperature-specific model structures, eigenvector bases, primitive cells, and 90 run controls |
| S1 · Fig. S1 | [`S1_hnemd_convergence/`](05_supplemental/S1_hnemd_convergence/): 300、585、800 K 双方向 `kappa.out` / six directional HNEMD outputs | 使用 A 的输入 / Use A inputs |
| S2 · Fig. S2 | [`S2_vacf/`](05_supplemental/S2_vacf/): 17 个原生体积 `mvac.out` / native-volume VACF outputs | 同目录的 [`inputs/`](05_supplemental/S2_vacf/inputs/): 51 个 `run.in`，以及 ±1% 体积结构 / 51 run controls and ±1% volume structures |
| S3 · Fig. S3 | 九个温度的 [`fc2.hdf5` 与 `POSCAR`](05_supplemental/S3_effective_modes/) / nine second-order force-constant files and unit cells | 同左 / Same files |
| S4 · Fig. S4 | [`gkma_running_integrals.csv`](05_supplemental/S4_gkma_convergence/gkma_running_integrals.csv): 九温度 0–50 ps 累积热导 / nine running integrals | 使用 D 的输入 / Use D inputs |
| S5 | [`Fig_S1.png`–`Fig_S4.png`](05_supplemental/S5_figures/) / reference images | — |

## 输入组装与运行 / Preparing and running a case

每个 `run.in` 保留原物理参数和随机种子，`potential` 路径统一为同目录的 `nep.txt`。统一 NEP 文件是 [`inputs/nep.txt`](inputs/nep.txt)。GPUMD 需要在运行目录中同时找到 `run.in`、`model.xyz` 和 `nep.txt`；GKMA 还需要 `eigenvector.in`。结构和本征矢按温度或度量存放，运行前复制到所选案例目录。

Each `run.in` retains its physical parameters and random seed; its `potential` path uses local `nep.txt`. The shared potential is [`inputs/nep.txt`](inputs/nep.txt). GPUMD expects `run.in`, `model.xyz`, and `nep.txt` in the case directory; GKMA additionally needs `eigenvector.in`. Structures and eigenvectors are stored by temperature or metric; copy them into the selected case directory before running.

例如，从仓库根目录准备 A 中 585 K 平行方向案例 / For example, prepare the 585 K parallel A case from the repository root:

```sh
CASE=01_A_tensor_transport/inputs/585K/para
cp inputs/nep.txt "$CASE/nep.txt"
cp 01_A_tensor_transport/inputs/585K/model.xyz "$CASE/model.xyz"
(cd "$CASE" && gpumd)
```

其他案例的模型来源 / Model source for other cases:

| 案例 / Case | 将 `model.xyz` 从何处复制 / Copy `model.xyz` from | 其他文件 / Additional file |
| --- | --- | --- |
| B native PJT | `01_A_tensor_transport/inputs/<T>K/model.xyz` | — |
| C HNEMD and fixed-metric PJT | `03_C_fixed_metric/inputs/<T>K/<metric>/model.xyz` | — |
| D GKMA | `04_D_modal_transport/inputs/<T>K/model.xyz` | Copy `04_D_modal_transport/inputs/<T>K/eigenvector.in` into the case directory |
| S2 VACF at V0 | `01_A_tensor_transport/inputs/<T>K/model.xyz` | — |
| S2 VACF at Vm1 or Vp1 | The `model.xyz` in the same volume folder | — |

## 数值口径 / Numerical definitions

`κtr = (κ∥ + 2κ⊥)/3`，各向异性比为 `κ∥/κ⊥`。A 表格中的十个 1 ns 时间块误差由原始 CSV 的总体标准差口径乘以 `sqrt(10/9)`，转换为样本标准误。B 的 585 K 值采用六条独立轨迹，550 和 650 K 各采用三条。D 的负号有效频率标记拟合二次势能面的负曲率方向，不代表负频传播声子。550→585 K 的负曲率子空间 GKMA 投影约为 0.002→0.088 W m⁻¹ K⁻¹。

`κtr = (κ∥ + 2κ⊥)/3`, and the anisotropy ratio is `κ∥/κ⊥`. The A table multiplies the source CSV's population-normalized block errors by `sqrt(10/9)` to report sample SEM for ten 1 ns blocks. B uses six independent 585 K trajectories and three at each of 550 and 650 K. In D, a negative signed effective frequency labels negative curvature of the fitted quadratic surface, not a propagating negative-frequency phonon. From 550 to 585 K, the GKMA projection into that subspace changes from about 0.002 to 0.088 W m⁻¹ K⁻¹.
