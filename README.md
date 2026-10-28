# GeTe heat transport | GeTe 热输运

本仓库提供论文 *Collapse of Heat Transport Anisotropy and Collective Orientational Decorrelation in GeTe* 的数据、模拟输入和复现脚本。研究结果与讨论见正文及补充材料。

Data, simulation inputs and reproduction scripts for the accompanying manuscript.

## 目录 / Contents

| 目录 / Directory | 内容 / Contents |
| --- | --- |
| [01_A_tensor_transport/](01_A_tensor_transport/) | HNEMD 数据与输入 / HNEMD data and inputs |
| [02_B_pjt_dynamics/](02_B_pjt_dynamics/) | PJT 数据与输入 / PJT data and inputs |
| [03_C_fixed_metric/](03_C_fixed_metric/) | 固定晶胞数据与输入 / Fixed-cell data and inputs |
| [04_D_modal_transport/](04_D_modal_transport/) | GKMA 数据与输入 / GKMA data and inputs |
| [05_supplemental/](05_supplemental/) | 补充数据及补图索引 / Supplemental data and figure map |
| [minimal_reproduction/](minimal_reproduction/) | 复现程序 / Reproduction programs |

## 快速开始 / Quick start

在独立的 Python 3.12 环境中，从仓库根目录执行 / Run from the repository root in a dedicated Python 3.12 environment:

```sh
python -m pip install -r minimal_reproduction/requirements.txt
python minimal_reproduction/rebuild_tables.py
python minimal_reproduction/reproduce_figures.py
```

输出位于 `minimal_reproduction/outputs/`。重新运行模拟需另备 GPUMD。

Outputs are written to `minimal_reproduction/outputs/`. Simulation reruns require GPUMD.

[运行说明 / Running](docs/reproduction.md) · [数据定义 / Data](docs/data_dictionary.md) · [版本与口径 / Archive notes](docs/provenance.md)
