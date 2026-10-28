#!/usr/bin/env python3
"""
convert_eigenvectors.py

将 HiPhive FORCE_CONSTANTS (Phonopy 格式) 转换为 GPUMD compute_gkma
所需的 eigenvector.in 二进制文件。

用法:
    python convert_eigenvectors.py <FORCE_CONSTANTS路径> <model.xyz路径> [输出路径]

    默认输出: 当前目录下的 eigenvector.in

GPUMD eigenvector.in 二进制格式 (参考源码 phonon/hessian.cu:460-491):
    [float × dim]     ω² 值 (单位: (2π·THz)², 符号标记虚频)
    [float × dim²]    本征矢矩阵 (列主序, dim × dim)
                      写入顺序: mode × (x_all, y_all, z_all)

其中 ω² 的转换因子 natural_to_THz = 1e6 / TIME_UNIT_CONVERSION² ≈ 9649,
将 eV/(amu·Å²) → (2π·THz)².
"""

import numpy as np
import struct
import sys
import os

# ── 物理常数 ──────────────────────────────────────────────
# GPUMD 自然时间单位: t_nat = sqrt(amu * Å² / eV) ≈ 1.01805e-14 s
# TIME_UNIT_CONVERSION = t_nat / 1e-15 s = 10.1805 (fs per natural time unit)
TIME_UNIT_CONVERSION = 10.180505772308774
NATURAL_TO_THZ = 1.0e6 / (TIME_UNIT_CONVERSION * TIME_UNIT_CONVERSION)

# 原子质量 (amu)
ATOMIC_MASSES = {
    "Ge": 72.630,
    "Te": 127.603,
}


def read_force_constants_phonopy(filepath):
    """读取 Phonopy 格式的 FORCE_CONSTANTS 文件.

    格式:
        n_atom  n_atom
        i  j
        Φ_xx Φ_xy Φ_xz
        Φ_yx Φ_yy Φ_yz
        Φ_zx Φ_zy Φ_zz
        ...

    Returns:
        fc: (n_atom, n_atom, 3, 3) ndarray in eV/Å²
    """
    if filepath.lower().endswith((".hdf5", ".h5")):
        import h5py
        with h5py.File(filepath,"r") as stream:
            fc=np.asarray(stream["force_constants"][:])
        if fc.ndim!=4 or fc.shape[0]!=fc.shape[1]:
            raise ValueError("A full, square second-order force-constant array is required")
        return fc
    with open(filepath) as f:
        lines = f.readlines()

    n_atom = int(lines[0].split()[0])
    fc = np.zeros((n_atom, n_atom, 3, 3))

    line_idx = 1  # skip the first header line (n_atom n_atom)
    for _ in range(n_atom * n_atom):
        parts = lines[line_idx].split()
        i = int(parts[0]) - 1
        j = int(parts[1]) - 1
        line_idx += 1
        for a in range(3):
            vals = lines[line_idx].split()
            line_idx += 1
            for b in range(3):
                fc[i, j, a, b] = float(vals[b])

    return fc


def read_masses_from_xyz(filepath):
    """从 extended xyz 文件读取原子种类和质量列表.

    Returns:
        masses: (n_atom,) ndarray in amu
        species: list of str
    """
    with open(filepath) as f:
        n_atom = int(f.readline().strip())
        f.readline()  # comment line
        species = []
        for _ in range(n_atom):
            sp = f.readline().split()[0]
            species.append(sp)

    masses = np.array([ATOMIC_MASSES[sp] for sp in species])
    return masses, species


def build_dynamical_matrix(fc, masses):
    """从力常数和质量构建动力学矩阵.

    D_{iα, jβ} = Φ_{iα, jβ} / sqrt(m_i * m_j)

    Args:
        fc: (n_atom, n_atom, 3, 3) force constants in eV/Å²
        masses: (n_atom,) atomic masses in amu

    Returns:
        D: (3*n_atom, 3*n_atom) mass-weighted dynamical matrix
    """
    n_atom = fc.shape[0]
    dim = 3 * n_atom
    D = np.zeros((dim, dim))

    inv_sqrt_m = 1.0 / np.sqrt(masses)
    for i in range(n_atom):
        for j in range(n_atom):
            D[3 * i:3 * i + 3, 3 * j:3 * j + 3] = fc[i, j] * inv_sqrt_m[i] * inv_sqrt_m[j]

    return D


def validate_acoustic_sum_rule(D, tol=1e-10):
    """验证声学求和规则: Γ 点应有 3 个零频模式."""
    eigvals = np.linalg.eigvalsh(D)
    n_zero = np.sum(np.abs(eigvals) < tol)
    if n_zero < 3:
        print(f"  ⚠ 警告: 仅检测到 {n_zero} 个零频模式 (预期 ≥3)")
        print(f"     最小 6 个特征值: {eigvals[:6]}")
    else:
        print(f"  ✓ 声学求和规则满足 ({n_zero} 个零频模式)")
    return eigvals


def write_eigenvector_gpumd(filepath, eigvals, eigvecs, masses, species):
    """写入 GPUMD 格式的 eigenvector.in 二进制文件.

    GPUMD 格式 (参考 phonon/hessian.cu:460-491):
      1. dim 个 float: ω² 值, 单位 (2π·THz)²
         (正值=实频, 负值=虚频)
      2. dim × dim 个 float: 本征矢矩阵 (列主序)
         写入顺序: 对每个 mode:
           x 分量 (全部原子) → y 分量 (全部原子) → z 分量 (全部原子)

    Args:
        filepath: 输出文件路径
        eigvals: (dim,) 特征值 in eV/(amu·Å²) — GPUMD 自然单位
        eigvecs: (dim, dim) 本征矢矩阵, 列主序
        masses: (n_atom,) 原子质量
        species: (n_atom,) 原子种类列表
    """
    dim = len(eigvals)
    n_atom = dim // 3

    # 转换特征值: eV/(amu·Å²) → (2π·THz)²
    om2_thz = (eigvals * NATURAL_TO_THZ).astype(np.float32)

    with open(filepath, "wb") as f:
        # 写入特征值 (ω²)
        f.write(om2_thz.tobytes())

        # 写入本征矢
        # GPUMD 顺序: mode (col), then dir (x,y,z), then atom
        for col in range(dim):
            for a in range(3):  # x, y, z
                for b in range(n_atom):
                    row = a + b * 3
                    val = np.float32(eigvecs[row, col])
                    f.write(struct.pack("f", val))

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  ✓ 已写入: {filepath} ({file_size_mb:.1f} MB)")
    print(f"    dim={dim}, n_atom={n_atom}")


def print_frequency_summary(eigvals, species):
    """打印频率分布摘要."""
    om2_thz = eigvals * NATURAL_TO_THZ
    nu_thz = np.sign(om2_thz) * np.sqrt(np.abs(om2_thz)) / (2.0 * np.pi)

    dim = len(eigvals)
    n_imag = np.sum(om2_thz < -1e-12)
    n_zero = np.sum(np.abs(om2_thz) < 1e-12)
    n_real = dim - n_imag - n_zero

    print(f"\n  频率分布 (共 {dim} 个模式):")
    print(f"    实频: {n_real}, 虚频: {n_imag}, 零频: {n_zero}")
    print(f"    范围: {nu_thz[n_zero:].min():.4f} – {nu_thz.max():.4f} THz")

    if n_imag > 0:
        imag_modes = np.where(om2_thz < -1e-12)[0]
        print(f"    虚频模式 (idx, ν/THz):")
        for idx in imag_modes[:6]:
            print(f"      mode {idx}: {nu_thz[idx]:.4f} THz")

    return nu_thz


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("用法: python convert_eigenvectors.py <FORCE_CONSTANTS> <model.xyz> [输出路径]")
        sys.exit(1)

    fc_path = sys.argv[1]
    xyz_path = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "eigenvector.in"

    if os.path.exists(out_path):
        raise FileExistsError(out_path)
    print(f"Force constants: {fc_path}")
    print(f"model.xyz:       {xyz_path}")
    print(f"输出:            {out_path}")

    # 1. 读取力常数
    print("\n[1/4] 读取力常数...")
    fc = read_force_constants_phonopy(fc_path)
    n_atom = fc.shape[0]
    dim = 3 * n_atom
    print(f"  n_atom={n_atom}, dim={dim}, FC shape={fc.shape}")

    # 2. 读取质量
    print("\n[2/4] 读取原子质量...")
    masses, species = read_masses_from_xyz(xyz_path)
    assert len(masses) == n_atom, f"model.xyz 原子数 ({len(masses)}) ≠ FORCE_CONSTANTS 原子数 ({n_atom})"
    ge_count = sum(1 for s in species if s == "Ge")
    te_count = sum(1 for s in species if s == "Te")
    print(f"  Ge: {ge_count}, Te: {te_count}")

    # 3. 构建并对角化动力学矩阵
    print("\n[3/4] 构建动力学矩阵并对角化...")
    D = build_dynamical_matrix(fc, masses)
    eigvals = validate_acoustic_sum_rule(D)

    # 对角化 (eigh 保证本征矢为实, 列主序)
    eigvals, eigvecs = np.linalg.eigh(D)

    print_frequency_summary(eigvals, species)

    # 4. 写入 eigenvector.in
    print("\n[4/4] 写入 GPUMD eigenvector.in...")
    write_eigenvector_gpumd(out_path, eigvals, eigvecs, masses, species)

    print("\n完成! 将 eigenvector.in 放置到 GKMA 运行目录中即可使用.")


if __name__ == "__main__":
    dim = 0  # for print_frequency_summary reference
    main()
