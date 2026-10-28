#!/usr/bin/env python3
"""
将 HiPhive CS 力常数文本格式 → phono3py HDF5。

fc2: ShengBTE 标准格式 → fc2.hdf5  (686, 686, 3, 3)
fc3: HiPhive FourPhonon 格式 → fc3.hdf5  compact: (2, 686, 686, 3, 3, 3)

用法:
    python convert_fc_to_hdf5.py <workdir>

读取 <workdir>/FORCE_CONSTANTS, FORCE_CONSTANTS_3RD, primitive.xyz
写入 <workdir>/fc2.hdf5, fc3.hdf5
"""

import sys
import os
import numpy as np
import h5py
from ase.io import read, write


# ─── fc2 解析（ShengBTE 标准格式）─────────────────────────────────────────

def parse_fc2_shengBTE(filename):
    """解析 ShengBTE FORCE_CONSTANTS → (n_atom, n_atom, 3, 3)."""
    with open(filename) as f:
        lines = f.readlines()

    # HiPhive 第一行可能是 "686" 或 "686   686"
    n_atom = int(lines[0].strip().split()[0])
    fc2 = np.zeros((n_atom, n_atom, 3, 3), dtype=np.float64)

    idx = 1
    for i in range(n_atom):
        for j in range(n_atom):
            idx += 1  # 跳过 "i j" 标记行
            for a in range(3):
                vals = list(map(float, lines[idx].split()))
                fc2[i, j, a] = vals
                idx += 1

    return fc2


# ─── fc3 解析（HiPhive FourPhonon 文本格式）──────────────────────────────

def parse_fc3_hiphive(filename):
    """解析 HiPhive FORCE_CONSTANTS_3RD → list of cluster dicts.

    每个 cluster:
      cluster_id       (1 行)
      R_j  x y z       (1 行, 3 浮点, 笛卡尔)
      R_k  x y z       (1 行, 3 浮点, 笛卡尔)
      i j k            (1 行, 3 整数, 原胞原子类型, 1-indexed)
      27 × "a b c val" (27 行, Fortran 序: a 最慢, c 最快)

    返回 list of dict: {'i','j','k', 'R_j', 'R_k', 'fc'(3,3,3)}
    """
    with open(filename) as f:
        lines = f.readlines()

    n_clusters = int(lines[0].strip())
    clusters = []
    idx = 1

    for _ in range(n_clusters):
        # 跳过空白行
        while idx < len(lines) and lines[idx].strip() == '':
            idx += 1
        if idx >= len(lines):
            break

        # 跳过 cluster ID (不参与计算)
        idx += 1

        # R_j
        R_j = np.array(list(map(float, lines[idx].split())), dtype=np.float64)
        idx += 1

        # R_k
        R_k = np.array(list(map(float, lines[idx].split())), dtype=np.float64)
        idx += 1

        # i j k (1-indexed 原胞原子类型)
        i, j, k = map(int, lines[idx].split())
        idx += 1

        # 27 个值，Fortran 序：a 最慢变, b 中间, c 最快
        fc = np.zeros((3, 3, 3), dtype=np.float64)
        for _ in range(27):
            parts = lines[idx].split()
            a, b, c = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1
            fc[a, b, c] = float(parts[3])
            idx += 1

        clusters.append({
            'i': i, 'j': j, 'k': k,
            'R_j': R_j, 'R_k': R_k,
            'fc': fc,
        })

    if len(clusters) != n_clusters:
        raise ValueError("Truncated third-order force-constant file")
    return clusters


# ─── 超胞原子索引映射 ──────────────────────────────────────────────────

def build_supercell_index(prim_file, n_rep=7):
    """构建超胞并返回位置、类型、原胞晶格、原胞原子位置。

    返回:
      sc_positions:  (n_sc, 3) 笛卡尔坐标
      sc_types:      (n_sc,)  1-indexed 原子类型
      prim_cell:     (3, 3)   原胞晶格
      prim_tau:      dict      type(1-indexed) → Cartesian position in primitive cell
      sym_to_type:   dict      chemical symbol → 1-indexed type
    """
    prim = read(prim_file)
    cell = np.array(prim.cell, dtype=np.float64)
    pos = prim.get_positions()      # 笛卡尔
    syms = prim.get_chemical_symbols()
    n_atom_prim = len(syms)

    # 符号 → 1-indexed 类型（保持与 primitive.xyz 一致的顺序）
    unique = list(dict.fromkeys(syms))
    sym_to_type = {s: i + 1 for i, s in enumerate(unique)}
    prim_types = np.array([sym_to_type[s] for s in syms], dtype=int)

    # 原胞内每个类型的笛卡尔位置
    prim_tau = {}
    for s, p in zip(syms, pos):
        prim_tau[sym_to_type[s]] = np.array(p, dtype=np.float64)

    n_sc = n_rep**3 * n_atom_prim
    sc_positions = np.zeros((n_sc, 3), dtype=np.float64)
    sc_types = np.zeros(n_sc, dtype=int)

    idx = 0
    for n1 in range(n_rep):
        for n2 in range(n_rep):
            for n3 in range(n_rep):
                R = n1 * cell[0] + n2 * cell[1] + n3 * cell[2]
                for a in range(n_atom_prim):
                    sc_positions[idx] = R + pos[a]
                    sc_types[idx] = prim_types[a]
                    idx += 1

    return sc_positions, sc_types, cell, prim_tau, sym_to_type


def wrap_into_supercell(R, cell, n_rep=7):
    """将笛卡尔坐标 R 包裹进超胞 [0, n_rep*cell) 内。

    处理浮点误差：R 常是晶格矢量的整数组合，但浮点计算会给出
    -0.00036 或 0.99993 这类接近整数的值。
    """
    cell_inv = np.linalg.inv(cell.T)  # (3,3), 行 = 倒格矢/2π
    frac = cell_inv @ R                # 分数坐标

    # 将近整数分量吸附到精确整数（修复浮点漂移）
    rounded = np.round(frac)
    near_int = np.abs(frac - rounded) < 1e-4
    frac = np.where(near_int, rounded, frac)

    n_cell = np.floor(frac).astype(int)
    frac_local = frac - n_cell          # ∈ [0, 1)
    n_wrapped = n_cell % n_rep

    # 包裹后的绝对笛卡尔位置
    w = np.zeros(3, dtype=np.float64)
    for i in range(3):
        w += (n_wrapped[i] + frac_local[i]) * cell[i]
    return w


def find_sc_index(R_cell_offset, sc_positions, sc_types, target_type,
                  prim_tau, cell, n_rep=7, tol=0.2):
    """根据 (原子类型, 胞偏移) 找到超胞原子索引。

    R_cell_offset: HiPhive 给出的胞偏移矢量（不含原胞内原子位置）
    target_type:   1-indexed 原子类型
    先加 prim_tau[target_type] 得到绝对位置，再匹配。
    """
    # HiPhive 的 R 是胞偏移，不含原胞内原子位置
    R_abs = np.asarray(R_cell_offset, dtype=np.float64) + prim_tau[target_type]

    mask = sc_types == target_type
    candidates = np.where(mask)[0]
    dists = np.linalg.norm(sc_positions[candidates] - R_abs, axis=1)
    best_local = np.argmin(dists)
    if dists[best_local] < tol:
        return candidates[best_local]

    # 包裹后重试
    R_wrapped = wrap_into_supercell(R_abs, cell, n_rep)
    dists2 = np.linalg.norm(sc_positions[candidates] - R_wrapped, axis=1)
    best2 = np.argmin(dists2)
    if dists2[best2] < tol:
        return candidates[best2]

    raise RuntimeError(
        f"无法匹配: type={target_type}, R_offset={R_cell_offset.tolist()}, "
        f"R_abs={R_abs.tolist()}, "
        f"min_dist={dists[best_local]:.6f}, wrapped_min_dist={dists2[best2]:.6f}"
    )


# ─── HDF5 写入 ──────────────────────────────────────────────────────────

def write_fc2_hdf5(fc2, filename, p2s_map=None):
    """写入 fc2.hdf5 (phono3py 4.x 兼容)."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('force_constants', data=fc2, compression='gzip')
        if p2s_map is not None:
            f.create_dataset('p2s_map', data=np.asarray(p2s_map, dtype='int64'))
        f.create_dataset('version', data=np.bytes_('4.3.3'))
    print(f"  → {filename}  (shape={fc2.shape})")


def write_fc3_hdf5(fc3, filename, p2s_map=None):
    """写入 fc3.hdf5 (phono3py 4.x 兼容, compact 格式)."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('fc3', data=fc3, compression='gzip')
        if p2s_map is not None:
            f.create_dataset('p2s_map', data=np.asarray(p2s_map, dtype='int64'))
        f.create_dataset('version', data=np.bytes_('4.3.3'))
    print(f"  → {filename}  (shape={fc3.shape})")


# ─── 主流程 ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <workdir>")
        sys.exit(1)

    workdir = os.path.abspath(sys.argv[1])
    fc2_file = os.path.join(workdir, 'FORCE_CONSTANTS')
    fc3_file = os.path.join(workdir, 'FORCE_CONSTANTS_3RD')
    prim_file = os.path.join(workdir, 'primitive.xyz')

    for p in [fc2_file, fc3_file, prim_file]:
        if not os.path.exists(p):
            print(f"错误: {p} 不存在")
            sys.exit(1)

    for name in ["fc2.hdf5", "fc3.hdf5", "POSCAR"]:
        if os.path.exists(os.path.join(workdir,name)):
            raise FileExistsError(os.path.join(workdir,name))

    # ── 1. 解析 fc2 ──
    print(f"解析 {fc2_file} ...")
    fc2 = parse_fc2_shengBTE(fc2_file)
    n_sc = fc2.shape[0]
    print(f"  fc2: {fc2.shape}")
    print(f"  |fc2| max: {np.max(np.abs(fc2)):.4f} eV/Å²")
    print(f"  diag avg: {np.mean([np.trace(fc2[i,i]) for i in range(n_sc)]):.4f}")

    # ── 2. 解析 fc3 ──
    print(f"\n解析 {fc3_file} ...")
    clusters = parse_fc3_hiphive(fc3_file)
    print(f"  clusters: {len(clusters)}")

    # ── 3. 构建超胞映射 ──
    print(f"\n构建超胞 ({n_sc} 原子) ...")
    prim = read(prim_file)
    n_prim = len(prim)
    sc_positions, sc_types, cell, prim_tau, sym_to_type = build_supercell_index(prim_file)

    # 打印原胞原子类型映射
    print(f"  原胞原子: {prim_tau}")
    # 快速验证：超胞位置是否和 fc2 的对角项一致
    diag_vals = np.array([np.linalg.norm(fc2[i, i]) for i in range(n_sc)])
    print(f"  超胞原子前10个 |fc2[i,i]|: {diag_vals[:10].round(2)}")
    print(f"  说明: 对角块应为大值, 若接近零可能索引顺序有问题")

    # ── 4. 构建 compact fc3 ──
    print(f"\n映射 clusters → compact fc3 (shape={n_prim},{n_sc},{n_sc},3,3,3) ...")

    fc3 = np.zeros((n_prim, n_sc, n_sc, 3, 3, 3), dtype=np.float64)
    n_nonzero = 0
    seen = {}  # (type, rounded_position_tuple) → sc_index

    for ci, cl in enumerate(clusters):
        if (ci + 1) % 1000 == 0:
            print(f"  {ci+1}/{len(clusters)} ... (已映射 {len(seen)} 唯一位点)")

        i_p = cl['i'] - 1  # 0-indexed

        key_j = (cl['j'], tuple(np.round(cl['R_j'], 8)))
        key_k = (cl['k'], tuple(np.round(cl['R_k'], 8)))

        if key_j not in seen:
            seen[key_j] = find_sc_index(cl['R_j'], sc_positions, sc_types,
                                        cl['j'], prim_tau, cell)
        if key_k not in seen:
            seen[key_k] = find_sc_index(cl['R_k'], sc_positions, sc_types,
                                        cl['k'], prim_tau, cell)

        j_sc = seen[key_j]
        k_sc = seen[key_k]

        fc3[i_p, j_sc, k_sc] = cl['fc']
        n_nonzero += 1

    print(f"  完成: {n_nonzero} 个 cluster 已映射, {len(seen)} 唯一位点")
    print(f"  fc3 非零块: {(np.abs(fc3).sum(axis=(3,4,5)) > 1e-8).sum()}/{fc3.shape[0]*fc3.shape[1]*fc3.shape[2]}")

    # ── 5. 写入 HDF5 ──
    print(f"\n写入 HDF5 ...")
    write_fc2_hdf5(fc2, os.path.join(workdir, 'fc2.hdf5'))
    write_fc3_hdf5(fc3, os.path.join(workdir, 'fc3.hdf5'))
    write(os.path.join(workdir, 'POSCAR'), prim, format='vasp',
          direct=True, vasp5=True)
    print("  POSCAR: temperature-matched primitive cell")

    print("Exported second- and third-order HDF5 force constants and POSCAR.")



if __name__ == '__main__':
    main()
