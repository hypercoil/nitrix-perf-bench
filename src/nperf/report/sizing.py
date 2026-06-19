# -*- coding: utf-8 -*-
"""Param-point sizing + labelling (shared L5 helper).

The single home for two pure functions of a ``param_point`` that the scaling,
economic, and coverage reports all need:

- ``size_elems`` -- the scalar scale axis a curve sorts on (the work / HBM
  driver, per op family);
- ``label`` -- the human row label.

Factored here so the three report layers share ONE definition (was duplicated
via a tool-imports-tool hack). No measurement, no metric arithmetic.
"""
from __future__ import annotations

from typing import Any, Dict, List


def prod(xs: List[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def size_elems(param: Dict[str, Any]) -> int:
    '''The scale axis the curve sorts on.  Image/grid ops: prod(spatial) *
    batch.  Graph ops (no ``shape``, carry ``n``): the node count * degree
    (~nnz, the work/memory driver for the sparse operator).'''
    # volreg: a (T, *spatial) series -- the work / HBM axis is T * voxels.
    if 'shape' in param and 'T' in param:
        return prod(param['shape']) * int(param['T'])
    # bbr: N boundary points -- the cost axis (volume-independent).
    if 'shape' in param and 'N' in param:
        return int(param['N'])
    # permutation_test: n_perm permutations over the spatial volume -- the
    # work / HBM axis is n_perm * voxels (uses 'subj', not 'N', so no clash
    # with the bbr shape+N branch above).
    if 'shape' in param and 'n_perm' in param:
        return prod(param['shape']) * int(param['n_perm'])
    # Real-anatomy points (a fixed-size real image, no 'shape'): the MNI152
    # template's voxel count at the given mm resolution (a stable constant).
    if 'data' in param:
        return {1: 197 * 233 * 189, 2: 99 * 117 * 95}.get(
            int(param.get('resolution', 2)), 99 * 117 * 95)
    if 'shape' in param:
        return prod(param['shape']) * int(param.get('batch', 1) or 1)
    # PCA family (n samples, d features, k components): the (n,d)@(d,k)-class
    # work axis. Checked before the bare ``n`` / ``V`` branches (PCA carries
    # both an ``n`` and a ``d``, which neither of those expects).
    if 'n' in param and 'd' in param:
        return (int(param['n']) * int(param['d'])
                * int(param.get('k', 1) or 1))
    # reml_fit_lowrank: the FaST-LMM low-rank win scales with the observation
    # count ``N`` (q, V fixed), NOT the voxel batch -- so the curve sorts on N.
    # Before the ``V`` branch (this case carries both V and N).
    if 'low_rank' in param and 'N' in param:
        return int(param['N'])
    # Batched LME ops carry the voxel batch ``V`` -- the linear scale axis.
    # Checked before ``n`` because reml_fit carries BOTH ``V`` and a per-group
    # ``n`` (constant across its points); ``V`` is the real scale axis.
    if 'V' in param:
        return int(param['V'])
    if 'n' in param:
        return int(param['n']) * int(param.get('degree', 1) or 1)
    # Paired / conditional family (c variables, d second-block/confounds, obs
    # samples): the input block c*obs is the HBM/scale driver (paired's c*d
    # cross-block and conditional's c^2 cov are sub-dominant at obs > c).
    # Before the bare ``d`` branch (these carry a non-scale ``d``: fixed at a
    # few confounds for conditional, so d^3 would wrongly collapse the tiers).
    if 'c' in param and 'd' in param and 'obs' in param:
        return int(param['c']) * int(param['obs'])
    # Cube-field ops (registration / morphology) carry a side length ``d`` ->
    # a (d, d, d) volume; batched ops (the transform-exps) carry a batch ``b``.
    if 'd' in param:
        return int(param['d']) ** 3
    if 'b' in param:
        return int(param['b'])
    # Connectivity ops (cov / precision family) carry a variable count ``c`` ->
    # the c x c matrix is the HBM driver + the c^3 inverse's axis.
    if 'c' in param:
        return int(param['c']) ** 2
    return 1


def label(param: Dict[str, Any]) -> str:
    # PCA family (n + d + k): n x d, components tag. Before the bare ``n`` /
    # ``V`` branches (PCA carries both n and d).
    if 'shape' not in param and 'n' in param and 'd' in param:
        return f'{param["n"]}x{param["d"]} k{param.get("k")}'
    # reml_fit_lowrank: N is the scale axis (q, V fixed). Before the V branch.
    if 'shape' not in param and 'low_rank' in param and 'N' in param:
        return f'N={param["N"]} q{param.get("q")}'
    # Batched LME ops (V the scale axis); before ``n`` (reml carries both).
    if 'shape' not in param and 'V' in param:
        return f'V={param["V"]}'
    # Graph ops are keyed by node count (n) + format / k; image ops by shape.
    if 'shape' not in param and 'n' in param:
        lbl = f'n={param["n"]}'
        if param.get('fmt'):
            lbl += f' {param["fmt"]}'
        if param.get('k') not in (None, 8):
            lbl += f' k{param["k"]}'
        return lbl
    # Paired / conditional family (c + d + obs); before the bare ``d`` branch.
    if 'shape' not in param and 'c' in param and 'd' in param \
            and 'obs' in param:
        return f'c{param["c"]} d{param["d"]} obs{param["obs"]}'
    if 'shape' not in param and 'd' in param:
        return f'{param["d"]}^3'
    if 'shape' not in param and 'b' in param:
        return f'b={param["b"]}'
    if 'shape' not in param and 'c' in param:
        return f'c={param["c"]}'
    # Real-anatomy points (no 'shape'): the dataset + mm resolution.
    if 'data' in param:
        return f'{param["data"]} {param.get("resolution", 2)}mm'
    # permutation_test: tag the permutation count (the scale axis).
    if 'shape' in param and 'n_perm' in param:
        return f'p{param["n_perm"]} {"x".join(str(s) for s in param["shape"])}'
    # volreg: a (T, *spatial) series -- tag the batch (the scale axis).
    if 'shape' in param and 'T' in param:
        return f'T{param["T"]} {"x".join(str(s) for s in param["shape"])}'
    # bbr: N boundary points -- the scale axis.
    if 'shape' in param and 'N' in param:
        return f'N{param["N"]} {"x".join(str(s) for s in param["shape"])}'
    shp = 'x'.join(str(s) for s in param.get('shape', []))
    b = param.get('batch')
    base = f'{b}*{shp}' if b else shp
    # Distinguish same-shape points by their structuring element / dtype, so a
    # box and a disk/ball at one grid size are *separate* rows -- not merged
    # (the morphology family has several SEs per shape; collapsing them hid the
    # fast box behind the slow ball).
    tags = []
    if param.get('se'):
        k = param.get('size', param.get('radius'))
        tags.append(f'{param["se"]}{k}' if k is not None else str(param['se']))
    if param.get('dtype') and param['dtype'] != 'float32':
        tags.append(str(param['dtype']))
    # registration cross-grid / anisotropy variants sit at the same fixed
    # shape as a shared-grid / isotropic point -- tag them so they are separate
    # rows (else _collect merges the world row onto the index row).
    if param.get('space') == 'world':
        tags.append('world')
    if param.get('spacing'):
        tags.append('aniso' + 'x'.join(str(s) for s in param['spacing']))
    return base + (' ' + ','.join(tags) if tags else '')
