# Registration: re-bench + brain-scale crossover / bias diagnosis

Post the nitrix registration refactor (`lax.scan` loop-roll `ddc2e10`, `Metric`
ADT, `TransformModel`, assembled small-P normal equations). Measured on the L4
(`jax-cuda12` vs `jax-cpu`), nitrix-only, one fixed config (`levels=2,
iterations=20`) swept over volume `24³ → 192³` so the curve isolates the
voxel-count axis `N`. Steady = warm min.

## 1. The refactor closed the cold-compile pathology (the re-bench)

The "registration slow on GPU" report was a **Python-unrolled** optimiser loop,
so the XLA cold compile scaled with `levels × iterations`. The refactor rolled
it to `lax.scan`. Measured effect:

| recipe | config | compile **pre-roll** | compile **post-roll** |
|---|---|---|---|
| rigid | L1×10 → L3×30 | 16.6 s → **141 s** | 4.3 s → **9.1 s** |
| affine | L1×10 → L3×30 | 24 s → **211 s** | 4.0 s → **11.0 s** |
| demons | L2×20 → L2×40 | 47 s → **98 s** | **6.82 s → 6.82 s** (identical) |

- Compile collapsed **10–20×** and is now **flat in iterations** — demons L2×20
  and L2×40 compile to the *same* number; the residual ~2× rigid spread is the
  *levels* (3 pyramid graphs), not the iteration count.
- The **affine L3×30 CPU compile** that previously failed XLA outright
  (`INTERNAL: failed to materialize symbols` — the unrolled graph too large) now
  compiles (~6.6 s). My earlier "affine CPU compile fails" finding was a genuine
  pre-roll artifact, **now resolved** (nitrix recorded the same in `92824d5`).
- Steady also improved (assembled small-P normal equations replaced the
  matrix-free autodiff-Jacobian CG): affine GPU L3×30 243 → 52 ms.

## 2. GPU-vs-CPU steady speedup across scale (the crossover)

`speedup = CPU_min / GPU_min`, at `levels=2, iters=20`:

| size | voxels | rigid | affine | demons |
|---|---|---|---|---|
| 24³ | 13.8 K | 4× | 2× | 10× |
| 32³ | 32.8 K | 7× | 5× | 29× |
| 48³ | 110 K | 18× | 14× | 43× |
| 96³ | 884 K | 22× | 32× | 42× |
| 128³ | 2.1 M | 23× | 33× | 31× |
| 160³ | 4.1 M | 28× | 37× | 28× |
| 192³ | 7.1 M | 25× | 34× | — |

**There is no small-size regime where CPU wins** — GPU leads at *every* measured
size (≥2×, even at 24³). The feared GPU-launch-overhead crossover does not occur
for these recipes: one call already does `levels × iters × (several warps)` of
work, enough to amortise launch.

**The structure is a GPU "knee," not a crossover.** GPU steady is
**overhead-bound** (≈flat: rigid 9→11→15 ms, affine 19→20→25 ms, demons 6→6→10
ms) up to ~48³, then turns **compute-bound** (∝ N: rigid 93→235→415→811 ms over
96³→192³, an 8.7× rise for 8× the voxels). CPU is ∝ N throughout. So the speedup
**climbs steeply below the knee** (CPU grows, GPU flat) and **plateaus above
it**: ~25× rigid, ~35× affine.

**Demons is the exception at large N:** its speedup peaks ~43× (48–96³) then
*declines* to ~28× (160³), because demons GPU steady grows **super-linearly**
(96³→160³: 80→648 ms = 8.1× for 4.6× the voxels) — the d-component SVF field +
scaling-squaring + Gaussian smooths are **memory-bandwidth-bound**, so demons
loses GPU efficiency fastest at scale.

## 3. The bias the old bench carried

The prior suite measured registration **only at 48³**. That single point sat
**exactly at the knee** *and* in the **now-fixed compile-dominated regime**, so
it was doubly misleading:

- It read a **transient compile pathology** (145 s) that the loop-roll has since
  erased — and that was never representative anyway: at brain scale the one-time
  compile (≤ ~30 s) amortises to nothing against per-call steady (192³ rigid:
  0.81 s/call) over a real many-volume run.
- It caught the GPU steady advantage **mid-climb** (14–43×), **understating** the
  brain-scale plateau (25–37×) the curve only reaches past the knee.

A single toy-size measurement of an iterative recipe biases *both* axes —
overstating compile, understating the asymptotic GPU win. The fix is the curve,
which is why the recipes now carry a brain-scale `large_param_points` tier.

## 4. HBM is *not* cleanly measurable here (a metric-bias caveat)

The size-tier discipline normally projects an OOM volume from a per-element HBM
rate. **That projection is not trustworthy for these recipes**, because cold
`peak_hbm` (compile + run share one process; the high-water spans both) conflates
three things: a fixed ~150 MB CUDA-context floor, the op working set, and
**erratic XLA autotune scratch**. The tell: the GPU HBM curve is non-monotonic,
with an **identical ~8.7 GB high-water at 128³ across all three recipes** (then
*dropping* at 160³/192³) — a shared autotuning allocation, not any op's working
set (per-voxel HBM swings 11 → 0.27 KB across the sweep).

What survives: the **ordering** at the clean small sizes (48–96³), where demons
(339 MB / 3.8 GB) is ~1.7–3× heavier per voxel than rigid/affine (203 MB / 1.3
GB) — demons is the HBM-heaviest recipe and would reach the ceiling first. But a
precise OOM voxel-count is *not* projectable from this metric as measured, and
**none of the three OOM'd in range** (rigid/affine to 192³, demons to 160³, on
the 23 GB L4). Honest takeaway: report the ordering, not an OOM size; a clean
HBM-scaling number would need a warm-only memory probe that excludes compile
scratch.

## 5. Follow-ups surfaced by the op-matrix re-inventory

Registration-relevant ops now public in `nitrix/docs/op_matrix.json` but not yet
benched: `register.{bending_energy, gradient_smoothness, jacobian_folding_-
penalty}` (regularisers), `linalg.implicit_minimize` (the new non-SSD IFT
layer), `metrics.joint_histogram`, and the transform-exps `geometry.{rigid_exp,
affine_exp, rigid_log}`.
