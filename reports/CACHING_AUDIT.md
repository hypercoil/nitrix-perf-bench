# Caching audit — can the bench silently fall back to an unreported cache?

A one-time, recorded audit (2026-06-09, L4, jax 0.10.0) so the question does not
have to be re-litigated. **The concern:** within an attempt the timing loop
(`core/timer.py::bench_call`) calls the function `warmup + repeats` times with
the *same* on-device arguments, and across runs the inputs are rebuilt from a
*fixed seed* (so they are bit-identical run-to-run). If JAX or a baseline
returned a cached result for identical inputs — instead of recomputing — the
reported steady time would be a no-op, not the op. This doc presents the
mechanism and a direct experiment; it states the numbers, not a verdict.

## How the per-attempt lifecycle is structured

(`measure.py::measure_attempt` + `core/timer.py` + `/scratch/nperf/env.sh`.)

1. `jax.clear_caches()` runs **per attempt**, and the persistent compile cache
   is **disabled** (`env.sh` unsets `JAX_COMPILATION_CACHE_DIR`; provenance
   hard-codes `compile_cache: "disabled"`). So no compiled executable survives
   into an attempt.
2. Each attempt is a **fresh OS subprocess** (`worker.py`), so no in-process
   JAX/XLA state survives across attempts or across runs.
3. Inputs are passed as **`jax.jit` arguments** (abstract tracers), never closed
   over as compile-time constants, and are rebuilt fresh per attempt.
4. The first call is timed separately as `compile_time` (trace + compile +
   execute); the post-warm-up calls are `steady_time`.

## Experiment

`jax.jit(lambda mv, fx: rigid_register(mv, fx, spec=L2x20).params)` at 96³, on
the GPU. Three measurements:

| measurement | value | what it exercises |
|---|---|---|
| cold compile (after `clear_caches`) | **12 155.7 ms** | trace + compile + execute |
| warm steady, input **A** (seed 0) | **56.88 ms** | warm executable, A |
| warm steady, input **B** ≠ A (seed 7), same shape | **57.67 ms** | warm executable, B |
| `compile / steady` | **214×** | — |
| `max|params(A) − params(B)|` | **0.0123** (≠ 0) | output vs input dependence |

Cross-check from the baselines, across the size sweep (`reports/REGISTRATION_-
SCALING.md` + the cross-tool sweep) — wall-clock vs voxel count:

| op (rigid, warm min) | 48³ | 96³ | 128³ | scaling |
|---|---|---|---|---|
| nitrix-jax — **L4 GPU** | 15 ms | 93 ms | 235 ms | grows ~∝N above the knee |
| nitrix-jax — host CPU | 215 ms | 2.02 s | 5.4 s | ~∝N |
| ANTs — host CPU | 39 ms | 132 ms | 274 ms | sub-linear (iteration-bounded) |
| dipy — host CPU | 1.39 s | 15.6 s | 35.2 s | super-linear |

(The domain tools — ANTs, dipy, SimpleITK — are **CPU-only** (ITK C++ /
numpy-scipy-cython); only `nitrix-jax` on `jax-cuda12` uses the L4 GPU, so "L4"
here means the GPU and the CPU rows ran on the box's host CPU. ANTs' *first*
call is ~17 s vs ~39 ms warm — a one-time ITK initialisation, captured as
`compile_time`; its warm calls scale with size.)

## What each number isolates (reader's inference)

- **cold 12 156 ms vs warm 56.9 ms (214×):** the first call after
  `clear_caches()` pays a full trace+compile; the warm calls do not. (JAX's jit
  cache is keyed on shape/dtype/static-args, not on input *values*, and it is
  cleared per attempt + the persistent cache is off.)
- **warm A 56.88 ms ≈ warm B≠A 57.67 ms:** a *different* input at the same shape
  costs the same as repeating the *same* input. (A value-keyed result cache
  would make the repeated-A case ~0 ms.)
- **`params(A) ≠ params(B)` by 0.012:** the output depends on the inputs, so the
  computation cannot have been constant-folded / DCE'd to an input-independent
  no-op at compile time. (Inputs enter as abstract jit arguments, not
  constants.)
- **Every implementation's wall-clock changes with input *size*:** a result that
  was served from a cache would be size-flat.

## Residual subtlety (left for the reader to weigh)

The inputs *are* deterministic across runs (rebuilt from a fixed seed, so
bit-identical). The measurements above bear on whether that determinism enables
any value-keyed caching. A belt-and-suspenders alternative would feed a
different input *value* per repeat (same shape, so the same warm executable is
still hit) — at the cost of adding data-dependent timing variance. The numbers
here are what the current design produces; the conclusion is the reader's.

## Reproduce

The experiment script is inline in the session log; re-run by jit-wrapping any
recipe, timing it after `jax.clear_caches()` (cold) and over a warm loop with
two different same-shape inputs, and comparing the two warm minima.
