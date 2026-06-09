# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 38926ea2bfa0195dd61917fdc124b0a009e99382 | bench: de1e1e3b2f4c26858e872fb2ad076dd7c2ff01d6
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-09T08:39:12.615599+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cpu** — None (None) | jax None | precision None | x64 None | isolation in_process | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `ants.registration` | ok | 545.98 ms / 585.49 ms | 2.766 s | 702 MB (rss) | n/a (no oracle) | — |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax` | ok | 5.28 ms / 5.37 ms | 21.833 s | 373.97 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | ok | 535.45 ms / 546.46 ms | 2.519 s | 702 MB (rss) | n/a (no oracle) | — |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 8.97 ms / 9.63 ms | 47.269 s | 373.97 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `ants.registration` | ok | 539.58 ms / 551.20 ms | 2.363 s | 702 MB (rss) | n/a (no oracle) | — |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax` | ok | 16.45 ms / 17.11 ms | 98.059 s | 373.97 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- **`mem` (peak_hbm) is a process-wide high-water mark** for any in-process rows. XLA's `peak_bytes_in_use` does not reset between attempts, so it only ever rises: once one attempt allocates a large buffer, later rows inherit that floor. Read the *jumps*, not the absolute per-row value. Re-run without `--in-process` for per-attempt isolation (annex §B).

