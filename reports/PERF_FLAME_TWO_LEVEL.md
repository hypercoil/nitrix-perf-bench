# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 039cf43ec5f270f26aac62e08fd731eb1b40563e | bench: 8ad140b883d4dd5a200eee20a86ff8303be72885
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T05:11:52.681969+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| flame_two_level | jax-cpu | V=1024,N=60 | `nitrix-jax` | ok | 115.36 ms / 123.64 ms | 1.288 s | 693 MB (rss) | ✓ 0.029×tol | 1.00x vs nitrix-jax |
| flame_two_level | jax-cuda12 | V=1024,N=60 | `nitrix-jax` | skipped | — | — | — | — | — |
| flame_two_level | jax-cpu | V=65536,N=60 | `nitrix-jax` | ok | 9.030 s / 9.256 s | 10.538 s | 840 MB (rss) | ✓ 0.025×tol | 1.00x vs nitrix-jax |
| flame_two_level | jax-cuda12 | V=65536,N=60 | `nitrix-jax` | skipped | — | — | — | — | — |
| flame_two_level | jax-cpu | V=8192,N=60 | `nitrix-jax` | ok | 928.06 ms / 968.14 ms | 2.155 s | 693 MB (rss) | ✓ 0.022×tol | 1.00x vs nitrix-jax |
| flame_two_level | jax-cuda12 | V=8192,N=60 | `nitrix-jax` | skipped | — | — | — | — | — |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

