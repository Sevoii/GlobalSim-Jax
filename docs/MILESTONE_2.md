# Milestone 2: accepted restricted subset

Reference revision: `58da7bc4693d9aa8f6340e348cba1c5b5fb5a856`. This is a narrow M2 acceptance, not complete simulator parity.

## Supported APIs

Stat conversion helpers are `adjust_overcap`, `team_rank_bonus`, `initial_base_wisdom`, and `ranked_base_stat`. `adjust_overcap` requires JAX x64 for its source-backed mixed-precision calculation and never changes global JAX configuration.

Standalone geometry supports only `CourseGeometry.load(10101, 1200.)`, `position`, and `advance` on the bundled captured 10101/1200m asset. It validates the table identity/schema and requires x64. Other courses and overrun positions reject. Geometry is accepted independently and is not integrated into the loop.

The flat/no-field runner boundary is:

`restricted.initialize(params, speed, stamina, power, guts, strategy, aptitude, ground, delay, distance)`

followed by:

`run_checked(params, state, config=RestrictedConfig(distance, dt, max_ticks), speed_stat, power, guts, strategy, aptitude, ground)`.

State and trace fields are `time`, `position`, `speed`, `hp`, `delay`, `finished`, `timed_out`, `finish_time`, `ticks`, `phase`, and `start_dash`. There is no initial-lane input, target-speed output, or active-mask output. The checked boundary requires nonempty equal `[batch, runner]` arrays, capacity at most 32, finite/range-valid inputs, supported state dtypes/sentinels, positive ground/distance, nonnegative finite dt, integer strategies 1–5, and unity-only speed plus acceleration-distance aptitude. `step` and `run` are prevalidated JIT kernels, not validation APIs.

The loop preserves the selected source-derived gate, start-dash minimum/cap/full-dt drain, speed/deceleration branch, pre-movement phase observation, HP branch, adjacent-tick finish interpolation, sticky wrapper timeout, and terminal freezing. It treats exact goal as a field scheduling stop. No runtime RNG is consumed after caller-supplied state.

## Evidence

`tools/restricted_oracle.rs` is a destination-only extracted adapter that imports pinned, read-only Rust source. It freezes six eight-tick cases: delayed non-unit-ground gate, phase crossing, finish overshoot, exact-goal field stop, dash exit, and HP-zero deceleration. The committed fixture has nine channels: time, position, speed, HP, delay, phase, dash, finished, and finish time. Tests assert every channel eagerly and under jitted `lax.scan`; a divergent `[2,2]` batch/runner regression compares every field against independent scalar eager and JIT runs.

The oracle generator itself uses local source paths and does not enforce the revision at runtime. Regenerate only after checking the pinned revision and source/data/oracle hashes in [ORACLE_PROVENANCE.json](ORACLE_PROVENANCE.json); otherwise retain the committed frozen fixture.

Final coordinator validation: `.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider` => 17 passed in 27.86s on Windows CPU, Python 3.13.7, JAX/JAXlib 0.12.0.1. `restricted-demo --ticks 1200` ended at `final_position=1200.084961`.

## Unsupported / non-claims

No full race API, field ordering, ranks/ties, lane behavior, blocking, pacing, rushing, competition, spurt, skills, random retail initialization, geometry integration, replay/result parity, GPU validation, or performance-parity claim. The existing benchmark is synchronized RNG primitive-only, not a restricted-loop or full-race benchmark.
