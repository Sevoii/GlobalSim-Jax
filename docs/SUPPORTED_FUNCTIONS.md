# Supported-function inventory

Reference: GlobalSim Rust revision `58da7bc4693d9aa8f6340e348cba1c5b5fb5a856`, `src/retail_random.rs` and `src/retail_physics.rs`. This derived package remains AGPL-3.0-only; `LICENSE` and `NOTICE` are retained.

## M3b deterministic primitive addition (partial)

`lane.comparison_sort_indices` ports the loaded unstable Rust introsort for a
padded row of at most 32 signed int32 keys. Its checked wrapper rejects rows
larger than 32 and non-integral or non-int32-representable keys. It is not a
generic stable sorting API. `lane.resolve_overlaps` ports the separate manager
pass: it sorts by truncating f32 lane keys, clears only the first active force
flag, applies strict overlap tests sequentially, and retains chained pushes
without rail clamping. `resolve_overlaps_checked` is a host one-row boundary.

Frozen evidence comes from `tools/lane_oracle.rs`, which imports pinned
`retail_sort.rs` and uses the source overlap pass, plus the existing copied
sort vectors. Tests cover active counts 0, 1, 2, 3, 16, 17, 18, and 32,
special duplicate-three permutations, exact strict thresholds, force-flag
preservation, chained pushes, eager/JIT execution, and independent batches.
`MoveState`, visibility/side-room, overtaking, supplied-target moving-lane
integration, automatic targeting, pacing, and all M3c systems remain
unsupported. This is not completion of M3b or M3.

Requires an application to set `jax_enable_x64=True` before use. Importing `globalsim_jax` does not change global JAX configuration. Float32 stores and documented float64 intermediates mirror the reference's mixed-precision port in eager CPU checks; results on another backend require validation.

Supported and fixture-tested:

- Explicit pytree `RetailRng` initialization, `next_int`, `next_double`, `integer` (ordinary, zero-width, and wide ranges), and `real`.
- `real_distance`, explicitly **experimental**: it is an isolated Rust hypothesis and is not evidence of established runtime draw ordering. Both float wrappers preserve Rust's sequential clamp comparisons for reversed/equal bounds.
- Stateless `phase_boundaries`, `target_speed_indices`, `advance_start_delay`, and `finish_time` (only fixture rows with `adjacent_ticks: true`).
- `bounded_stat`; parameter-driven course/base/minimum/spurt/target speeds, blocking cap, initial HP, HP state rate/drain, and acceleration (including ordered `acceleration_from_base`) through `Parameters.default()`.

Fixture policy: `tests/fixtures/retail_rng_dotnet.json`, `retail_physics_port.json`, and `retail_finish_crossings.json` are byte copies of the pinned Rust fixture files, retaining their source/provenance fields. RNG and physics use exact integer/f64 or float32-bit assertions. Finish crossings use `atol=1e-5` because the exported values are float results; the two explicitly excluded thinned/non-adjacent intervals are not used.

M1 alone did not include initialization, geometry or field behavior; accepted M2/M3a additions are listed below. Automatic targeting/pacing, skills, replay, full-race parity and GPU validation remain unsupported. Invalid scalar integer bounds raise `ValueError`; parameter validation beyond the captured default data and arbitrary dynamic invalid inputs is not a public compatibility guarantee yet.

CPU JIT HP-drain exactness: the implementation preserves the reference float32 `(gap * gap) / denominator` division, with optimization barriers around its float32 numerator, denominator, and quotient store. This prevents CPU XLA reciprocal-multiply contraction; the copied fixture passes exact eager, standalone-JIT, and composed-JIT assertions on JAX 0.12.0.1 CPU. Other backends still require independent validation.

## Milestone 2 restricted additions

The source-backed stat helpers `adjust_overcap`, `team_rank_bonus`, `initial_base_wisdom`, and `ranked_base_stat` are supported for their tested conversion order. `adjust_overcap` requests float64 and therefore calls `require_x64()` without changing JAX global configuration.

`CourseGeometry.load(10101, 1200.)`, `position`, and `advance` are supported only for the bundled captured 10101/1200m asset. They validate identity/table schema/finite data and require x64. Other course IDs and overrun positions reject. Geometry is separate from the flat loop.

`restricted.initialize`, `RestrictedConfig`, and `run_checked` support a fixed shape `[batch, runner]`, capacity 32, flat/no-field profile. The checked entry validates shapes, finite/range/dtype state, finish-time sentinel (`NaN` precisely for unfinished lanes), positive ground/distance, nonnegative finite dt, strategies 1â€“5, and unity-only speed plus acceleration-distance aptitude factors. `restricted.step`/`run` are deliberately prevalidated JIT kernels. The profile contains no skills, RNG initialization, geometry integration, slopes, field reads, ranks, lanes, blocking, pacing, rushing, spurt, or race-result logic.

Independent evidence: `tools/restricted_oracle.rs` imports pinned Rust `json.rs` and `retail_physics.rs` read-only and freezes six 8-tick source-extracted cases: delayed non-unit-ground gate, phase crossing, finish overshoot, exact-goal field stop, dash exit, and HP-zero deceleration. `tests/fixtures/restricted_trace.json` stores nine channels as f32 bits/flags: time, position, speed, HP, delay, phase, dash, finished, finish time. Tests assert every channel eager and jitted `lax.scan`, including divergent `[2,2]` batch/runner lanes against independent scalar scans. M2 timeout is only a wrapper safety cap; it is sticky and terminal rows are frozen padding.

Pinned source, oracle, captured-data, and frozen-fixture hashes are in [ORACLE_PROVENANCE.json](ORACLE_PROVENANCE.json). Coordinator final validation for this inventory was 17 tests passed in 27.86s on Windows CPU / Python 3.13.7 / JAX 0.12.0.1.

The oracle executable does not enforce its source revision at runtime. Treat the manifest revision and hashes as a required precondition before regenerating committed expected data.

## Milestone 3a fixed-lane additions

`field.ranks_checked`, `field.blocking_snapshot_checked`, `initialize_field`, and `field.run_checked` support the bounded fixed-lane M3a profile. Front blocker IDs/gaps and side flags are sampled before movement; field movement is runner-index ordered and reads a blocker speed live for its cap. No-front is the checked sentinel `(-1, NaN)`. Fresh state starts with minimum 3; synthetic/nonfresh attachment requires an explicit finite nonnegative prior minimum. The bounded wrapper lets a crossing pass complete, then freezes every field-state leaf before any post-goal tick; it is not retail post-goal parity.

Evidence includes three independent Rust field traces (8 ticks Ã— 3 runners Ã— 19 channels), eager/JIT scans, two-batch scalar independence, an ordered live-speed counterexample, rank/boundary query fixtures, and a custom front-lane f32-store vector. M3b/M3c and whole M3 remain unsupported.


## Milestone 3b accepted standalone primitives

lane.comparison_sort_indices and lane.resolve_overlaps provide capacity-32 padded prevalidated kernels; checked scalar wrappers validate input domains and return active results. Exact source sort permutations, chained overlap correction and force flags are preserved. Inactive tails are covered by regression tests.

lane_move.step and step_checked provide supplied-target lateral movement with explicit MoveState, MoveContext and MoveParameters pytrees. Modifier rows have fixed capacity 16. step_checked accepts finite float32 leaves with canonicalized integer/boolean context, requires positive max_lane and nonnegative dt, rejects int32 overflow and direction flags outside 0/1. Callers enable JAX x64 explicitly. Full-state Rust cases and the 700-vector legacy fixture pass eager and JIT comparisons. The approximately threshold handles source subnormals under CPU XLA.

These primitives are accepted separately from M3a field integration. Side-room/visibility and overtaking helpers, automatic targets/pacing, moving-lane field scheduling, complete M3b/M3, full-race and GPU claims remain unsupported. Final validation: 43 tests passed; scope and provenance are in MILESTONE_3B.md and REVIEW_M3B.md.


## M3b accepted standalone query helpers (2026-11-17)

lane_queries.side_room/is_visible/is_target and their checked wrappers are accepted within the documented real finite f32 storage domain. Side-room uses capacity-32 supplied peer order and an explicit active count, retains the last qualifying inner blocker, and represents absence as NaN/-1. Visibility preserves f32 ratio/f64 width/f32 store; overtaking narrows its f64 inputs before comparisons. Checked APIs require x64, reject overflow/nonzero-subnormal operands and side-room subnormal differences/spaces, reject invalid shapes/types/flags, and retain f64 underflow-to-zero. Low-level callers prevalidate these restrictions. Rust fixtures cover20 side/12 visibility/26 overtaking cases plus2 unsupported diagnostics. All valid rows pass checked/eager/JIT paths; coordinator full suite51passed79.46s. See REVIEW_M3B_QUERIES.md final acceptance. Field integration remains a separate active milestone.


## M3b supplied-target moving-field integration

moving_field.initialize, step, run and run_checked integrate supplied persistent targets with the accepted restricted flat field. Constant max_lane/first_move_point; empty lane modifiers; no automatic target/pacing/skill policy or variable geometry. Cached entry side permissions/rank/front ID-gap, ordered live blocker speeds, full lateral dt on partial gates, post-longitudinal overlap and whole-state terminal padding follow the documented schedule. FieldState.lane is authoritative. Blocking timers remain f64, with source mixed-precision lateral expressions. Checked entry points require explicit x64, validate finite/range/dtype state and source f32 storage, and reject nonzero f32 subnormal operands/side differences; f64 blocked timers are never narrowed. moved is current-pass wrapper telemetry rather than unported skill-receiver sticky state.

Nine independent Rust cases cover all23state fields and8observation fields, eager/JIT scans, batch independence, whole-state terminal byte freezing, three source-order counterexamples and ordered live blocker speed. Details: MILESTONE_3B_INTEGRATION.md and tools/MOVING_FIELD_ORACLE.md. CLI: python -m globalsim_jax.cli moving-field-demo --ticks 8. Luna accepted this bounded profile; coordinator full-suite result is recorded in STATUS/HANDOFF. No full M3b/M3, complete race, post-goal retail scheduling or GPU claim.

## TargetState cache collection (review pending)

`lane_target.collect`, `collect_roster`, `collect_checked`, and
`collect_roster_checked` expose only the deterministic collection snapshot:
owner-excluded stable peer rows, visible mask, side-room state, front cache
fields, and force-in. The unsorted standalone path supports at most 32 peers; roster collection
supports at most 32 total runners (31 peers after removing its owner). The
sorted nearby optimization is omitted. Checked entry points require x64 and reject invalid shape/type/range,
nonfinite values, float32 overflow, and nonzero float32 subnormal operands or
derived query quantities. Visibility-ratio validation applies only to rows that
pass the source valid-cone and widest-lane gates, because only those rows reach
Rust's division; this is a conservative checked-domain guard, not a claimed
source-output mismatch. `TargetCache` is not a mutable `TargetState`: target
retention, cooldown and other lifecycle fields, target selection, PassingSearch,
RNG, pacing and field integration remain unsupported. Luna independently accepted
the bounded snapshot with 4 focused tests passing; root verified exact source
extraction, byte-identical oracle regeneration and all 40 provenance hashes.
The coordinator full-suite gate passed: 62 tests in 363.84s on Windows CPU. See REVIEW_M3B_TARGET_CACHE.md
and tools/TARGET_CACHE_ORACLE.md. This is not completion of M3b/M3.

## Accepted TargetState persistent updates (2026-11-17)

`initialize_target_state`, `update_targets`, and `update_targets_checked` cover
only default-feature `TargetState::update_targets` atop an unsorted,
owner-excluded cache snapshot. Target rows are capacity 32 with `-1` inactive
padding; a stale nonnegative int32 target is valid and source-dropped before
any gather. `previous_can_in` is `-1`/`0`/`1` for None/false/true, while a NaN
f32 bit pattern represents absent `extra_outward`. The fixed snapshot has no
`sorted_owner` representation: this explicitly excludes the source sorted
nearby-roster optimization.

The checked boundary requires x64, exact leaf dtypes/shapes/sentinels and
canonical zero inactive horse padding/false visibility/`-1` target tails. It
rejects nonfinite/overflow inputs and nonzero f32 subnormal scalar, pairwise
blocker/cone and source-evaluated predicate-ratio domains before dispatch; the
minimum-subnormal blocker divergence is therefore an explicit unsupported
diagnostic. `dt` and supplied cooldown are nonnegative. The source-built oracle
has eight supported four-update traces plus that diagnostic; tests bit-compare
all state/cache leaves eager, jitted scan, checked and selected vmap paths.
Luna accepted this bounded transition after independently passing17 focused tests.
Root full-suite gate:75passed342.51s; byte-identical source oracle rebuild and
all45provenance hashes verified. This does not include final target selection,
RNG, pacing, field integration, experimental gap refresh, full simulation or GPU
support. See REVIEW_M3B_TARGET_UPDATE.md and tools/TARGET_UPDATE_ORACLE.md.
