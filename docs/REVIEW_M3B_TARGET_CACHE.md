# M3b automatic target cache: preliminary source review

Reviewer: GPT-6-Luna  
Status: accepted bounded standalone snapshot implementation after final re-review.  
Reference: read-only `C:/Users/a/RustroverProjects/GlobalSim` at `58da7bc4693d9aa8f6340e348cba1c5b5fb5a856`.

## Bounded unit

This review covers only `TargetState` collection: the `collect_cached` reference scan and `collect_roster` owner-excluded roster mapping, including `visible`, `side`, `front_cache`, and `force_in`. The sorted nearby scan is an optional optimization and may be omitted if explicitly documented. Collection is not target generation: `update_targets`, `PassingSearch`, target lifecycle, shared RNG consumption, pacing/pacemaker selection, and field integration remain unsupported. Evidence for this source subset must not be described as full-engine parity.

## Source-backed acceptance criteria

- **Preserve roster identity and order.** `collect_roster` forms `cached` from all roster rows before owner followed by all rows after owner. Every returned visible peer index and `SideRoom.inner_blocker_index` is in this owner-excluded cached index space. Keep roster rows in that stable order regardless of finish status; do not compact active peers or accidentally return original roster indices.
- **Match the scalar scan's first-front choice.** `collect_cached` visits peers in cached order and updates the front only for a strictly nearer qualifying row. Equal-distance ties therefore keep the first cached peer. Candidate conditions are `position_gap > 0`, strictly inside `front_distance`, and lateral distance `<= front_lane_limit`. The boundary helper has mixed precision: compute `narrowing = (gap / front_distance) * 0.6f32`, then widen `narrowing` and `front_lane` for `(1.0 - narrowing) * front_lane`, and narrow the product once to f32. Preserve every cast and comparison boundary.
- **Match side and inner-blocker overwrite behavior.** Initial permissions are `owner.lane > 0.0001` and `owner.lane < max_lane`; spaces start at owner lane and `max_lane - owner lane`. A peer is in side range when `!(abs(position_gap) >= side_distance)` (including the source's NaN comparison behavior if such inputs are admitted). Update nearest inward/outward spaces and strict lane thresholds in source order. Each qualifying inward blocker overwrites the prior lane/index, so the *last* such peer in cached order wins. Outward blockers only clear `can_out`. Preserve strict versus inclusive comparisons exactly.
- **Match visible order and predicate.** Append visible cached indices in increasing cached order when `is_visible(position_gap, lane_gap, vision)` succeeds; preserve its forward-only gap and cone-edge rules from the source helper. Do not reorder visibility results or encode them in an order that changes public output.
- **Match force-in.** `force_in` is true iff empty inward movement is enabled, `phase < 1`, owner lane is at/above `force_in_threshold`, and at least one peer has `abs(position_gap) < side_distance`, positive inward gap, and inward gap strictly below `side_lane`. This is an existential flag, distinct from the last-overwritten side blocker.
- **Preserve non-collection state.** Collection refreshes cached peers and the four derived outputs; it does not reset targets, cooldown, force-check, extra outward movement, previous-can-in, skip-passing, or unrelated lifecycle state. Verify repeated collection and owner changes leave these fields unchanged while replacing only collection outputs.
- **Checked numeric domain.** Public checked entry points should reject nonfinite values and unsupported nonzero f32 subnormals (including values that become subnormal after narrowing), while preserving valid f64 inputs that narrow to signed zero if the established checked-domain policy allows them. Validate roster/capacity/owner bounds before gather/index operations. Fixed-capacity padding must never leak padded rows into `visible`, side calculations, front selection, or `force_in`; test maximum supported occupancy and rejection beyond it.
- **Transformation evidence.** Compare all collection outputs and unchanged state fields against the independent source-produced fixture, including eager and `jit` paths. Also show `vmap`/batch isolation (one row's peers and owner indices cannot affect another), active masks, owner-excluded index mapping, stable visible order, equal-front ties, last-inner overwrite, strict boundary cases, and padding/capacity behavior. Root owns independent oracle audit; do not present generator self-consistency as independent provenance.

## Current disposition

The Rust source supports this deterministic slice and the inspected deterministic kernel matches the source scan. Final independent focused command `.\\.venv\\Scripts\\python.exe -m pytest tests/test_lane_target.py -q -p no:cacheprovider` passed **4 tests in 11.37s**. Tests compare every cache field, float bits, owner/peer storage, count, padding and visible slots across the 74 source rows in eager/JIT paths; compare all valid direct checked rows to the fixture; cover four owner-removal roster cases including the 31-peer maximum, checked roster results, batched isolation, and poisoned inactive padding. Checked inputs exercise invalid shape/capacity/dtypes, complex and post-cast overflow, source-derived overflow/subnormals, and x64 policy. No cache-output mismatch or remaining finding was identified in this bounded review.

Coordinator final gate (reported by root; not independently run by Luna): `.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider` => **62 passed in 363.84s**. Root reports rebuilding the oracle as **raw-byte identical** for all 74 peer and 4 roster cases, verifying every extracted original collection source span, all 40 provenance manifest hashes, and an unchanged pinned Rust checkout at `58da7bc4693d9aa8f6340e348cba1c5b5fb5a856`.

### Previous findings and resolution

1. **Resolved — checked direct collection oracle assertion.** Every valid row from the 74-case fixture now runs through `collect_checked` and compares all exposed fields and storage invariants, including empty/full peer counts. The four roster cases also compare `collect_roster_checked` results.

2. **Resolved — checked numeric-domain validation.** `_canonical_collection_inputs` now evaluates the source's visibility gates and validates only ratios Rust actually computes, rejecting nonfinite and nonzero-subnormal ratios before dispatch. Tests cover nonnumeric/complex inputs, f32 cast overflow, overflowing coordinate differences, subnormal coordinate differences and a source-evaluated subnormal visibility ratio. Existing shape, capacity, scalar-type, nonfinite, threshold, owner-index, and x64 checks remain covered.

The deterministic source algorithm, peer indexing, first-front tie behavior, last-inner overwrite, force-in gates, padding mask and eager/JIT/batch evidence are consistent on the reviewed vectors. This acceptance is limited to the standalone `TargetCache` snapshot return: it does not preserve or mutate a full `TargetState` lifecycle. Target retention, cooldown and other lifecycle fields, attachment, `update_targets`, PassingSearch, shared RNG, pacing, and moving-field integration remain deferred. This does not establish complete M3b, full simulation parity, or GPU behavior/performance.

## Source locations

- `src/retail_lane.rs`: `TargetState`, `collect`, `collect_roster`, `collect_nearby`, `collect_cached` (approximately lines 84–258).
- `src/retail_lane.rs`: `is_visible` and `front_lane_limit` (approximately lines 484–511).
