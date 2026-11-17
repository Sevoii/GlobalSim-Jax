# M3b TargetState cache collection

2026-11-17. Coordinator: root. Implementer: Terra 5.6 through Herdr
`jax-terra` (`wC:p6`). Independent reviewer: Luna 6 through Herdr
`jax-luna` (`wC:p7`). These replace the historical p3/p5 sessions.

Accepted baseline: 58 tests, supplied-target moving-field integration. This
unit is **accepted as a standalone snapshot** and does not enable automatic targeting.

## Bounded contract

Port `retail_lane.rs` TargetState `collect_cached` and the stable owner-excluded
roster path. Rust remains authoritative at revision
`58da7bc4693d9aa8f6340e348cba1c5b5fb5a856`, read-only. The optional sorted-nearby
optimization is excluded; its unsorted fallback provides the reference result.

Represent horses as position/lane/speed/target f32 rows and cache a fixed
capacity peer roster with explicit count. A visible mask represents the source
visible Vec in ascending stable peer order. Front and inner-blocker indices
refer to owner-excluded peer slots. Preserve the first nearest front on ties,
the last qualifying inner blocker, side permissions/spaces, owner and front
parameter snapshots, and the exact force-in Boolean gates. Include all peers,
without filtering finished runners or reordering by position.

Standalone peer capacity is 32; complete roster capacity is 32 including owner.
Inactive storage is wrapper padding. Require caller-enabled x64 and state the
checked finite/non-subnormal numerical domain explicitly. Invalid checked
inputs must fail before the kernel; kernels are prevalidated JIT entry points.

## Evidence gate

- Compile mechanically extracted original Rust collection code with accepted
  query helpers, record extraction boundaries/adaptations and manifest hashes.
  Verify source revision and manifest before generating expected values.
- Assert every exposed cache field, floating bits, visible ordering and sentinel
  eagerly and under JIT. Test independent batched outputs versus scalar calls.
- Cover empty/full peer capacity, owner removal at first/middle/last slots,
  first-front ties, last-inner overwrite, strict side/front and inclusive vision
  boundaries, force-in threshold/phase/empty gates, and poisoned padding.
- Exercise successful checked calls and bad shapes/types/scalars, overflow,
  subnormal operands/intermediates, and x64 policy.
- Terra implements and runs focused tests; Luna writes the independent review;
  root audits source/oracle, runs the complete suite, and records the final gate.

Target retention, eligibility, cooldown lifecycle, PassingSearch, final target
generation, shared RNG consumption, pacing and moving-field target integration
remain next tasks. No complete M3b/M3, retail simulation or GPU claim follows
from this standalone snapshot unit.

## Final gate

Terra implemented the standalone snapshot; Luna independently accepted it after
the checked-boundary findings were fixed (4 focused tests in 11.37s). Root
full suite: 62 passed in 363.84s. Exact original source extraction, byte-identical
74-peer/4-roster-case Rust fixture reproduction and all 40 provenance hashes
verified; pinned Rust checkout remains unchanged. See
`tools/TARGET_CACHE_ORACLE.md` for extraction and wrapper representations.
