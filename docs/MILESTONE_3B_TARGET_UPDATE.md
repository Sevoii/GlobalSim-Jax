# M3b persistent target updates

2026-11-17. Root coordinates; Terra 5.6 implements through Herdr `jax-terra`
(`wC:p6`); Luna 6 independently reviews through `jax-luna` (`wC:p7`).
Baseline: accepted standalone cache collection, 62 tests. This unit is accepted within the default-feature update-only profile. Source remains read-only at GlobalSim revision
`58da7bc4693d9aa8f6340e348cba1c5b5fb5a856`.

## Scope

Port `TargetState::update_targets` for the default Rust feature configuration,
with persistent fixed-capacity ordered target indices and explicit lifecycle
state. Attach the accepted `TargetCache` snapshot as an input without changing
its existing contract. Current owner and live peers are separate inputs; every
live peer slot must have the same identity as its cached slot. No sorting or
finished-peer compaction is permitted between collection and update.

Preserve these source observations and operations:

1. Reuse cached front only if owner position/lane and front distance/lane
   parameters match **bitwise**. Otherwise recompute from cached peers and the
   current owner with the original front query, not from live peer positions.
2. Reset skip-passing, publish cached-side inward permission to previous-can-in.
   Experimental gap refresh is excluded and must not silently activate.
3. Eligibility uses cached peer position/speed relative to current owner, live
   peer target/speed, and a deferred blocker query over cached peers plus the
   current owner. Preserve f32 subtraction and predicate comparison order.
4. Drop obsolete prior indices, remember the surviving existing count, append
   new eligible visible peers in cached order, revalidate existing entries,
   then append the front peer if absent regardless of eligibility. Preserve
   the resulting ordered Vec; a set or sort is not equivalent.
5. A nonempty list resets f32 cooldown. An empty list subtracts dt only when
   cooldown was positive, retaining negative overshoot and setting force-check
   when it crosses zero. Force-check is otherwise sticky. Preserve unrelated
   lifecycle state, including optional extra-outward storage and force-in.

Expose initialization, a prevalidated JIT kernel and a checked host boundary.
Capacity is 32 peers; every target slot uses owner-excluded cached indices.
Document lifecycle optional/sentinel and padding representations explicitly.
Checked state may reject unreachable synthetic duplicate lists, but must state
this restriction and retain source-supported stale nonnegative indices for
the source drop pass. Host validation must not dispatch a discarded kernel.

## Acceptance gates

- Independently generated Rust oracle with unchanged original `update_targets`,
  collection code, `has_front_blocker` and original overtaking helper. Name all
  extraction boundaries, harness adaptations and default-feature assumptions.
  Verify pinned revision and hashes before generating raw fixture bytes.
- Multi-update traces compare the complete lifecycle and unchanged cache/state
  fields, integer/flag values and floating bits eager/JIT scan. Selected
  independent batches must equal scalar runs; valid checked calls must compare
  complete outputs to Rust, including empty/full capacity.
- Discriminating witnesses for retention/append/front order, stale indices,
  eligible retained targets outside visibility, current-owner versus cached
  observations, live target/speed versus cached speed, blocking by another
  cached peer, bitwise cache fallback, and signed/zero/negative cooldown behavior.
  Preserve current-owner inclusion in the blocker iterator by exact extraction
  and helper evidence; it cannot discriminate finite indirect eligibility:
  eligible peers must be ahead of the owner, whereas the owner could block a
  peer only from ahead. Direct-front eligibility skips that blocker query.
- Preserve sentinel/padding state and unrelated lifecycle fields across updates.
  Reject bad dtype/shape/capacity/alignment, unsupported feature flags, numeric
  overflow/subnormal operands and relevant differences/divisions/cooldown
  subtraction before dispatch. Require caller-enabled x64. Prevalidated scans
  must document any invariant numerical-domain requirements.
- Terra runs focused checks and fixes review findings; Luna writes independent
  criteria and final review in REVIEW_M3B_TARGET_UPDATE.md/REVIEW.md; root audits
  source extraction, rebuilds the fixture, verifies provenance and runs the full
  suite. Update STATUS/HANDOFF and supported inventory before ending work.

PassingSearch, final lane-target generation, target RNG consumption, pacing,
moving-field integration, experimental gap refresh, skills and broader M3c
remain separate. This unit adds no full simulation or GPU parity claim.

## Final acceptance gate

Terra5.6 implemented JAX/checked boundaries/tests; root supplied and audited the
independent original-source oracle; Luna6 independently accepted the bounded
transition,17focusedtests26.75s. Root complete suite:75passed342.51s.
Eight supported four-update traces plus one unsupported diagnostic, complete
represented state/cache comparisons, exact original-source extraction,
byte-identical Rust regeneration and all45provenance hashes verified. The Rust
checkout remains clean and pinned. Previous pairwise numeric/state validation
findings are resolved; see final REVIEW_M3B_TARGET_UPDATE.md. PassingSearch,
final target/RNG and field integration remain next, not enabled by this gate.
