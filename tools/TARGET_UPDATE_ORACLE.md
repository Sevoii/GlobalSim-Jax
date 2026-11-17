# Persistent target-update oracle

AGPL-3.0-only. Source: read-only GlobalSim revision
`58da7bc4693d9aa8f6340e348cba1c5b5fb5a856`.

Root built this independent expected-value adapter while Terra implements the
JAX transition/tests. `build_target_update_oracle.py` checks the original lane
and overtaking source hashes against `docs/ORACLE_PROVENANCE.json`, then copies
these source spans verbatim:

- Horse, TargetParameters type (without JSON loader), TargetContext, TargetState.
- TargetState::collect, collect_cached, and update_targets.
- front_index_iter and has_front_blocker.
- The entire original overtaking.rs module.

The accepted `lane_query_oracle.rs` supplies original Row/BlockParameters/SideRoom
types and query helpers, including mixed-precision front_lane_limit/is_visible.
Its own manifest hash and original source provenance remain dependencies.
No extracted collection/update/eligibility equations or method bodies change.
Only handwritten destination inputs, serialization and diagnostic assertions
are in `target_update_harness.rs`. rustfmt applies to that harness, never to the
original extracted spans. Regeneration also requires checking the pinned git
revision and all dependency hashes; the generator's two hash checks are partial.

Compile with the default feature set (no experimental-gap-refresh cfg):

```
.venv/Scripts/python.exe tools/build_target_update_oracle.py
rustc --edition=2021 -Awarnings -O tools/target_update_oracle.rs -o tools/target_update_oracle.exe
```

Capture executable stdout using Python subprocess.check_output/Path.write_bytes,
preserving its final newline. Never replace expected values with JAX output.

## Scope and representation

Each case collects once using original source, installs explicit initial target
and lifecycle state, then runs four original update_targets calls with the same
cached roster and independently supplied current owner/live values. There is
no target search, RNG, repeated collection, field loop, pacing or skill code.
TargetState optional side/front caches are populated by collection; absent
side/front-cache structures are outside the JAX checked profile.
The source-only sorted_owner field remains None because this adapter uses
unsorted collect, which sets it to None. It is omitted from JAX state/output;
the sorted-nearby optimization and sorted-owner lifecycle are unsupported.

All float values serialize as u32 bits. Targets and visible indices retain
source Vec order. Extra-outward serializes `[present, bits]`; None uses canonical
f32 NaN bits. previous-can-in is -1/0/1; absent front/inner indices use -1.
The complete state output includes the original cached horses, visible Vec,
side room, cached front identity/parameters, and every represented lifecycle
scalar/flag. JAX's visible mask must match ascending source visible indices.
JAX target/cache padding is wrapper storage, with separate tests and explicit
checked padding rules.

The `cache.owner` four-value horse snapshot is retained from the collection
input by the harness; original front_cache itself stores only its Row. Likewise,
the separate immutable cache force-in value is a wrapper snapshot of collection;
source update_targets leaves its force_in field unchanged. These representations
do not claim additional Rust-owned state or full collection lifecycle attachment.

## Discriminating evidence

Eight supported cases, 32 updates:

- Stale index99, retained-old ordering, visible newcomer append, removal and
  unconditional front append: lists `[2,0,1,3]`, `[2,1,3]`, `[2,1,0,3]`, `[1,0,3]`.
- Empty list cooldown overshoots negative, with sticky force-check.
- Cached gap/speed versus live comparisons alternates `[0]`, `[]`, `[0]`, `[]`.
- Other cached peer blocking under changing current front distance alternates
  `[]`, `[0]`, `[]`, `[0]`.
- One-bit front-distance and current-owner position/lane changes require local
  fallback, producing `[]`, `[0]`, `[0]`, `[]`; stored cached front remains absent.
- Full32 reversed retained targets survive invisible collection and keep order.
- True force-in, absent optional values and exact-zero cooldown behavior.
- Negative-zero cooldown/extra/collection owner are preserved, with current
  owner zero-sign changes; this checks bit preservation, not an output-changing
  signed-zero fallback witness.

The unsupported diagnostic stores two normal peer coordinates whose positive
peer-to-peer difference has bits1 (minimum subnormal). Original Rust emits an
empty list; the initial unguarded CPU JAX draft emitted target0 because XLA
flushed that blocker gap to zero. Root independently confirmed both behaviors;
checked updates must reject that domain. This diagnostic is not included as a
supported JAX differential case.

Current-owner inclusion in the blocker iterator is verified by exact original
source extraction. It cannot change finite indirect eligibility: an eligible
peer is ahead of owner, whereas owner blocking it requires owner to be ahead.
The source peer-blocking case supplies the actual eligibility witness instead.

This is original-source subset evidence. It is not a complete engine replay,
final automatic targeting, GPU correctness or performance evidence. Final
review, fixture hashes, reproducibility and validation results are recorded in
REVIEW_M3B_TARGET_UPDATE.md, ORACLE_PROVENANCE.json and STATUS/HANDOFF.
