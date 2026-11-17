# Completed task record (2026-11-17)

This assignment is complete for the accepted bounded update-only milestone.
Root took the independent Rust oracle/harness work; Terra completed JAX and tests;
Luna independently accepted. Full suite75passed342.51s; see current HANDOFF for
next work. Instructions below are historical and must not restart completed work.

# Current implementer task

Root coordinates; you are Terra 5.6 implementing; Luna 6 reviews. User authorized
completion of this next milestone. Read AGENTS.md and required project docs,
then MILESTONE_3B_TARGET_UPDATE.md and REVIEW_M3B_TARGET_UPDATE.md. Do not edit
the Rust checkout or reviewer documents. All implementation stays in destination.

The prior Terra session produced initial `lane_target.py` persistent-update code,
incomplete tests, and a compiled Rust oracle with **4** four-update families in
`tests/fixtures/target_update_oracle.json`. These are UNACCEPTED. Baseline was
62 accepted tests. Root verified source revision/clean checkout and 40 baseline
hashes before this work. Current manifest's implementation hash will need an
update because the implementation is changing.

## Finish this work

1. Make the destination-only oracle harness readable: move its Python raw-string
   `h` to `tools/target_update_harness.rs`, have the generator read it, and run
   rustfmt on the harness only. Never reformat original extracted source. Preserve
   current raw oracle stdout bytes when splitting. Rust literals need leading
   zero (`0.4`), and format literal braces must be doubled. Fix compile/patch
   errors and continue; they are routine work, not reasons to stop the milestone.
2. Preserve the existing 4 families and complete the oracle with these cases:
   - Front fallback: collected owner `[0, .2, 24, 25]`, cached/live peer
     `[2, .2, 30, 30]`, empty prior list. Updates: original owner/distance2;
     original owner/distance next f32 above2; owner position.5/distance2;
     owner position.5/lane.5/distance2. Stored cache.front stays absent. This
     should discriminate bitwise parameter fallback and current-owner fallback.
   - Full capacity/invisible retention: collected owner `[0,.4,24,25]`, vision0;
     32 peers at positions `3+i*.1`, lane.4, speed20,target26; live speed20,
     target24. Initial targets reversed31..0; four updates preserve ordered
     capacity32 targets despite no visible peers.
   - Force-in/optional/zero timer: owner `[0,.2,24,25]`, cached/live peer
     `[0,.15,20,26]`; collection has force_in=true/can_in=false. Initial
     extra=None, previous_can_in=None, cooldown.1, skip=true. dt sequence
     `[0,.1,.1,0]` checks exactzero and stickyforce. Preserve all unrelated state.
   - Unsupported diagnostic: owner `[-10,.2,24,25]`, cached peers at minnormal
     f32 and nextafter(minnormal,+inf), both lane.4/speed20/target26; live0
     `[100,.4,26,24]`, live1 `[100,.4,26,26]`. Source must omit target0 because
     peer1 blocks peer0 across a positive minimum-subnormal gap. Draft checked
     JAX returns target0 because XLA flushes the gap. Confirm in Rust, then
     reject this unsupported domain at the checked boundary with a regression.
3. Oracle must serialize all initial lifecycle, collection config and horses,
   every update's current owner/live/dt/front parameters/cooldown parameter,
   and complete output state. Capture raw subprocess stdout bytes. Update
   oracle generator/extraction note/provenance, with original method bodies
   and original overtaking module unchanged. No JAX expected-value generation.
4. Address all Luna intermediate findings: validate every state/cache leaf and
   sentinel/dtype/shape before dispatch; reject infinity/overflow extra-outward;
   guard pairwise blocker position/lane differences/front narrowing, predicate
   ratios and relevant cooldown differences; define padding policy. Accept
   stale nonnegative target indices for source's drop pass. Never narrow an
   overflowing integer before checking bounds. Preserve cache inputs unchanged.
5. Replace draft update tests with full independent-oracle comparisons of every
   state leaf eager/JIT scan/checked, vmap versus scalar, input/cache immutability,
   meaningful source-order witnesses and checked-boundary regressions. Test both
   optional states and true force-in preservation, full/empty capacity and padding.
   Require explicit x64; unsupported feature flags must not silently activate.
6. Run focused cache+update tests and fix failures. Update module documentation,
   supported inventory, provenance hashes, extraction note, STATUS/HANDOFF with
   actual commands/results and remaining exclusions. Root will run full suite;
   do not duplicate it. Report a stable source-backed candidate to root for
   Luna's final review. Do not end at scaffolding or an ordinary compiler error.

Current-owner inclusion in the blocker iterator is required by exact source,
but cannot discriminate finite indirect eligibility: that peer must be ahead of
the owner, whereas owner blocking it would require the owner to be ahead. Use
another cached peer for the behavioral blocker witness. The existing4families
already show `[2,0,1,3]` ordering, negative cooldown overshoot/stickiness,
cached/live eligibility alternation and peer-block alternation.

PassingSearch/final target/RNG/pacing/field integration/experimental gap refresh
remain excluded. Do not claim full simulation or GPU parity. Continue until all
remaining implementation and validation here is complete; root owns acceptance.
