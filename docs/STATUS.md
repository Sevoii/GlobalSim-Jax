# Current checkpoint: persistent target updates accepted

2026-11-17: Terra5.6 implemented the bounded persistent TargetState update kernel, checked boundaries and differential tests; root built/audited the independent original-source Rust oracle; Luna6 independently reviewed and accepted the default-feature update-only profile. Coordinator full suite: `.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider` => **75 passed in 342.51s**. Terra combined cache/update focused run:17passed22.00s; Luna independent focused run:17passed26.75s. No open finding remains for this bounded unit.

Supported APIs: `lane_target.initialize_target_state`, `update_targets`, `update_targets_checked` atop an explicitly supplied collected TargetCache. Preserve ordered stale-target removal/visible append/old revalidation/unconditional front append, exact bitwise front-cache reuse or local cached-roster fallback, cached position/speed gaps with live peer target/speed, cached-peer plus current-owner blocker construction, f32 cooldown overshoot/exactzero/signedzero, sticky force_check, previous_can_in and skip reset. Every unrelated represented lifecycle/cache leaf is preserved. Target capacity32; stale nonnegative int32 indices (including99) are dropped safely. Optional extra uses f32 NaN for None, previous-can-in uses-1/0/1. Only collected side/front caches are supported; source sorted_owner=None is omitted under the unsorted profile.

Checked updates require caller-enabled x64 and exact scalar/array dtypes/shapes/sentinels. Inactive horse rows must be canonical zero, visible tails false and target tails-1; malformed/noncanonical checked state rejects before dispatch. dt/cooldown parameter are nonnegative; prior cooldown may be negative after overshoot. Unsupported nonfinite/overflow/nonzero-subnormal operands and relevant pairwise blocker/cone, predicate ratio and scalar timer arithmetic reject. The previously confirmed pairwise-subnormal divergence is fixed by explicit rejection; its original Rust empty-target result remains frozen as an unsupported diagnostic, not supported XLA parity.

Evidence:8supported original-source traces x4updates, complete lifecycle/cache bit serialization, every represented output leaf compared eager/JIT scan/checked, vmap versus scalar and input immutability. Source outputs distinguish ordering, live/cached split, cached-peer blocking and one-bit front-parameter/current-owner fallback; cover empty/full32/invisible-retained rows, true force-in, optional values and signedzero. Root independently rebuilt fixture bytes exactly, verified unchanged source extraction including original overtaking module, all45manifest hashes, and clean original Rust at58da7bc4693d9aa8f6340e348cba1c5b5fb5a856. Contract/review: MILESTONE_3B_TARGET_UPDATE.md and REVIEW_M3B_TARGET_UPDATE.md. Exact extraction/harness/representation limits: tools/TARGET_UPDATE_ORACLE.md.

Next milestone: port source PassingSearch and final TargetState::target/target_sorted for the default unsorted profile, preserving corridor rounding, stable target/visible ordering, cached crowd expansion versus live lane observations, target retention/force-check, optional outward initialization and exact shared RNG draw consumption. Keep source experimental-gap-refresh excluded. Then integrate collection/update/final targets in ordered moving-field scheduling (including the source context refresh after rushing callbacks) and port pacing/pacemaker selection and broader M3c. Do not invent a target policy or reuse post-target fixture values as update-only results. Existing retail_lane_port/retail_lane_state_port fixtures may supplement independent discriminating source traces. Automatic final target generation, collection lifecycle attachment, RNG target consumption, field integration, pacing, skills, full M3b/M3, full simulation/post-goal parity and GPU validation/performance remain unsupported.

Coordination: root is main coordinator, Terra5.6 codes and Luna6 reviews through Herdr. Current agents: `jax-terra`wC:p6 GPT-5.6-Terra high (fresh session01a1eb0b-de34-7730-ad30-9b6c66a6cc21), `jax-luna`wC:p7 GPT-6-Luna medium, samewC:t1. Earlier Terra medium session ended after partial work and was replaced by the SAME model with higher effort; no model substitution. Usage limits were low, so inspect live state/limits before reuse. Root owns Herdr control/source-oracle audits/fullsuite; workers should not invoke Herdr or approve dialogs. Preserve main.py/.idea/.venv and the original Rust repository. No publishing/deployment.

## Historical checkpoints

Earlier entries are superseded by the current checkpoint above.

# Current checkpoint: standalone target cache accepted

2026-11-17: Terra 5.6 implemented and corrected the standalone TargetCache collection unit; Luna 6 independently accepted it after resolving both checked-boundary findings. Root coordinator full suite: `.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider` => **62 passed in 363.84s**. Terra focused file: 4 passed in 11.12s; Luna independent focused file: 4 passed in 11.37s.

Supported: source `collect`/unsorted `collect_roster`/`collect_cached` snapshot, stable owner-excluded peer indices, visible-mask order, first-front ties, last-inner overwrite, side spaces/permissions, owner/front cache and force-in gates. Standalone peer capacity32; roster capacity32 total (31 peers). Checked APIs validate/canonicalize on the host, require caller-enabled x64, reject unsupported nonfinite/overflow/nonzero-subnormal operands and derived query values. Visibility-ratio guards apply only where Rust evaluates the division. Poisoned inactive padding is ignored and zeroed in returned storage. TargetCache returns a snapshot, without mutating or preserving a full TargetState lifecycle.

Evidence:74 independent original-source Rust peer cases plus4 actual roster cases; every cache field checked eager/JIT, checked APIs against source values, selected vmap isolation, owner first/middle/last removal, empty/full capacity and numerical boundaries. Root freshly compiled the oracle and reproduced fixture bytes exactly, verified exact extracted source spans and all40manifest hashes, and confirmed the original Rust checkout is clean at58da7bc4693d9aa8f6340e348cba1c5b5fb5a856. Review: docs/REVIEW_M3B_TARGET_CACHE.md. Contract: docs/MILESTONE_3B_TARGET_CACHE.md. Extraction/schema/adaptations: tools/TARGET_CACHE_ORACLE.md. No open finding remains for this bounded unit.

Next: port persistent TargetState `update_targets` (retention ordering, cached eligibility with live speed/target and front-cache fallback, cooldown/force flags), then PassingSearch and final target generation with exact shared RNG consumption, followed by moving-field integration and pacing/pacemaker selection. Keep the experimental-gap-refresh feature explicitly excluded unless separately requested/validated. Existing retail_lane_state_port.json may seed cases but must not replace new discriminating cached/live and RNG traces. Automatic targeting/lifecycle attachment, pacing, broader M3c, skills and GPU evaluation remain unsupported. No complete M3b/M3, full retail/post-goal parity or GPU speedup claim.

Coordination: user reaffirmed root as main coordinator, Terra5.6 coding and Luna6 reviewing through Herdr. Current banner-verified agents: `jax-terra` wC:p6 GPT-5.6-Terra; `jax-luna` wC:p7 GPT-6-Luna, same wC:t1. Historical p3/p5 agents no longer exist. Inspect live state before reuse; do not substitute models. Root owns source/oracle audits and full-suite gate. Preserve original Rust, main.py, .idea and .venv.

## Previous accepted baseline and historical checkpoints

# Status

2026-11-17 TARGET UPDATE IMPLEMENTER CHECKPOINT (pending Luna/root gates): Terra
completed the bounded JAX update candidate and its checked contract. It now
validates all represented state/cache leaves, exact dtype/shape/sentinel and
canonical-padding rules; accepts stale nonnegative int32 targets for the source
drop pass; preserves optional lifecycle fields/cache inputs; rejects infinities,
overflow and nonzero scalar/pairwise/predicate subnormal domains before JAX.
The root-built original-source fixture has 8 supported x4-update families plus
the source-confirmed pairwise-minimum-subnormal diagnostic. Focused command
`.venv\\Scripts\\python.exe -m pytest tests\\test_lane_target_update.py tests\\test_lane_target.py -q -p no:cacheprovider`
=> **17 passed in 22.00s** (all full-state bits eager/JIT scan/checked, selected
vmap isolation, cache/input immutability, invalid-domain regressions). Manifest
records the update oracle/tool/fixture/test and current implementation hashes.
No full suite was run; this is not acceptance or broader target/RNG/pacing/field/
simulation/GPU support.

2026-11-17: supplied-target moving-lane field integration accepted. Terra 5.6 implemented the JAX core/differential tests; root corrected host boundaries, built the actual-source Rust adapter and CLI; Luna 6 independently reviewed and accepted the bounded profile. Coordinator full suite: 58 passed in 302.77s. Terra moving-field focused file:4passed218.82s; coordinator checked file:3passed3.09s; Luna independently ran checked tests plus ordering witnesses:4passed10.23s.

Supported profile: restricted flat constant-width field, supplied persistent lane targets, empty lane modifiers, cached tick-entry side permissions/rank/front identity-gap, runner-ordered lateral then longitudinal movement with live blocker speed, full lateral dt for partial gates, overlap after all new positions, and whole-state first-finish/all-timeout padding. Mixed f32/f64 source evaluation is retained; blocking timers stay f64. Checked APIs reject invalid shape/type/range, overflow and unsupported nonzero f32 subnormal operands/side differences, and require caller-enabled x64. moved is per-pass wrapper telemetry, not unported sticky skill-receiver state.

Evidence:9 independent Rust cases x8ticks x3runners; all23state channels exact-compared eager/JIT,8diagnostic observation channels checked on active entry passes, realbatch independence and bytewise terminal freeze. Three original-source counterfactuals differ for cached side/rank and post-position overlap; active live-blocker-speed case included. Root rebuilt the fixture byte-for-byte, verified exact source extraction except declared insertion/timeout guard and all35manifest hashes. Original Rust checkout clean at58da7bc4693d9aa8f6340e348cba1c5b5fb5a856. See REVIEW_M3B_INTEGRATION.md final acceptance and tools/MOVING_FIELD_ORACLE.md extraction/adaptations.

Remaining: automatic TargetState/PassingSearch target generation, pacing/pacemaker selection, M3c rushing/competition/spurt/shared RNG, skills and GPU evaluation. Complete M3b/M3, full simulation/post-goal retail parity and GPU correctness/performance are not established.

## Historical checkpoints

2026-11-17 ACTIVE integration checkpoint: moving_field.py ordered kernel+scan+initial host validation implemented, UNACCEPTED. Root rejected an earlier simplified oracle draft and replaced it with actual original-helper Rust adapter. Final independent fixture9cases?8ticks?3runners,23statechannels+8observationchannels; three incorrect-order counterfactuals provably differ. Exact source/adaptations: tools/MOVING_FIELD_ORACLE.md. Terra currently implements tests/test_moving_field.py comparing every field eager/JIT/batch and terminal freeze; host validation/CLI/finalLuna review remain. Query milestone accepted baseline51tests. Keep existingagents/models.

2026-11-17 CHECKPOINT: standalone side-room/visibility/overtaking helpers accepted by Luna 6. Coordinator full suite 51 passed in 79.46s; Luna focused 8 passed in 2.80s. Independent query Rust oracle reproduced byte-for-byte; 20 side-room/12 visibility/26 overtaking plus2 unsupported diagnostics;33 provenance hashes verified. Checked profile rejects f32 overflow/nonzero-subnormal operands and side-room subnormal deltas/spaces, preserves f64 underflow-to-zero, and requires explicit x64. M3a remains fixed-lane; no full M3b/GPU claim.

ACTIVE NEXT: supplied-target moving-lane field integration per docs/MILESTONE_3B_INTEGRATION.md. Terra5.6 codes; Luna6 reviews; inspect Herdr live state before duplicate prompts. Preserve all51acceptedtests and pinnedread-onlyRust. Automatic target selection/pacing is excluded.

2026-11-17 ACTIVE: standalone side-room, visibility and overtaking helper milestone. Plan: docs/MILESTONE_3B_QUERIES.md. Accepted baseline: 43 tests. Terra 5.6 implements; Luna 6 reviews. Moving-lane field integration follows helper acceptance; preserve source mixed f32/f64.

2026-11-17: Luna published preliminary source-backed numerical, ordering, checked-domain, and fixture acceptance criteria for `side_room`, `is_visible`, and `overtaking::is_target` in `docs/REVIEW_M3B_QUERIES.md`. Read-only criteria pass only; no implementation review or tests performed. Await stable helper implementation for bounded final review.

2026-11-17: Static review of initial `lane_queries.py` recorded missing post-f32-overflow validation for checked scalar inputs in side-room, visibility, and overtaking wrappers. Overtaking Boolean flag type checks and source helper ordering inspected as correct. See `docs/REVIEW_M3B_QUERIES.md`; no tests or implementation edits in this review pass.

2026-11-17: Query checked-domain follow-up: finite pre-cast inputs can narrow to XLA-flushed nonzero f32 subnormals, reproducing coordinator divergences in overtaking and visibility. Review now requires source-faithful subnormal handling or explicit checked rejection of nonzero subnormal narrowed values and side-room stored intermediates, while preserving f64-to-f32 underflow-to-zero. No tests or implementation edits.

2026-11-17: Final bounded query snapshot: focused files pass 6/6 (1.84s); current wrappers reject overflow/nonzero-subnormal domain and preserve f64 underflow-to-zero. Low-level helper algorithms accepted within source domain. Checked helper milestone remains pending positive checked-output tests, a side-room bool-row rejection, normal-input/subnormal-intermediate and visibility-overflow tests, a target-comparison narrowing witness, and query oracle/fixture manifest entries. See `docs/REVIEW_M3B_QUERIES.md`; no full suite.

# Status

2026-11-17 CURRENT M3b query helpers: added standalone fixed-capacity
`lane_queries` side-room, visibility, and public Boolean overtaking kernels,
with checked boundaries and a destination Rust oracle. Focused first pass:
`tests/test_lane_queries.py` => 2 passed. This is in review; no field
integration, targeting, pacing, or full M3b claim.

2026-11-16: M3b sort/overlap and standalone supplied-target MoveState accepted by Luna 6 after Terra 5.6 implementation. Coordinator full suite: 43 passed in 83.75s; Luna focused lane suite: 9 passed in 29.37s. Independently rebuilt 13-case Rust movement oracle reproduces the frozen fixture byte-for-byte; extracted source blocks are unchanged, all 30 manifest hashes match, and the 700-vector copied fixture matches the Rust original. Rust checkout remains clean at 58da7bc4693d9aa8f6340e348cba1c5b5fb5a856.

Implemented: exact capacity-32 comparison-sort permutations and ordered overlap correction; standalone MoveState with lane, target, speed, continuation time/direction, checked scalar normalization and integer bounds. Full-state eager/JIT vectors cover blocked momentum, reversals, zero target/speed, continuation, after-goal movement and late rank. A subnormal-safe approximately comparison preserves the Rust zero threshold under CPU XLA.

Remaining: side_room/is_visible and overtaking helpers, supplied-target moving-lane field integration, automatic TargetState/pacing, M3c and later milestones. M3a field remains fixed-lane. Standalone after-goal movement does not enable post-goal field scheduling. Full M3b, full simulation parity and GPU correctness/performance remain unvalidated.

## Historical checkpoints

2026-11-16 CURRENT M3b MoveState task: `globalsim_jax/lane_move.py` now has a
standalone supplied-target source-order kernel, copied 700-vector
`retail_lane_port.json`, and a frozen 13-case full-state Rust oracle. The
oracle mechanically extracts pinned `SideRoom`, `MoveParameters` (loader
omitted), `MoveState`, `MoveContext`, `approximately`, and `MoveState::step`,
while including pinned `retail_physics.rs`. It compiles with Rust 2021. Full
state lanes/target/speed/continuation-time/direction/moved compare eager and
jitted-vmap; checked Python scalar canonicalization and bounds are covered.
Focused validation: lane primitives, legacy moves, and full-state moves: 9
passed in 22.85s. No field moving-lane integration, targeting, pacing, or full
M3b claim.
No moving-field integration, automatic targeting, pacing, or full M3b claim.

2026-11-16 CURRENT M3b bounded primitive task: implemented and pending Luna
review. `globalsim_jax.lane` now has an exact fixed-capacity (32) JAX port of
`retail_sort::comparison_sort`, including source size-2/3 networks, equal-key
partition behavior beyond 16, and heap fallback, plus source-ordered
`resolve_overlaps`. Checked wrappers reject unsupported rows rather than cap
them. `tools/lane_oracle.rs` imports pinned `retail_sort.rs` and freezes
source-extracted sort/overlap rows in `tests/fixtures/lane_oracle.json`; hashes
are recorded in the manifest. Focused validation after checked-boundary fixes:
`python -m pytest tests/test_lane_primitives.py -q -p no:cacheprovider` => 4
passed. M3b MoveState, side-room/visibility, overtaking, and moving-lane
integration remain next; no full M3b/M3 claim.

2026-11-16 FINAL: M3a accepted by Luna6 after Terra5.6 implementation. Coordinator final full suite on the strengthened two-runner freeze test: 34 passed in 55.69s. Earlier independent full run: 34 passed in 53.59s; strengthened targeted regression: 1 passed in 3.94s. Both rebuilt Rust field oracles reproduce frozen fixtures; manifest hashes match and Rust checkout is clean. field-demo --ticks16 runs. M3b/M3c and full milestone3 remain unfinished; GPU validation remains future work.

2026-11-16: Luna published preliminary source-backed M3b acceptance criteria in `docs/REVIEW_M3B.md` for deterministic sort/overlap, visibility/side room, overtaking predicate, and supplied-target `MoveState` integration. No implementation was reviewed or changed in this criteria pass. Automatic target selection and pacing remain explicitly excluded; M3a remains accepted within its prior bounded contract.

2026-11-16: Bounded independent review of the stable M3b sort/overlap unit: focused tests `tests/test_lane_primitives.py` pass 4/4 (15.99s); exact Rust ID permutations and active outputs verified. Sort/overlap kernel behavior accepted within tested vectors. Checked f32-to-i32 lane bound at +2^31 and nonempty padding-tail assertions remain in `docs/REVIEW_M3B.md`; no full suite or MoveState review was performed.

2026-11-16: Bounded static review of the first standalone `lane_move.py` snapshot recorded checked-input canonicalization, pre-inward zero-speed return ordering, and full-state edge coverage findings in `docs/REVIEW_M3B.md`. No tests were run while the draft is being edited; no MoveState acceptance yet.

2026-11-16: Added read-only M3b supplied-target integration criteria to `docs/REVIEW_M3B.md`: cached tick-entry side permissions, per-runner lateral-before-longitudinal ordering, gate-fraction distinction, post-longitudinal overlap, and M3a crossing/terminal freeze. No tests or implementation edits; final MoveState review remains pending a stable snapshot.

2026-11-16: Bounded re-review of stable lane kernels: focused primitive+MoveState tests pass 6/6 in 20.75s; +2^31 checked lane rejection and active padding tails were manually confirmed. Sort/overlap kernel behavior accepted in scope, with nonempty padding-tail regression still requested. MoveState kernel canonicalization and pre-inward zero check now match source ordering, but MoveState acceptance awaits compiled extracted-source Rust full-state fixture/tests; current oracle harness has a format placeholder error. No full suite.

2026-11-16: M3a implementation is complete and awaiting final Luna review. The checked fixed-lane field profile provides ranks/ties, front/side blocking snapshots, ordered stored-minimum/live-speed cap movement, and full-state first-finish/all-timeout freezing. Independent Rust evidence includes three 8-tick/3-runner/19-channel field cases, two-batch scalar independence, an order-sensitive cap counterexample, query boundaries, and custom f32 front-lane storage. Coordinator complete validation: 34 passed. `field-demo --ticks 8` runs the checked profile. M3b pacing/lane/overtaking and M3c rushing/competition/spurt/shared RNG remain deferred; whole M3 is not complete.

2026-11-16: Milestone 2 restricted subset accepted narrowly by Luna final review. It contains source-backed stat conversion helpers, accepted standalone course10101/1200m geometry, and a checked flat/no-field `[batch,runner]` loop. `initialize`/immutable `RestrictedConfig`/`run_checked` validate host inputs; `step`/`run` are prevalidated JIT kernels. Six pinned-Rust eight-tick cases assert nine per-tick channels eager/JIT, while divergent `[2,2]` batch/runner traces match scalar runs. `docs/ORACLE_PROVENANCE.json` records final source/oracle/fixture hashes. Coordinator final validation: 17 passed in 27.86s; `restricted-demo --ticks 1200` produced `final_position=1200.084961`. This is restricted flat/no-field evidence only, not full-race or GPU parity.

2026-11-16: Milestone 1 numerical foundation implemented and accepted by GPT-6-Luna final review; coordinator validation complete. Added installable `globalsim_jax` package, copied pinned independent RNG/physics/finish fixtures and parameter data, CLI primitive benchmark, supported-function inventory, and tests. Existing `main.py`, IDE, and venv files preserved. CPU environment: Python 3.13.7, JAX/JAXlib 0.12.0.1, `CpuDevice(id=0)` only; no GPU backend validated. Final validation: `python -m pytest tests -q -p no:cacheprovider` => 8 passed (20.80s; independent coordinator rerun 17.25s). Exact eager and JIT assertions pass for all copied physics fixture outputs, including HP drain in both standalone and composed JIT calls. HP drain now uses literal f32 division with barriers around numerator, denominator, and quotient to prevent XLA reciprocal contraction. Synchronized primitive benchmark: batch 1024 Ã— 10 `rng.next_double`, CPU, coordinator ~2.20M samples/s (not a race or GPU result). Editable installation succeeded under coordinator escalation: `pip install -e .[dev]`. Package import preserves global JAX x64 configuration. Benchmark command is synchronized primitive-only and not a race/GPU speed result.

2026-11-16: Milestone 2 started. `docs/MILESTONE_2.md` records the initially proposed fixed-shape flat/no-field subset and explicitly rejects interactions and skills. Coordinator review correctly retains M2 requirements for source-backed initialization/stat conversion and fixture-backed geometry as required work; neither may be silently omitted. Added source-backed `globalsim_jax.initialization` helpers for overcap, team-rank, mood and clamped base stats, with regression tests. `globalsim_jax/restricted.py` is an in-progress runnable scaffold only: it is not supported or review-ready because source-order gaps remain (dash state/HP branch, deceleration and HP-empty handling, phase timing, full-dt HP drain, f64 gate movement, actual course distance, validation, geometry integration, and independent Rust oracle trace). M1 plus stat tests: 9 passed. No M2 parity claim yet.

2026-11-16: M2 geometry bounded task complete: bundled source-derived captured table `course_10101.json`; implemented `CourseGeometry.load/position/advance` for 10101/1200m only, rejecting overrun/other courses. Copied `retail_geometry_port.json`; all supported pre-goal rows now pass eager and batched JIT exact-bit checks, with endpoint and host invalid-input coverage. Destination-only `tools/geometry_oracle.rs` compiles pinned `json.rs`/`retail_geometry.rs` read-only and confirmed the fixture at the prior one-ULP row. JAX quaternion normalization needed explicit f32 operation barriers to reproduce that source result. Full M2 remains in progress; geometry is not integrated into the restricted loop yet.

2026-11-16: Restricted source oracle bounded task complete: `tools/restricted_oracle.rs` compiles pinned `json.rs`/`retail_physics.rs` read-only and froze `tests/fixtures/restricted_trace.json`. It supplies three independent eight-tick no-field cases (delayed non-unit ground, phase crossing, near-goal freeze), with per-tick f32 bit channels. JAX differential tests remain the next M2 task.

2026-11-16: Restricted differential bounded task complete: `tests/test_restricted.py` loads only frozen Rust-oracle rows and asserts all eight channels (time, position, speed, HP, phase, dash, finished, finish-time bits) for eager stepping and jitted `lax.scan`. Terminal flags and clock are frozen after finish/timeout; timeout is explicitly wrapper truncation, not retail behavior. The fixture is regenerated byte-for-byte with Python `subprocess.check_output` from the destination oracle (avoids PowerShell UTF-16 redirection). Validation: `python -m pytest tests -q -p no:cacheprovider` => 13 passed (21.87s). Restricted loop remains flat/no-field and its factors are the oracle's explicit unity aptitude values; no full-race claim.

Final coordinator run after both residual review fixes: 8 passed in 17.59s, no xfails; installed globalsim-jax CLI benchmark runs. Source Rust repository remains unchanged. Next: milestone 2 restricted race loop and separate WSL2/CUDA backend validation; neither is implemented yet.

2026-11-17: Final bounded M3b query-helper review accepted standalone `side_room`, `is_visible`, and public-Boolean `is_target` within the documented checked domain. Focused validation: `.venv\Scripts\python.exe -m pytest tests/test_lane_queries.py tests/test_lane_query_checked.py -p no:cacheprovider -q` => 8 passed in 2.80s. See `docs/REVIEW_M3B_QUERIES.md`; this does not accept field integration, remaining M3b/M3c, or GPU.

2026-11-17: Luna completed a read-only design review of supplied-target moving-field integration. Source-backed schedule corrections and acceptance criteria are in `docs/REVIEW_M3B_INTEGRATION.md`, including cached ranks, entry-side roster order, gate early return, per-runner live roster refresh, post-movement overlap, and M3a whole-batch truncation limits. No implementation review/tests or M3b field-integration acceptance yet; Terra implementation and independent Rust subset evidence remain next.

2026-11-17: Re-evaluation resolved the initial `moving_field.py` minimum-prepass finding as a false positive: `prepare_blocking` carries `state.minimum` through an ordered loop, exposing only earlier indices' updated minima and leaving later indices' stored minima visible, then movement consumes each owner's own result. No actionable core-step issue remains in the static snapshot review. Independent Rust fixture/eager/JIT/batch acceptance remains pending; see `docs/REVIEW_M3B_INTEGRATION.md`.

2026-11-17: Bounded static review of moving-field Rust oracle found its pinned source modules, Rust-generated 9×8×3 fixture, manifest hashes, and cached-side/post-overlap/rank counterfactual witnesses consistent with the documented restricted extraction. Static host-boundary review found `run_checked` accepts nonzero f32 subnormals and derived side-room subnormal gaps, then calls the unchecked low-level query kernel; this is outside the established checked query profile. Terra should reject that domain or establish a bounded alternative before final review. No tests run; see `docs/REVIEW_M3B_INTEGRATION.md`.


2026-11-17 final bounded M3b supplied-target moving-field review: Luna accepted the constant-width, no-skills/no-pacing profile with explicit targets and M3a whole-batch truncation. The checked host subnormal/range/target-initialization findings are resolved. Luna focused checked+witness selection: 4 passed in 10.23s; oracle/fixture hashes match manifest. Terra reported all nine 8-tick cases/23 channels pass eager and jitted scan (153.36s); root full suite was still running, so its result remains a separate coordinator gate. No full M3, automatic targeting/pacing, or GPU claim. See docs/REVIEW_M3B_INTEGRATION.md.
