# Current checkpoint: persistent target updates accepted

2026-11-17: Terra5.6 implemented the bounded persistent TargetState update kernel, checked boundaries and differential tests; root built/audited the independent original-source Rust oracle; Luna6 independently reviewed and accepted the default-feature update-only profile. Coordinator full suite: `.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider` => **75 passed in 342.51s**. Terra combined cache/update focused run:17passed22.00s; Luna independent focused run:17passed26.75s. No open finding remains for this bounded unit.

Supported APIs: `lane_target.initialize_target_state`, `update_targets`, `update_targets_checked` atop an explicitly supplied collected TargetCache. Preserve ordered stale-target removal/visible append/old revalidation/unconditional front append, exact bitwise front-cache reuse or local cached-roster fallback, cached position/speed gaps with live peer target/speed, cached-peer plus current-owner blocker construction, f32 cooldown overshoot/exactzero/signedzero, sticky force_check, previous_can_in and skip reset. Every unrelated represented lifecycle/cache leaf is preserved. Target capacity32; stale nonnegative int32 indices (including99) are dropped safely. Optional extra uses f32 NaN for None, previous-can-in uses-1/0/1. Only collected side/front caches are supported; source sorted_owner=None is omitted under the unsorted profile.

Checked updates require caller-enabled x64 and exact scalar/array dtypes/shapes/sentinels. Inactive horse rows must be canonical zero, visible tails false and target tails-1; malformed/noncanonical checked state rejects before dispatch. dt/cooldown parameter are nonnegative; prior cooldown may be negative after overshoot. Unsupported nonfinite/overflow/nonzero-subnormal operands and relevant pairwise blocker/cone, predicate ratio and scalar timer arithmetic reject. The previously confirmed pairwise-subnormal divergence is fixed by explicit rejection; its original Rust empty-target result remains frozen as an unsupported diagnostic, not supported XLA parity.

Evidence:8supported original-source traces x4updates, complete lifecycle/cache bit serialization, every represented output leaf compared eager/JIT scan/checked, vmap versus scalar and input immutability. Source outputs distinguish ordering, live/cached split, cached-peer blocking and one-bit front-parameter/current-owner fallback; cover empty/full32/invisible-retained rows, true force-in, optional values and signedzero. Root independently rebuilt fixture bytes exactly, verified unchanged source extraction including original overtaking module, all45manifest hashes, and clean original Rust at58da7bc4693d9aa8f6340e348cba1c5b5fb5a856. Contract/review: MILESTONE_3B_TARGET_UPDATE.md and REVIEW_M3B_TARGET_UPDATE.md. Exact extraction/harness/representation limits: tools/TARGET_UPDATE_ORACLE.md.

Next milestone: port source PassingSearch and final TargetState::target/target_sorted for the default unsorted profile, preserving corridor rounding, stable target/visible ordering, cached crowd expansion versus live lane observations, target retention/force-check, optional outward initialization and exact shared RNG draw consumption. Keep source experimental-gap-refresh excluded. Then integrate collection/update/final targets in ordered moving-field scheduling (including the source context refresh after rushing callbacks) and port pacing/pacemaker selection and broader M3c. Do not invent a target policy or reuse post-target fixture values as update-only results. Existing retail_lane_port/retail_lane_state_port fixtures may supplement independent discriminating source traces. Automatic final target generation, collection lifecycle attachment, RNG target consumption, field integration, pacing, skills, full M3b/M3, full simulation/post-goal parity and GPU validation/performance remain unsupported.

Coordination: root is main coordinator, Terra5.6 codes and Luna6 reviews through Herdr. Current agents: `jax-terra`wC:p6 GPT-5.6-Terra high (fresh session01a1eb0b-de34-7730-ad30-9b6c66a6cc21), `jax-luna`wC:p7 GPT-6-Luna medium, samewC:t1. Earlier Terra medium session ended after partial work and was replaced by the SAME model with higher effort; no model substitution. Usage limits were low, so inspect live state/limits before reuse. Root owns Herdr control/source-oracle audits/fullsuite; workers should not invoke Herdr or approve dialogs. Preserve main.py/.idea/.venv and the original Rust repository. No publishing/deployment.

Validation commands from destination:
- `.venv/Scripts/python.exe -m pytest tests/test_lane_target_update.py tests/test_lane_target.py -q -p no:cacheprovider`
- `.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider`
- `.venv/Scripts/python.exe tools/build_target_update_oracle.py`
- `rustc --edition=2021 -Awarnings -O tools/target_update_oracle.rs -o <temporary executable>`
Capture executable stdout with Python subprocess/check_output/write_bytes, retaining its final newline. Verify revision and all45manifest hashes before regeneration; generator checks two original source hashes and requires the accepted lane_query_oracle dependency. Full suite currently takes about6minutes; avoid duplicate runs.

## Historical checkpoints

Earlier entries are superseded by the current checkpoint above.

# Current checkpoint: standalone target cache accepted

2026-11-17: Terra 5.6 implemented and corrected the standalone TargetCache collection unit; Luna 6 independently accepted it after resolving both checked-boundary findings. Root coordinator full suite: `.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider` => **62 passed in 363.84s**. Terra focused file: 4 passed in 11.12s; Luna independent focused file: 4 passed in 11.37s.

Supported: source `collect`/unsorted `collect_roster`/`collect_cached` snapshot, stable owner-excluded peer indices, visible-mask order, first-front ties, last-inner overwrite, side spaces/permissions, owner/front cache and force-in gates. Standalone peer capacity32; roster capacity32 total (31 peers). Checked APIs validate/canonicalize on the host, require caller-enabled x64, reject unsupported nonfinite/overflow/nonzero-subnormal operands and derived query values. Visibility-ratio guards apply only where Rust evaluates the division. Poisoned inactive padding is ignored and zeroed in returned storage. TargetCache returns a snapshot, without mutating or preserving a full TargetState lifecycle.

Evidence:74 independent original-source Rust peer cases plus4 actual roster cases; every cache field checked eager/JIT, checked APIs against source values, selected vmap isolation, owner first/middle/last removal, empty/full capacity and numerical boundaries. Root freshly compiled the oracle and reproduced fixture bytes exactly, verified exact extracted source spans and all40manifest hashes, and confirmed the original Rust checkout is clean at58da7bc4693d9aa8f6340e348cba1c5b5fb5a856. Review: docs/REVIEW_M3B_TARGET_CACHE.md. Contract: docs/MILESTONE_3B_TARGET_CACHE.md. Extraction/schema/adaptations: tools/TARGET_CACHE_ORACLE.md. No open finding remains for this bounded unit.

Next: port persistent TargetState `update_targets` (retention ordering, cached eligibility with live speed/target and front-cache fallback, cooldown/force flags), then PassingSearch and final target generation with exact shared RNG consumption, followed by moving-field integration and pacing/pacemaker selection. Keep the experimental-gap-refresh feature explicitly excluded unless separately requested/validated. Existing retail_lane_state_port.json may seed cases but must not replace new discriminating cached/live and RNG traces. Automatic targeting/lifecycle attachment, pacing, broader M3c, skills and GPU evaluation remain unsupported. No complete M3b/M3, full retail/post-goal parity or GPU speedup claim.

Coordination: user reaffirmed root as main coordinator, Terra5.6 coding and Luna6 reviewing through Herdr. Current banner-verified agents: `jax-terra` wC:p6 GPT-5.6-Terra; `jax-luna` wC:p7 GPT-6-Luna, same wC:t1. Historical p3/p5 agents no longer exist. Inspect live state before reuse; do not substitute models. Root owns source/oracle audits and full-suite gate. Preserve original Rust, main.py, .idea and .venv.

Validation commands from destination:
- `.venv/Scripts/python.exe -m pytest tests/test_lane_target.py -q -p no:cacheprovider`
- `.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider`
- `rustc --edition=2021 -Awarnings -O tools/target_collect_oracle.rs -o <temporary executable>`
Capture Rust stdout with Python subprocess/check_output/write_bytes. Verify pinned revision and every manifest hash before regeneration. The target-cache generator checks its primary source hash, but this does not replace dependency/revision verification. The complete suite currently takes about6minutes; avoid duplicate runs.

## Historical checkpoints

Earlier entries below are superseded by this current checkpoint.

# Current restart checkpoint: moving-field integration accepted

2026-11-17: supplied-target moving-lane field integration accepted. Terra 5.6 implemented the JAX core/differential tests; root corrected host boundaries, built the actual-source Rust adapter and CLI; Luna 6 independently reviewed and accepted the bounded profile. Coordinator full suite: 58 passed in 302.77s. Terra moving-field focused file:4passed218.82s; coordinator checked file:3passed3.09s; Luna independently ran checked tests plus ordering witnesses:4passed10.23s.

Supported profile: restricted flat constant-width field, supplied persistent lane targets, empty lane modifiers, cached tick-entry side permissions/rank/front identity-gap, runner-ordered lateral then longitudinal movement with live blocker speed, full lateral dt for partial gates, overlap after all new positions, and whole-state first-finish/all-timeout padding. Mixed f32/f64 source evaluation is retained; blocking timers stay f64. Checked APIs reject invalid shape/type/range, overflow and unsupported nonzero f32 subnormal operands/side differences, and require caller-enabled x64. moved is per-pass wrapper telemetry, not unported sticky skill-receiver state.

Evidence:9 independent Rust cases x8ticks x3runners; all23state channels exact-compared eager/JIT,8diagnostic observation channels checked on active entry passes, realbatch independence and bytewise terminal freeze. Three original-source counterfactuals differ for cached side/rank and post-position overlap; active live-blocker-speed case included. Root rebuilt the fixture byte-for-byte, verified exact source extraction except declared insertion/timeout guard and all35manifest hashes. Original Rust checkout clean at58da7bc4693d9aa8f6340e348cba1c5b5fb5a856. See REVIEW_M3B_INTEGRATION.md final acceptance and tools/MOVING_FIELD_ORACLE.md extraction/adaptations.

Remaining: automatic TargetState/PassingSearch target generation, pacing/pacemaker selection, M3c rushing/competition/spurt/shared RNG, skills and GPU evaluation. Complete M3b/M3, full simulation/post-goal retail parity and GPU correctness/performance are not established.

Next milestone: plan and port automatic lane TargetState (collect_roster/update_targets/PassingSearch) using the source cached/live observations and exact shared RNG draw order; then pacing/pacemaker and broader M3c. Source fixtures retail_lane_state_port.json and retail_lane_port.json are useful but do not replace new discriminating ordering/RNG traces. Do not invent a target/pacing policy. Read PORT_PLAN, STATUS, SUPPORTED_FUNCTIONS, MILESTONE_3B_INTEGRATION and the final review before new implementation.

Project: C:/Users/a/PycharmProjects/GlobalSim-Jax. Reference: C:/Users/a/RustroverProjects/GlobalSim (read-only). Preserve main.py/.idea/.venv. The original Rust revision is pinned above; verify it and all manifest hashes before regenerating fixtures. Generators do not enforce the revision automatically. Moving oracle includes accepted lane_move_oracle/lane_query_oracle modules and pinned physics/sort; compile with Rust2021. Actual fixture dt is f32 .066600002348423 (not mathematical1/15), distance1200, explicit movement parameters(.001,.3,.02,.05,1.5). Source/data/executable expected values must never be replaced by JAX-generated output.

Commands from destination:
- .venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
- .venv/Scripts/python.exe -m globalsim_jax.cli moving-field-demo --ticks 8
- rustc --edition=2021 -Awarnings -O tools/moving_field_oracle.rs -o <temporary executable>
Capture oracle bytes using Python subprocess/check_output/write_bytes, not PowerShell redirection. Optional Rust gate fraction None is serialized as NaN diagnostic payload; JAX helper uses zero plus a separate gate flag. Tests normalize diagnostic absence and compare observations only on active entry rows. Every physical/wrapper state leaf remains asserted, including f64 timers and terminal padding.

Agents remain in Herdr: jax-terra wC:p3 gpt-5.6-terra, jax-luna wC:p5 gpt-6-luna; last turns completed. Inspect live state and current usage limits before prompting; both budgets were low. Do not substitute models without user direction. Verify HERDR_ENV=1 before control, use explicit agent names/pane IDs, and use elevated tool access for the elevated Herdr server. Never answer an agent approval UI on its behalf. The native agent tool does not expose Terra5.6; use Herdr. Root owns final source/oracle verification, validation and handoff.

The complete moving-field test file takes about3.5minutes, full suite about5minutes. Do not launch duplicate background runs. Poll the existing exec session and read actual failure output; test-harness encoding/shape corrections are not physics changes. Earlier simplified oracle drafts were rejected and replaced: current oracle uses real source MoveState/side_room/overlap, never hardcoded permissions or lane equations.

## Historical checkpoints

Earlier entries below are superseded and may mention resolved defects or unaccepted drafts.

2026-11-17 ACTIVE integration checkpoint: moving_field.py ordered kernel+scan+initial host validation implemented, UNACCEPTED. Root rejected an earlier simplified oracle draft and replaced it with actual original-helper Rust adapter. Final independent fixture9cases?8ticks?3runners,23statechannels+8observationchannels; three incorrect-order counterfactuals provably differ. Exact source/adaptations: tools/MOVING_FIELD_ORACLE.md. Terra currently implements tests/test_moving_field.py comparing every field eager/JIT/batch and terminal freeze; host validation/CLI/finalLuna review remain. Query milestone accepted baseline51tests. Keep existingagents/models.

2026-11-17 CHECKPOINT: standalone side-room/visibility/overtaking helpers accepted by Luna 6. Coordinator full suite 51 passed in 79.46s; Luna focused 8 passed in 2.80s. Independent query Rust oracle reproduced byte-for-byte; 20 side-room/12 visibility/26 overtaking plus2 unsupported diagnostics;33 provenance hashes verified. Checked profile rejects f32 overflow/nonzero-subnormal operands and side-room subnormal deltas/spaces, preserves f64 underflow-to-zero, and requires explicit x64. M3a remains fixed-lane; no full M3b/GPU claim.

ACTIVE NEXT: supplied-target moving-lane field integration per docs/MILESTONE_3B_INTEGRATION.md. Terra5.6 codes; Luna6 reviews; inspect Herdr live state before duplicate prompts. Preserve all51acceptedtests and pinnedread-onlyRust. Automatic target selection/pacing is excluded.

2026-11-17 ACTIVE: standalone side-room, visibility and overtaking helper milestone. Plan: docs/MILESTONE_3B_QUERIES.md. Accepted baseline: 43 tests. Terra 5.6 implements; Luna 6 reviews. Moving-lane field integration follows helper acceptance; preserve source mixed f32/f64.

# Current restart checkpoint: M3b primitives accepted

2026-11-16: M3b sort/overlap and standalone supplied-target MoveState accepted by Luna 6 after Terra 5.6 implementation. Coordinator full suite: 43 passed in 83.75s; Luna focused lane suite: 9 passed in 29.37s. Independently rebuilt 13-case Rust movement oracle reproduces the frozen fixture byte-for-byte; extracted source blocks are unchanged, all 30 manifest hashes match, and the 700-vector copied fixture matches the Rust original. Rust checkout remains clean at 58da7bc4693d9aa8f6340e348cba1c5b5fb5a856.

Implemented: exact capacity-32 comparison-sort permutations and ordered overlap correction; standalone MoveState with lane, target, speed, continuation time/direction, checked scalar normalization and integer bounds. Full-state eager/JIT vectors cover blocked momentum, reversals, zero target/speed, continuation, after-goal movement and late rank. A subnormal-safe approximately comparison preserves the Rust zero threshold under CPU XLA.

Remaining: side_room/is_visible and overtaking helpers, supplied-target moving-lane field integration, automatic TargetState/pacing, M3c and later milestones. M3a field remains fixed-lane. Standalone after-goal movement does not enable post-goal field scheduling. Full M3b, full simulation parity and GPU correctness/performance remain unvalidated.

Next task: implement source-backed side_room/is_visible and overtaking predicates with independent Rust boundary fixtures, then supplied-target moving-lane integration. Read docs/MILESTONE_3B.md and the final decision plus integration criteria in docs/REVIEW_M3B.md. Integration must retain tick-entry side permissions, distinguish the front-blocker snapshot, apply lateral movement before each runner longitudinal update, honor full/partial gate behavior, resolve overlap after all movement using post-longitudinal positions, finish the crossing tick, and freeze every state leaf thereafter. Automatic target selection/pacing is a separate unimplemented system.

Project: C:/Users/a/PycharmProjects/GlobalSim-Jax. Reference Rust: C:/Users/a/RustroverProjects/GlobalSim (read-only). Preserve IDE/venv/main.py. Run .venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider from project root. Verify pinned revision and manifest hashes before regenerating expected fixtures; generators do not enforce the revision automatically. Movement oracle: rustc --edition=2021 -Awarnings -O tools/lane_move_oracle.rs -o <temporary executable>; capture bytes using Python subprocess, not PowerShell redirection.

Herdr agents: jax-terra wC:p3 uses gpt-5.6-terra; jax-luna wC:p5 uses gpt-6-luna. Inspect live state before prompting or launching duplicates. Last turns completed; usage budgets were nearly exhausted, so inspect for a limit message before assuming an agent can continue. Do not substitute a different model without user direction. HERDR_ENV must be 1; elevated Herdr server access uses tool escalation. Never answer an agent approval dialog on its behalf. Root owns coordination, source/provenance checks and final validation.

## Historical checkpoints

The following entries are historical and may describe resolved defects or superseded scope.

# Active M3b deterministic lateral work (2026-11-16)
M3a accepted with34 baseline tests. New work is deterministic lateral mechanics: exact retail comparison sort and overlaps, then MoveState/side-room/visibility/overtaking primitives and supplied-target field integration. Automatic lane TargetState, stochastic pacing/pacemaker selection and broader M3c remain unported; do not claim fullM3b or fullM3 complete from deterministic primitives alone. Terra5.6 jax-terra wC:p3 codes; Luna6 jax-luna wC:p5 reviews. Current task: exact comparison sort and overlap resolver up to32 active rows, including sourceequalkeypermutations above16. Require unchanged originalRust modules/oracle fixtures, exact eager/JIT/batch evidence. Read docs/MILESTONE_3B.md and REVIEW_M3B.md. Existing lane.py scaffold is notaccepted until its fullsort path and tests pass. Preserve sourceRust tree and existingM3a API. Before regenerating fixtures verify pinned58da7bc4693d9aa8f6340e348cba1c5b5fb5a856/sourcehashes. Use explicit destinationworkdir C:/Users/a/PycharmProjects/GlobalSim-Jax. Inspect agentstate before duplicateprompts afterrestart.

---
# Current M3b primitive checkpoint (2026-11-16)

The first M3b primitive task is implemented and pending Luna review:
`lane.py` contains the exact 32-entry source comparison introsort and ordered
overlap manager pass; checked wrappers reject unsupported values and capacity.
`tools/lane_oracle.rs` freezes independent Rust outputs in
`tests/fixtures/lane_oracle.json`, with hashes in `ORACLE_PROVENANCE.json`.
Focused eager/JIT/batch validation is `tests/test_lane_primitives.py` (4
passed). Next is MoveState, side-room/visibility and overtaking; automatic
targeting, pacing, M3c, full M3 and GPU claims remain unsupported.

MoveState standalone unit is now ready for review: its 13 full-state cases
come from `tools/lane_move_oracle.rs`, a mechanical extraction of the pinned
source routine plus pinned physics module. `tests/test_lane_move_full.py`
asserts every state field eager and jitted-vmap; checked scalar normalization,
int32 bounds, direction flags, and the subnormal approximately floor are
covered. Focused lane primitive/move/full-state run: 9 passed in 22.85s.
Moving-field integration and the remaining M3b helpers are still deferred.

---
# Current restart summary: M3a accepted (2026-11-16)
Final coordinator full suite: 34 passed in 55.69s on current tests. Field demo runs; Rust oracles reproduce fixtures and reference checkout is unchanged.
Luna6 accepted the bounded ranks/blocking/ordered fixed-lane field portion after Terra5.6 implementation and corrections. M3a is complete for this scope; milestone3 as a whole is NOT complete. Code includes one-based rank/tie queries, front/side snapshot queries, live ordered blocker-speed cap reads, sequential stored minima, f64 timers, checked host APIs, explicit prior minima for nonfresh attachment, batch-local terminal validation and whole-state terminal freezing. Integrated batches complete a crossing tick then freeze before post-goal updates; they also stop on the wrapper tick cap. No skills/pacing/lane movement/geometry integration or RNG consumption is included.
Evidence: three independently generated Rust cases, eight ticks/three runners/19 channels, eager/JIT scans, two-race batch independence, an actual ordered-cap counterexample, and independent f32 custom-parameter vector. Strengthened finish/timeout test uses two interacting runners and nonzero timers and checks every state leaf byte-for-byte in later calls/padded rows. Original Rust checkout remains clean at pinned commit58da7bc4693d9aa8f6340e348cba1c5b5fb5a856. Coordinator rebuilt both field Rust oracles and reproduced frozen JSON; all recorded manifest hashes match. See REVIEW_M3.md final decision and MILESTONE_3.md exact contract.
Next work: M3b pacing, lane selection/overlap and overtaking; M3c rushing/competition/spurt/shared RNG; M4 skills; then GPU validation. Geometry remains standalone. Do not claim full milestone3, full game parity or GPU acceleration. Before regenerating fixtures manually verify the pinned commit/source hashes; generator revision checking is not automatic.
Run `.venv/Scripts/python.exe -m globalsim_jax.cli field-demo --ticks 16` from project root. Agents remain in Herdr: jax-terra wC:p3 (gpt-5.6-terra), jax-luna wC:p5 (gpt-6-luna); inspect live state before reuse. Destination C:/Users/a/PycharmProjects/GlobalSim-Jax; Rust reference is read-only. Use explicit workdirs. Prior entries below are historical checkpoints and may describe resolved defects.

---
# Active milestone 3a checkpoint (2026-11-16)
M3a implemented; final Luna review pending. Checked fixed-lane field profile, three frozen Rust 8-tick/3-runner/19-channel cases, query/custom-parameter fixtures, eager/JIT scans, two-batch scalar independence, ordered-cap counterexample, whole-state terminal freeze and `field-demo` are complete. Coordinator validation: 34 passed. Exact contract and hashes: `docs/MILESTONE_3.md`, `docs/ORACLE_PROVENANCE.json`. M3b pacing/lane/overlap/overtaking and M3c rushing/competition/spurt/shared RNG remain; whole M3 is not complete. Rust stays pinned/read-only at 58da7bc4693d9aa8f6340e348cba1c5b5fb5a856.

---
# Active milestone 2 checkpoint (2026-11-16)
M3a current work: `globalsim_jax.field` now has one-based rank/tie queries, exact fixed-lane front/side snapshots, f64 blocking timers, ordered stored-minimum preparation, and an integrated `lax.fori_loop` movement pass that reads snapshot blocker ID/gap but live carry speed for the cap. It freezes a batch only after a crossing pass, not as post-goal retail parity. `tests/test_field.py` currently has 5 passing checks; `tools/field_oracle.rs`/`field_query_trace.json` cover only query evidence. Remaining M3a gate before acceptance: source-extracted ordered tick oracle with multi-tick, multi-batch live-speed/minimum traces; checked field host API; CLI field demo; full suite and independent review. Do not claim M3a accepted.

CURRENT M2 checkpoint: accepted narrowly by Luna final review. Geometry is accepted for the separate bundled course10101/1200m component; it is not loop-integrated. The flat/no-field loop uses `initialize`, immutable `RestrictedConfig`, and `run_checked` as the host boundary, while `step`/`run` are prevalidated JIT kernels. Six frozen eight-tick Rust-oracle cases cover delayed non-unit ground, phase crossing, overshoot finish, exact-goal field stop, dash exit, and HP-zero deceleration. Nine channels are exact-compared eager/JIT; a divergent `[2,2]` batch/runner check compares every field against scalar execution. Source/oracle/fixture hashes are in `docs/ORACLE_PROVENANCE.json`. Coordinator final validation: 17 passed in 27.86s; `restricted-demo --ticks 1200` ended at `1200.084961`. The generator has no runtime revision enforcement; regeneration requires checking the pinned revision and manifest hashes first. No full race, field, skills, geometry integration, GPU, or performance parity claim.

---
# Current restart summary (supersedes historical checkpoints below)
2026-11-16: M1 accepted after Terra 5.6 implementation and Luna 6 independent review. Coordinator final test: 8 passed in 17.59s, no expected failures. Installed CLI benchmark works. Literal f32 HP division with numerator/denominator/quotient barriers matches fixture assertions in eager, standalone JIT and composed JIT. Earlier HP blocker and install errors below are resolved history. Editable install succeeds; JAX floor is 0.4.36, tested JAX/JAXlib 0.12.0.1 on Windows CPU / Python 3.13.7. Import does not mutate x64 configuration. Fixtures, default parameters and licensing verified byte-for-byte against pinned reference. Rust source checkout is clean.

Read docs/PORT_PLAN.md, docs/STATUS.md, docs/SUPPORTED_FUNCTIONS.md and the final decision in docs/REVIEW.md. No full race API or GPU validation exists. Next implementation is M2 restricted ordered race loop with initialization/stat conversion and geometry; GPU evaluation requires WSL2/Linux CUDA setup and independent fixture verification. Do not infer full-race speedups from the RNG primitive benchmark.

Herdr agents remain available: jax-terra wC:p3 (gpt-5.6-terra) and jax-luna wC:p5 (gpt-6-luna); inspect live state before reuse. Source-of-truth project path C:/Users/a/PycharmProjects/GlobalSim-Jax; Rust reference read-only C:/Users/a/RustroverProjects/GlobalSim at 58da7bc4693d9aa8f6340e348cba1c5b5fb5a856. Explicit workdir on all tool calls. Keep main.py/.idea intact. The existing .venv now contains project dependencies and editable package.

---
# Handoff

## Target update implementer handoff (2026-11-17)

Candidate is ready for Luna's independent review. The fixed-capacity unsorted
cache profile deliberately excludes `sorted_owner`; checked calls require
canonical padding and reject nonzero f32 subnormal scalar/pairwise/predicate
domains, including the independently source-confirmed blocker-gap diagnostic.
Stale nonnegative int32 target indices are accepted for the source drop pass.
Root's frozen source oracle supplies 8 supported four-step cases and 1 diagnostic.
Terra focused validation:
`.venv\\Scripts\\python.exe -m pytest tests\\test_lane_target_update.py tests\\test_lane_target.py -q -p no:cacheprovider`
=> **17 passed in 22.00s**. It compares every state/cache leaf by bits in eager,
jitted scan and checked paths, selected vmap/scalar isolation and boundary
rejections. Root owns remaining independent review/full-suite gates; no full
simulation, automatic target selection, RNG, pacing, field integration or GPU
claim is authorized.
Start with PORT_PLAN.md and STATUS.md. Current task: implement and review milestone 1. Launch gpt-5.6-terra via Herdr for coding; gpt-6-luna for review. Local Codex models_cache.json lists both. Parent native collaboration tool cannot launch Terra 5.6; use Herdr CLI. HERDR_ENV=1; Herdr server is elevated, so shell commands accessing it require escalation. Never bypass blocked approval UI. Reference fixtures under tests/fixtures include retail_rng_dotnet.json, retail_physics_port.json and retail_finish_crossings.json. Destination contains only IDE, venv and starter main.py before plan creation. No full-race backend yet.

## Live orchestration (2026-11-16)
- Herdr workspace wC, tab wC:t1. Coordinator pane wC:p1; coder jax-terra pane wC:p3 (GPT-5.6-Terra); reviewer jax-luna pane wC:p5 (GPT-6-Luna).
- Launch through `herdr pane run ... "codex -m ... -s workspace-write -a on-request -C ..."`. Herdr agent start failed because local codex is a PowerShell function, which its Start-Process launcher cannot execute. Raw pane run invoked the existing wrapper correctly. Both model banners verified.
- Terra was assigned M1 implementation; Luna assigned preliminary review and later final implementation review. Both warned to use explicit destination workdirs/absolute edit paths because initial tools appeared to inherit source cwd.
- Venv Python 3.13.7 initially lacks jax/jaxlib/pytest. GPU NVIDIA RTX 3060, driver 610.88. WSL2 Ubuntu installed, stopped; GPU execution not validated.
- Read status using `herdr agent get jax-terra`, `herdr agent read jax-terra --source visible` (working agents cannot expose long scrollback). Prompt without --wait queues steering; inspect state before retrying. Elevated Herdr access requires normal tool escalation.
- Next coordinator gate: collect Terra validation, ask Luna for final code review, send concrete findings to Terra, rerun tests, update all handoff evidence. Do not stop at agent launch and claim implementation done.

## Coordinator checkpoint: numerical review in progress
Historical checkpoint (resolved): an early CPU-XLA HP-drain JIT result differed by one ULP. The final implementation retains literal f32 division and places barriers around its numerator, denominator, and quotient; exact eager, standalone-JIT, and composed-JIT fixture validation passes. CPU JAX 0.12.0.1 is available; no GPU backend is detected. Source git status remained clean.

## Terra implementation update (2026-11-16)
- M1 source, independent copied fixtures, README, CLI and inventory are present. The source-faithful f32 HP quotient fix (barriers around numerator, denominator, and quotient) removed the CPU-XLA one-ULP mismatch. Validation is now `.\\.venv\\Scripts\\python.exe -m pytest tests -q -p no:cacheprovider` => **8 passed** with exact eager, standalone-JIT, and composed-JIT copied-fixture assertions on CPU. This does not establish full-race or GPU parity.
- CPU JAX is 0.12.0.1 on `CpuDevice(id=0)`; CUDA/WSL remains deferred. The benchmark is results-only/synchronized primitive measurement. Coordinator subsequently verified `pip install -e .[dev]` succeeds; the earlier sandbox temp-directory error is historical only.
- Final reviewer should verify fixture provenance, exact operation order, the resolved HP quotient regression, and package install in a normal environment.

## M3b reviewer checkpoint (2026-11-16)
Preliminary deterministic lateral mechanics criteria are in `docs/REVIEW_M3B.md`. Scope covers exact comparison-sort ordering, lane overlap resolution, visibility/side-room and overtaking predicates, and supplied-target `MoveState` integration with independent Rust vectors plus eager/JIT/batch checks. No implementation edits or M3b acceptance occurred in this pass. Automatic target selection and pacing remain outside this bounded slice. M3a remains accepted; full M3 and GPU claims remain unsupported.

Bounded in-progress snapshot review found the new `lane.py` checked wrappers reference missing `np` import, unchecked int32 key conversion, and no post-f32-narrowing finite check in overlap wrapper; see current findings in `docs/REVIEW_M3B.md`. Terra is editing, so these findings and exact-ID fixture test coverage must be rechecked after changes. No tests were run in this snapshot.

Stable first-unit follow-up supersedes that intermediate note: `tests/test_lane_primitives.py` passed 4/4 in 15.99s; exact Rust ID orders and active overlap outputs checked in eager/JIT and selected batched cases. Kernel behavior is accepted within the tested capacity-32 vectors. Remaining review notes: checked lane cast bound at f32 `2**31` and explicit preservation test for nonempty padded tails. No full suite or MoveState review yet; automatic targeting/pacing remain excluded.

Current bounded MoveState snapshot review is recorded in `docs/REVIEW_M3B.md`: checked scalar leaves need normalization (or stricter rejection), the approximately-zero test must precede inward scaling as Rust does, and copied vectors currently omit continuation/direction whole-state checks for blocked/reversed/after-goal branches. No tests were run during Terra's draft; this unit remains unaccepted.

Read-only integration design criteria appended to `docs/REVIEW_M3B.md`: tick-entry side permissions remain frozen; lateral `MoveState` is interleaved per runner before its longitudinal update; fully closed gates return before lateral work, partial gate openings use full dt laterally but fractional longitudinal distance; overlap runs once on post-movement rows; complete crossing tick then preserve M3a all-leaf terminal freeze. Full manager compilation is not required for the extracted oracle, but its copied-source scope must be explicit and must not be described as full-engine parity. No tests were run.

Latest bounded lane review: `.venv\\Scripts\\python.exe -m pytest tests/test_lane_primitives.py tests/test_lane_move.py -p no:cacheprovider -q` => 6 passed in 20.75s. Sort/overlap behavior accepted for tested vectors; +2^31 rejects and manual nonempty padding probe preserves tails, but automated tail assertion is still needed. MoveState kernel now canonicalizes checked leaves and checks zero speed before inward scaling; do not accept until the extracted-source oracle compiles, its generated full-state fixture is tested eagerly/JIT, and checked API scalar cases are covered. One harness format placeholder mismatch was observed and Terra is fixing it. No full suite was run.

2026-11-17 query milestone review criteria: see `docs/REVIEW_M3B_QUERIES.md` for exact source order/precision, checked-domain boundaries, sentinels and independent oracle coverage for standalone side-room, visibility and overtaking helpers. This was read-only criteria work; no implementation review or tests. Await a stable Terra snapshot for final review. Field integration, automatic targets/pacing and GPU remain outside this milestone.

Bounded static review of the initial query wrappers found no post-f32-narrowing finiteness checks for scalar side-room/visibility/overtaking inputs; fix advice is in `docs/REVIEW_M3B_QUERIES.md`. Public Boolean flag type validation and source query operation order inspected correctly. Tests remain pending the expanded Rust fixture; no tests were run here.

Domain follow-up records coordinator-confirmed XLA subnormal-flush divergences for overtaking and visibility. Post-cast finite checks alone are insufficient: choose source-faithful subnormal arithmetic or reject nonzero f32 subnormal operands and relevant side-room/visibility/overtaking intermediates at checked boundaries, while keeping f64 values that narrow fully to signed zero valid. See `docs/REVIEW_M3B_QUERIES.md`; no tests were run by Luna.

Final bounded query review: focused tests `test_lane_queries.py` + `test_lane_query_checked.py` passed 6/6 in 1.84s. Low-level source algorithms accepted within tested values and explicit checked subnormal/overflow domain. Do not mark checked API milestone complete until bool peer-row rejection, positive checked API fixture assertions, subnormal-delta and finite-overflow visibility tests, target-comparison f64-narrowing witness, and oracle/fixture manifest hashes are recorded. No full-suite run or M3b/full-field/GPU claim.

2026-11-17 final bounded query review supersedes the pending entry above: Luna accepted the standalone side-room, visibility and public overtaking query helpers within the documented finite/non-subnormal f32 checked domain. Focused command `.venv\Scripts\python.exe -m pytest tests/test_lane_queries.py tests/test_lane_query_checked.py -p no:cacheprovider -q` passed 8 tests in 2.80s. Interim checked-wrapper and fixture follow-ups are resolved; current fixture has 20/12/26 supported rows and 2 unsupported diagnostics, with oracle/fixture provenance recorded. See `docs/REVIEW_M3B_QUERIES.md`. Field integration, remaining M3b/M3c features, full simulation parity and GPU claims remain outside this acceptance.

2026-11-17: Luna read-only design review for supplied-target moving-field integration is recorded in `docs/REVIEW_M3B_INTEGRATION.md`. Criteria cover tick-entry rank and side-cache observations, stable owner-excluded roster order, closed-gate return timing, ordered live-roster refresh, post-movement overlap, and the distinction between M3a batch truncation and retail post-goal behavior. Plan is feasible under stated bounded assumptions, with variable lane-width configurations and active batches containing already-finished runners requiring explicit API boundaries. No tests or code review were performed; implementation and source-extracted subset fixtures remain pending.

2026-11-17 core-step review correction: Luna re-traced `prepare_blocking` dataflow and withdrew the minimum-prepass finding as a false positive. The `fori_loop` starts with prior stored minima, reads the current carry at each blocker index, and updates only the current owner; later-index blockers therefore remain old while earlier-index blockers are refreshed, matching source order. `moving_field.step` consumes each owner's own computed minimum and gathers live blocker speed from the evolving runner carry. No actionable core-step bug was found in this static snapshot. No tests were run; final independent oracle/test review remains pending.

2026-11-17 oracle/host snapshot update: Luna inspected `tools/moving_field_oracle.rs`, its extraction note and fixture metadata. Pinned Rust helper modules, source hash manifest entries, nine 8-tick/3-runner cases, and three correct-vs-wrong snapshot witnesses are consistent with the documented restricted Rust adapter; they do not claim full-engine parity. One checked-boundary issue remains: `moving_field.run_checked` validates finiteness but accepts f32 subnormals and derived subnormal side-room gaps before calling low-level `side_room`. See `docs/REVIEW_M3B_INTEGRATION.md`. No tests were run during edits; Terra is still adding differential tests, and final review remains pending.






2026-11-17 final bounded M3b supplied-target moving-field review: accepted by Luna for the constant-width, explicit-target, no-skills/no-pacing profile under the checked finite/non-subnormal domain and M3a whole-batch truncation policy. Luna independently ran the three checked-boundary tests plus witness test: 4 passed in 10.23s. Terra reported the full 9-case eager/JIT trace pass in 153.36s; root full suite was still running. Current checked validation rejects subnormal state/derived side differences and bad targets; oracle/fixture hashes match manifest. See docs/REVIEW_M3B_INTEGRATION.md. Remaining M3b/M3c, automatic targeting/pacing, full engine parity and GPU remain unsupported.
