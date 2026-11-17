# M3a independent review â€” accepted within bounded scope

**Decision (2026-11-16): accept M3a** for the extracted, fixed-lane, receiver-free blocking profile layered on M2. This decision is not acceptance of all M3 or full race parity. Earlier snapshot findings in this review thread were interim; the final checks below supersede them.

## Evidence reviewed

- `globalsim_jax/field.py` implements one-based ranks with stable lower-index ties; pre-movement front/side snapshots; front inclusion and side exclusion of finishers; f32-stored block parameters; f64 blocking timers; and ordered fixed-lane movement. Blocker index/gap come from the pre-movement snapshot while the blocker speed is read live in runner order.
- Three independently copied-source Rust tick cases (`higher_blocker`, `lower_blocker_delayed`, `closed_gate`) each cover 8 ticks Ã— 3 runners Ã— 19 channels. Tests compare the frozen traces in eager and JIT scan execution, compare two independent batches against scalar runs, and include a negative control proving the updated lower-index blocker speed changes the later runner cap. Rank output is explicitly labeled `pre_movement`; it is not represented as the engine's post-movement published rank.
- Query fixtures cover rank/tie and front/side boundaries. A separate custom-parameter Rust vector distinguishes f32 storage before widening from unrounded host arithmetic; eager and JIT both match it.
- `test_whole_field_state_is_bitwise_frozen_after_finish_and_alltimeout` now uses two runners with a one-meter gap and lane offsets 0/0.02. It asserts front, side and all-block counters become nonzero before checking every `FieldState` leaf byte-for-byte after a later step and three padded scan rows. The all-timeout case starts with nonzero f64 timers. This addresses the prior one-runner/no-counter coverage weakness.
- Checked API fixes verified in the current implementation: fresh attach versus explicit prior-minimum history, nonnegative minimum validation, all-timeout attach truncation, per-batch post-goal validation, query x64/shape/dtype/finiteness guards, and f32 rounding of custom block parameters.
- Final validation evidence: reviewer full suite was 34 passed in 53.10s before the final test strengthening. Coordinator then ran the strengthened two-runner terminal regression independently (1 passed in 3.94s), followed by the CURRENT full suite (34 passed in 55.69s). Both Rust field oracles were rebuilt from current adapter source and reproduced the frozen fixtures. These results supersede the intermediate test snapshots above.

## Accepted contract and limits

The fixed-lane profile assumes no skills, callbacks, or current-speed receivers mutate peer state between ordered runner updates. Under that contract, observed speed equals current speed, and the source cap/floor projection is equivalent for the supported input range. Finished lanes are frozen after the crossing pass; this does not implement retail post-goal movement, minimum, deceleration, or HP policy. The rank value emitted by the bounded step is pre-movement telemetry.

M3b pacing, lane movement/selection, overlap/overtaking and M3c rushing, competition, spurt, shared RNG and skills remain **out of scope and unaccepted**. No full-engine, GPU, or speedup claim follows from this review.

## Provenance precondition

Before regenerating expected fixtures, verify the pinned Rust revision and hashes recorded in `docs/ORACLE_PROVENANCE.json`. The generators do not enforce the reference revision at runtime. This is a regeneration/handoff precondition, not a blocking implementation defect; fixture values are versioned regression evidence for the extracted subset, not universal official game parity.

