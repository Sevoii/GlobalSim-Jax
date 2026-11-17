# Milestone 1 review: criteria and plan risks

Current bounded cache-collection review: [REVIEW_M3B_TARGET_CACHE.md](REVIEW_M3B_TARGET_CACHE.md). Luna accepted the standalone `TargetCache` snapshot after 4 focused tests passed; root reports the full suite passed (62) and the oracle rebuild was raw-byte identical with all 40 provenance hashes verified. Lifecycle/attachment, target updates, PassingSearch, RNG, pacing, and integration remain deferred. Full-suite and provenance results are coordinator-reported, not independently rerun by Luna.

Current persistent target-update review: [REVIEW_M3B_TARGET_UPDATE.md](REVIEW_M3B_TARGET_UPDATE.md). Luna accepted the bounded default-feature update after an independent focused run (17 passed in 26.75s); root reports the coordinator full suite passed (75), all 45 manifest hashes matched, and the oracle rebuild was raw-byte identical with exact source extraction and a clean pinned Rust checkout. Coordinator results are not independently rerun by Luna. PassingSearch, final target selection/shared RNG, attachment/lifecycle, integration, pacing, experimental gap refresh, full M3b, full simulation parity, and GPU claims remain out of scope.

Reviewer: GPT-6-Luna  
Status: preliminary plan/reference review; implementation review pending coordinator request.  
Reference revision: `58da7bc4693d9aa8f6340e348cba1c5b5fb5a856` in `C:/Users/a/RustroverProjects/GlobalSim` (read only).

## Evidence and scope

The reference RNG is `src/retail_random.rs`. The retail numerical functions are in `src/retail_physics.rs`; the plan's phrase “bounded stat/root and feasible core parameter-driven speed/HP/acceleration helpers” needs a precise exported inventory before M1 can be called done. Useful independent regression files in the reference checkout are `tests/fixtures/retail_rng_dotnet.json`, `tests/fixtures/retail_physics_port.json`, and `tests/fixtures/retail_finish_crossings.json`.

The RNG fixture contains rows for seeds `0`, `1`, `-1`, `953795537`, `i32::MAX`, and `i32::MIN`, with `integers`, `doubles`, and `wide` sequences. Physics fixture `expected_bits` stores exact f32 bit patterns and has a `provenance` field: selected outputs were regenerated from the GlobalSim engine under v75 rules; inputs came from the reconstructed Python port; the fixture itself says it is a regression snapshot and not all Python-port rounding. Finish fixture rows carry source hashes, crossing inputs, expected times and `adjacent_ticks`; its scope explicitly excludes two thinned crossing intervals from adjacent-tick interpolation validation. Preserve these distinctions when copying and describing fixtures: expected values are versioned regression evidence, not universal official game parity.

At this preliminary pass, the destination has no `tests/` directory or fixture copies. The plan requires reuse of the Rust fixtures; implementation review must verify the copied bytes/schema and provenance are intact, or document any deliberate transformation and why it remains independent.

## Concrete implementation review criteria

### RNG arithmetic and draw stream

- Match Rust `RetailRandom::new` exactly for all signed 32-bit seeds, especially `i32::MIN`: absolute-value special handling, wrapping subtraction, conditional addition of `i32::MAX`, and the 4×55 state-mixing loops must preserve signed i32 semantics. Do not rely on JAX's default integer width or unsigned overflow behavior accidentally matching.
- Match `next_int` index cycling, signed wrapping subtraction, `i32::MAX` decrement, negative correction, state replacement, and draw count. Test long sequences across index wrap, not only a short prefix.
- Match `next_double` as f64 division by `i32::MAX`; conversion/division order must be explicit. Test exact fixture doubles.
- Match `integer(lower, upper)` bounds checking and width calculation in i64. The ordinary path consumes exactly one draw even at zero width. The wide path consumes two draws and uses Rust signed remainder/parity semantics for negative values. Verify endpoints, `i32` extremes, and following RNG state after each branch.
- `real` and `real_distance` both consume one draw, but have deliberately different f32/f64 operation sequences. Confirm they remain distinct, clamp semantics for reversed/equal bounds, and exact following stream position. Do not infer one wrapper's semantics from the other's comment.
- Conditional and batched RNG calls must preserve per-lane state and draw consumption. A vectorized mask must not advance inactive lanes or cross-contaminate streams. Compare each batch lane against a separately executed scalar stream after mixed branch patterns.

### Numerical behavior and JAX execution

- Translate reference operation order by dtype, not just formula. In `retail_physics.rs`, several functions intentionally use f32 intermediates, f64 intermediates, and a single f32 store; the physics fixture checks exact f32 bits. Review each implemented operation against source order and the declared exact/tolerance policy.
- Check JAX/XLA fusion and reassociation under eager and `jit` paths. If exact reference rounding is required, use explicit casts/rounding barriers where necessary and demonstrate the same results in both modes. Float64 support must be enabled by the caller before importing/running the package; package import must not mutate `jax_enable_x64` or other global JAX config.
- Validate shape/dtype/range inputs on public entry points. Invalid ordered bounds, unsupported strategies/phases, nonfinite physical parameters, invalid denominators, and unsupported batch shapes should fail clearly rather than silently clamp, wrap, or approximate. Preserve reference clamping only where the Rust helper explicitly does so.
- For `phase_boundaries`, `target_speed_indices`, `advance_start_delay`, and `finish_time`, compare boundary-adjacent values and exceptional inputs to the Rust behavior. In particular preserve the direct f64 quotient in `target_speed_indices`; start-delay zero-time guard; and f64 interpolation ratio followed by f32 return in `finish_time`.
- Keep RNG state a valid explicit pytree for `jit`/`vmap`/batch APIs. Confirm scalar and batch APIs have documented shapes and that jitted calls execute rather than only trace. Batch-vs-scalar tests must use independent initial states and cover divergent draw counts.
- No implicit import of global race/course configuration or unrelated data. Parameterized helpers should accept explicit parameters; missing/incomplete parameter data should raise an actionable error.

### Scope, fixtures, benchmark, and claims

- Publish an exact M1 supported-function inventory and mark unsupported features. In particular, do not describe M1 as a full race, simulator parity, full field behavior, or full-race GPU backend. Do not add a public race API before the stated later milestones.
- Fixture validation must load copied fixture files as test inputs and compare outputs to stored expected values; do not generate expected values from the implementation under test. Preserve source revision, hashes, provenance fields, and exclusions. Treat exact-bit checks separately from tolerance-based checks.
- Benchmark only implemented numerical primitives, disclose backend/device, batch sizes, warmup, compilation handling, and synchronized timings (`block_until_ready`). Label results as primitive throughput. CPU measurements do not establish GPU behavior or speedup. GPU discovery is an environment check, not a performance result.
- Preserve AGPL-3.0-only terms and reference/data attribution in copied license and NOTICE material. Keep IDE and `.venv` files untouched.

## Preliminary plan risks / clarifications for implementer

1. **The “feasible core” is underspecified.** The plan should be backed by an exact inventory tied to reference functions. The Rust physics file includes more helpers than the compact list implies; implementing selected formulas without validation or parameter provenance risks accidental invented behavior. Narrowing the supported set is acceptable if explicit.
2. **Reference fixture claims need careful wording.** The physics fixture provenance explicitly describes a versioned mixed-source regression snapshot, while finish data is filtered to adjacent crossings. They are valuable independent checks but do not prove universal retail/game parity. The RNG fixture's .NET vectors are much stronger for RNG sequence verification, but still cover only listed seeds and calls.
3. **JAX integer support is an implementation hazard.** Seed/state values are signed i32 while some setup and wide-range calculations need wider intermediates. Make wrap/truncation points explicit and test with JIT enabled, including negative modulo behavior.
4. **JAX float behavior may differ across eager/JIT/backend.** Exact f32 fixture bits can be affected by fusion, reassociation and hardware math. State which outputs require exactness, which permit tolerance, and test the same contract under each supported execution mode. Enabling x64 alone does not guarantee matching mixed-precision sequencing.
5. **`real_distance` scope is ambiguous.** Rust marks it an isolated hypothesis, not a general validated runtime consumer. Keep it clearly experimental/qualified and test it independently; do not imply the whole retail draw order is established.
6. **Fixture arrival is currently a gate.** No fixture files are present in the destination at this preliminary review. M1 review should not accept locally regenerated expected data as a substitute for the pinned reference fixtures.

## Review status

The preliminary plan review is complete. Snapshot findings are recorded below; the coordinator's requested final implementation review and independent test pass are still pending. Final findings should include destination file paths and line numbers, severity, impact, and a concrete correction; record independent validation commands/results here without modifying implementation, `STATUS.md`, or `HANDOFF.md`.

## Historical snapshot review: first implementation files

This section preserves the initial findings and is superseded by their final dispositions below.

Reviewed `globalsim_jax/rng.py`, `globalsim_jax/physics.py`, and `globalsim_jax/precision.py` from the destination snapshot on 2026-11-16. These findings are actionable but provisional until Terra's tests and the coordinator's final-review request. No implementation files were edited and no tests were run while the test suite is being added.

### Findings

1. **High - `minimum_speed` changes a required f32 rounding point.** In `globalsim_jax/physics.py:44`, `course_speed(distance)` and `minimum[0]` are widened to f64 and multiplied there. Rust first calls `start_dash_speed`, whose multiplication is f32 and stored as f32 (`src/retail_physics.rs:325-326`), then widens that rounded result for the later sum (`:329-334`). This can change output bits at boundary inputs. Compute the f32 start-dash product first, then widen that value for the f64 sum; add a fixture/boundary case where the f32 product differs from the f64 product rounded once at the final store.

2. **High - `blocking_speed_cap` subtracts coefficients in f32.** In `globalsim_jax/physics.py:46`, `(self.blocking[2] - self.blocking[1])` is evaluated before either coefficient is widened. Rust widens both coefficients before subtraction (`src/retail_physics.rs:337-342`). Widen each coefficient before subtracting, then retain the stated f64 multiply/add and f32 rate store. Add an exact-bit case using coefficients whose f32 difference loses precision in the subsequent interpolation.

3. **High - acceleration modifiers are reduced instead of applied in order.** In `globalsim_jax/physics.py:55`, `jnp.sum(f32(modifiers))` does not implement Rust's left-to-right fold (`src/retail_physics.rs:491-495`, called by `acceleration_from_base_iter` at `:476-481`). Reduction trees can reassociate additions and change f32 bits. Apply modifiers sequentially in their input order, with a JAX-compatible fixed-shape loop/scan for compiled batches. Include the reference's cancellation vectors `base=1`, modifiers `[16777216, -16777216]` versus `[-16777216, 16777216]` and verify eager/JIT behavior.

4. **High - `advance_start_delay` uses the wrong remaining value when the input countdown is already nonpositive.** In `globalsim_jax/physics.py:19-22`, `after = remaining - dt` is used to calculate the fraction unconditionally. Rust skips subtraction when incoming `remaining <= 0` and returns `-remaining/dt` for negative remaining, or `1` at zero (`src/retail_physics.rs:109-122`). For example `remaining=-0.02, dt=0.0666` should yield approximately `0.3003`, not `1.3003`. Select the countdown value according to the Rust branch before calculating the fraction; add negative, zero, exact-crossing, and still-positive cases.

5. **Moderate - inactive zero-time delay output can be non-finite.** The Rust helper's `dt == 0` branch returns the unchanged countdown and no fraction (`src/retail_physics.rs:111-113`). The JAX API represents this as `active=False` plus a numeric fraction, but `physics.py:22` still divides by `dt`; negative remaining therefore produces infinity (and zero can produce NaN, depending on lowering). Ensure the inactive payload is a documented finite sentinel and test `dt=0` for positive, zero, and negative countdowns. Consumers must use the `active` mask rather than interpret that sentinel as a crossing fraction.

6. **Moderate - `integer` does not reject reversed bounds.** `globalsim_jax/rng.py:55-68` has no `lower <= upper` validation, while Rust asserts ordered bounds (`src/retail_random.rs:49-51`). Reversed bounds flow through the normal branch with negative width and produce a value instead of an error. Add a clear host-side validation path or a documented checked wrapper compatible with JIT, and cover equal bounds (which must still consume one draw) separately from reversed bounds.

### Other checks in this snapshot

- `precision.py:4-6` checks x64 when called and does not mutate JAX global configuration on import, which meets the no-silent-global-config-change criterion for this module.
- RNG initialization and next-int arithmetic use explicit int32 arrays; the fixture and JIT tests still need to establish signed-boundary behavior and wrapping as intended. `integer`'s wide-range branch also needs independent fixture and subsequent-state checks once tests land.
- `real` and `real_distance` are kept distinct in the code. Exact behavior for reversed bounds is not yet tested; Rust's sequential comparisons are not generally equivalent to `jnp.clip` when lower exceeds upper. Either reject reversed real bounds explicitly or preserve the Rust comparison behavior, and document the choice.
- `Parameters.default()` loads the packaged `global_parameters.json` only when explicitly called; this snapshot shows no import-time config mutation. Final review should verify data attribution/hash and clarify this default's provenance in package documentation.

### Validation status

No independent test command was run during this initial snapshot pass, as requested while Terra added tests. The numerical mismatches recorded above were subsequently rechecked and their disposition is recorded in the final review section below.

## Final milestone 1 review

Review snapshot: 2026-11-16, after Terra removed the HP strict-xfail and raised the declared JAX floor. The original six findings have been checked against the current source. This is an acceptance review of the narrow CPU numerical foundation only.

### Resolution of the six initial findings

| Finding | Current status | Evidence |
|---|---|---|
| `minimum_speed` widened the start-dash product before its f32 store | **Resolved** | `globalsim_jax/physics.py:45-46` calls `start_dash_speed` as f32, then widens its returned f32 for the sum. Physics fixture exact-bit checks pass. |
| `blocking_speed_cap` subtracted f32 coefficients before widening | **Resolved** | `physics.py:47-48` widens both coefficients before subtraction. Physics fixture exact-bit checks pass. |
| Acceleration modifiers used an unordered reduction | **Resolved** | `physics.py:80-84` folds modifiers in order; cancellation-order eager/JIT test passes. |
| Start-delay fraction used `remaining - dt` even when the incoming remainder was nonpositive | **Resolved** | `physics.py:17-24` only decrements positive remainder and computes from the selected post-branch value; negative, zero, crossing, and inactive-zero-time tests pass. |
| `dt == 0` could expose a nonfinite inactive fraction | **Resolved** | `physics.py:22-24` returns a finite zero payload for inactive lanes; zero-time tests pass. |
| Reversed integer bounds were not rejected | **Resolved for host/eager inputs; dynamic traced inputs are a documented caller precondition** | `rng.py:55-60` raises `ValueError` for concrete reversed bounds. A dynamic invalid pair passed through `jit` can still return a value; `docs/SUPPORTED_FUNCTIONS.md` explicitly excludes arbitrary dynamic invalid inputs from the low-level compatibility guarantee. A public checked wrapper can be added if dynamic validation becomes part of the API. |

The additional reversed-real-bound issue from the snapshot review is also resolved: `rng.py:76-84` uses Rust's ordered comparisons rather than `jnp.clip`, and `tests/test_rng.py` checks both wrappers.

### Validation and provenance evidence

- Independent final command: `.venv\\Scripts\\python.exe -m pytest -p no:cacheprovider tests` => **8 passed in 18.01s, no xfails**. Coordinator separately reported 8 passed in 20.80s. The suite checks all 140 fixture vectors in eager mode, standalone JIT, and the composed JIT tuple including HP drain (`tests/test_physics.py:15-35`).
- CLI benchmark command: `.venv\\Scripts\\python.exe -m globalsim_jax.cli benchmark --batch 8192 --repeats 100` => `primitive=rng.next_double`, CPU `cpu:cpu:0`, 0.496564s, 1,649,736 samples/s. It synchronizes each result and excludes compilation warmup. This is primitive-only CPU throughput, not race or GPU performance.
- Fixture/data hashes match the pinned source byte-for-byte: `retail_rng_dotnet.json` `53C0800AF32E3AB0B630216597799CFF87BB933A77AAC49DC1636A986773B958`; `retail_physics_port.json` `AEDF1C4FFE7D634ADE0E734DFA112F1BF27DC40CC73E601D688D183A96B7392B`; `retail_finish_crossings.json` `AB7570695D6A00481746F338FB7996EE9CA26FF732A806A18EC7905F190A1187`; packaged `global_parameters.json` `4655BD004AD7A9A46104283FD42E9AFB1C0E9835CCB1B86BE47940C100FE99E1`. `LICENSE` and `NOTICE` hashes also match the reference: `D8A6CC31ABC16B6748C7A21F21611F5A1EC33F67D22CA23D7DA1C19B95496BEE` and `005CC9E23ADBC2BF95D611A74784923ED7FEBE621466F1F0095B5D766CF50B15` respectively.
- `pyproject.toml:12` now requires `jax>=0.4.36`, matching the intended API floor. README and `docs/SUPPORTED_FUNCTIONS.md` state the x64 setup, experimental `real_distance`, unsupported race/GPU scope, and primitive benchmark limits. `__init__.py`, `rng.py`, and `physics.py` do not mutate JAX global config at import; x64 is checked when numerical APIs run. Coordinator reports editable installation succeeded under escalation; I did not repeat the install.

### Final disposition of the HP operation-order finding

**Resolved.** Current `globalsim_jax/physics.py:82-87` keeps numerator, denominator, and division in f32 with optimization barriers, then widens the stored f32 quotient for later f64 products. This follows Rust `src/retail_physics.rs:451-452`. The exact expression was first recorded above, then applied by Terra. The current suite passes fixture checks for the standalone and composed JIT paths; no f64 division approximation remains in the reviewed implementation.

#### Exact passing f32 barrier candidate

This is the sequence from the local eager/JIT fixture experiment. Keep numerator, denominator, and division in f32; widen only after the quotient has been stored as f32:

```python
squared = jax.lax.optimization_barrier(gap * gap)
denominator = jax.lax.optimization_barrier(self.hp_drain[1])
quotient = jax.lax.optimization_barrier(squared / denominator)
drained = jax.lax.optimization_barrier(
    quotient.astype(jnp.float64) * self.hp_drain[2].astype(jnp.float64)
)
```

Terra applied this candidate with barriers for the subsequent `rated`, `value`, and `factor` stores. The final independent run passed the exact eager, standalone-JIT, and composed-JIT fixture assertions, including HP drain.

### Review decision

**Accept the narrow milestone 1 CPU numerical foundation for the listed primitives.** The six initial findings are resolved within the stated API contract, the HP operation now follows the Rust f32 division sequence, and the composed JIT fixture assertion includes HP. The current independent run reports 8 passed with no expected failures. This acceptance is limited to the listed, fixture-backed CPU primitives on the tested environment. It is not acceptance of complete simulator parity, a full-race API, GPU correctness, or GPU speedup; other backends still need independent validation.
