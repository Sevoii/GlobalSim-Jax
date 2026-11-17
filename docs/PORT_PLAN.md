# GlobalSim JAX port plan

Owner: coordinating Codex session. Implementer: gpt-5.6-terra via Herdr. Reviewer: gpt-6-luna via Herdr.
Reference: C:/Users/a/RustroverProjects/GlobalSim at 58da7bc4693d9aa8f6340e348cba1c5b5fb5a856.
Destination: C:/Users/a/PycharmProjects/GlobalSim-Jax.

## Objective and boundaries
Build a batched JAX backend for the validated retail race runtime. Preserve the Rust reference unchanged. GPU parallelism is primarily independent races/seeds, not simultaneous runner updates. This is a staged port, not an assertion of complete simulator parity. Keep AGPL-3.0-only attribution and copy LICENSE/NOTICE with derived code/data. Preserve existing .idea, .venv and starter main.py.

## Milestone 1: executable numerical foundation (current assignment)
- Installable Python package globalsim_jax, pyproject.toml, README and CLI/demo/benchmark.
- Explicit precision policy: require JAX x64 for mixed-precision reference calculations; do not silently change global JAX settings on package import. Explain setup and precision limitations.
- Pure JAX retail_random.rs initialization, next_int, next_double, integer (including wide ranges), real and real_distance. Explicit pytree RNG state; exact conditional draw consumption; jit and batched execution.
- Port the compact stateless retail physics primitives: phase_boundaries, target_speed_indices, advance_start_delay, finish_time, bounded_stat/root and feasible core parameter-driven speed/HP/acceleration helpers. Do not substitute invented equations. Record an exact supported-function inventory; narrow scope explicitly if necessary.
- Reuse Rust RNG/physics/finish fixtures as independent expected values, retaining provenance. Add meaningful tests for signed seed boundaries, zero-width draws, batch-vs-scalar independence, precision boundaries, and jit execution. Record exact versus tolerance-based assertions.
- A results-only numerical throughput benchmark with warmup and block_until_ready synchronization. Clearly label primitive benchmarks, not full-race speedups. Report backend/device and timings. Run CPU checks; attempt GPU discovery without claiming CPU evidence proves GPU behavior.
- No full-race public API until ordered field, skills and initialization are ported. Unsupported features must error rather than approximate silently.

## Milestone 2: restricted race loop
Port initialization/stat conversion and geometry tables, then a no-skills ordered movement/HP loop with bounded arrays. Finish interpolation, stopping masks and timeout flags. Compare tick traces against Rust for the same restricted inputs. Fixed-shape [batch,runner] state and explicit capacity limits.

## Milestone 3: faithful field behavior
Port ranks/ties, blocking, pacing, lane selection/overlaps, overtaking, rushing, competition and spurt. Preserve per-pass snapshots and live roster observation points from race_field_retail.rs, including later runners seeing earlier movement updates. Preserve shared RNG ordering.

## Milestone 4: skills and production validation
Compile conditions into numeric instructions or generated JAX code; bounded active/pending skill storage, timers/observers, priority passes and peer effects. Explicit overflow rejection. Differential tests against official and Rust fixtures. Enable only demonstrated configurations. Separate approximate support if ever requested.

## Milestone 5: GPU evaluation
WSL2/Linux CUDA JAX environment on RTX 3060 if available. Batch sweep 1,32,256,1024,8192, same course/roster. Compare complete results-only throughput with current Rust prepared multithreaded batches. Account for compilation, preparation/transfers and steady state separately. Full replay support is deferred. Do not promise a speedup before measurement.

## Review and handoff gates
Terra implements M1, runs tests and updates STATUS.md/HANDOFF.md. Luna independently reviews correctness, fixture independence, numerical rounding, batching, API claims and benchmark fairness, writing docs/REVIEW.md. Terra fixes findings; coordinator reruns validation and records residual issues. Keep durable progress on every material change, commands/results, environment blockers and next tasks. No deployment/publishing. No unrelated source changes.
