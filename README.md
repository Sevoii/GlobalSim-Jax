# GlobalSim JAX

AGPL-3.0-only, fixture-backed JAX port derived from the pinned GlobalSim Rust reference. Milestone 2 adds a deliberately restricted flat/no-field runner loop; it is **not** a complete race simulator and has no field behavior, skills, geometry-loop integration, or GPU-parity claim.

Tested on Python 3.13.7 with JAX 0.12.0.1 CPU on Windows. The package requires JAX 0.4.36 or newer because its pytree dataclass registration relies on inferred fields (the tested version is recommended).

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m globalsim_jax.cli restricted-demo --ticks 8
.\.venv\Scripts\python.exe -m globalsim_jax.cli field-demo --ticks 8
.\.venv\Scripts\python.exe -m globalsim_jax.cli benchmark --batch 8192 --repeats 100
```

Before importing the package, enable x64 explicitly (JAX global configuration is never changed on import):

```python
import jax
jax.config.update("jax_enable_x64", True)
from globalsim_jax import seed
```

The package uses float32 stores where the reference does, and float64 intermediates where its Rust port documents wider calculations. Fixture tests use exact bit assertions where applicable and documented tolerances only for exported finish crossings.

See [the exact supported-function inventory](docs/SUPPORTED_FUNCTIONS.md), including the experimental status of `real_distance` and explicit unsupported scope.

For the M2 loop, create input state with `restricted.initialize`, then call `restricted.run_checked` with a frozen `RestrictedConfig`. The checked entry requires nonempty equal `[batch, runner]` arrays, runner capacity at most 32, finite values, positive ground/distance, nonnegative finite `dt`, integer strategies 1â€“5, and the currently evidenced unity speed and acceleration-distance aptitude factors. Its lower-level `step` and `run` functions are prevalidated JIT kernels, not host validation APIs. The bundled `CourseGeometry` API is a separately tested course10101/1200m geometry component and is not consumed by this flat runner loop.

Frozen Rust-oracle and captured-asset hashes are recorded in [docs/ORACLE_PROVENANCE.json](docs/ORACLE_PROVENANCE.json).

M3a adds a checked fixed-lane field profile through `globalsim_jax.field.run_checked`. It supports ranks/ties, front/side blocking snapshots, ordered blocker minimum/cap observation, and bounded first-finish truncation. `field-demo` reports positions and truncation for that profile. This is not complete M3: pacing, lane selection/overlaps, overtaking, rushing, competition, spurt, shared RNG, skills, and retail post-goal updates remain unsupported.

`globalsim-jax benchmark` is a synchronized primitive-only benchmark. It is not a full-race or GPU speedup result.

Windows CPU validation is the current supported environment. An RTX 3060 is known to be present, but CUDA JAX is not installed or validated here. GPU work is deferred to a separate WSL2/Linux CUDA JAX setup and must independently establish correctness and performance.


Run the restricted supplied-target moving-field demo with:

```powershell
python -m globalsim_jax.cli moving-field-demo --ticks 8
```

This integrates deterministic lane movement/overlap with the flat no-skills field and retains source f32/f64 calculation order. Targets are supplied, lane width is constant, and the wrapper freezes after the first finish or all-runner timeout. Automatic lane targets/pacing and GPU validation remain future work. See docs/HANDOFF.md for current validated scope and restart instructions.
