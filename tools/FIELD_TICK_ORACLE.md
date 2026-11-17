# M3a field-tick oracle

`field_tick_oracle.rs` is a destination-only, fixed-zero-lane extraction using pinned Rust `json.rs` and `retail_physics.rs` at `58da7bc4693d9aa8f6340e348cba1c5b5fb5a856`. It models only unit aptitude, flat/no-skills/no-pace fixed-lane movement, snapshot blocker IDs/gaps, ordered stored-minimum refresh, live blocker-speed caps, and first-finish truncation. It is not a complete field oracle.

Build only after checking the pinned revision and hashes in `docs/ORACLE_PROVENANCE.json`:

`rustc tools/field_tick_oracle.rs -O -o tools/field_tick_oracle.exe; if ($LASTEXITCODE -ne 0) { throw 'rustc failed' }; .\.venv\Scripts\python.exe -c "import pathlib, subprocess, json; data=subprocess.check_output([str(pathlib.Path('tools/field_tick_oracle.exe').resolve())]); json.loads(data); pathlib.Path('tests/fixtures/field_tick_trace.json').write_bytes(data)"`

The frozen JSON has complete bit-level initial metadata and eight rows of three runners. Each runner row stores time, position, speed, HP, delay, phase, dash, finish time, stored minimum, blocker index/gap, side, rank, front/side/all f64 timers, finished, truncated, and runner index.
