# Target-cache subset oracle

AGPL-3.0-only; source reference is GlobalSim revision
`58da7bc4693d9aa8f6340e348cba1c5b5fb5a856`. The Rust checkout is read-only.

`build_target_collect_oracle.py` mechanically extracts the original `Horse`,
`TargetParameters` type (without its JSON loader), `TargetContext`, `TargetState`,
and the complete `collect`, `collect_roster`, `collect_nearby`, `collect_cached`
methods into `target_collect_oracle.rs`. Root verified these extracted spans
against the original source. No collection equations or method bodies change.
The generator checks the source SHA-256 against `ORACLE_PROVENANCE.json`;
check the reference git revision and all dependency hashes before regeneration.

The generated module imports the previously accepted `lane_query_oracle.rs`
query extraction for original row/block/side types and helper functions. Its
dependency hash and original `retail_lane.rs` hash are in the same manifest.
The optional sorted-nearby method compiles but is not invoked: roster tests
explicitly pass `None` for the sorted argument. No random draws are involved.

Destination-only harness code constructs horses, contexts, parameters and fresh
source `TargetState` values, calls collection, and serializes results. It does
not implement target selection or modify the extracted collection algorithm.
Unused context/parameter/lifecycle fields receive inert values. The fixture
therefore proves only a collection snapshot, not lifecycle preservation or an
engine replay. JAX stores a visible mask instead of a Vec; ascending true slots
must equal the serialized source Vec. JAX zero padding is wrapper storage and
is separately tested with poisoned inactive rows.

The fixture contains 74 standalone peer cases and 4 actual unsorted roster
cases, including first/middle/last owner removal and a 32-runner roster.
Standalone counts range from 0 through 32 peers; a complete roster supports
32 total runners, hence at most 31 peers after removing its owner. Every
horse scalar and side-space/blocker float is serialized as f32 bits. Absent
blocker lanes use Rust f32 NaN bits; absent front/inner indices use -1. Source
visible peer indices, flags and configuration accompany each case. Boundary
cases include adjacent f32 values around front-lane, side-lane and visibility
thresholds. These are source-generated expected values, never JAX output.

Compile from the destination root:

```
rustc --edition=2021 -Awarnings -O tools/target_collect_oracle.rs -o tools/target_collect_oracle.exe
```

Capture stdout using Python `subprocess.check_output` and `Path.write_bytes`;
PowerShell text redirection can append line endings and break byte identity.
Root independently rebuilt this oracle and reproduced the fixture byte for
byte, verified the exact extraction, all 40 manifest hashes, and a clean pinned
Rust checkout. Final coordinator suite results are recorded in STATUS/HANDOFF.
