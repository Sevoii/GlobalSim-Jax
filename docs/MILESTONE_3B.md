# Milestone 3b: deterministic lateral mechanics

Reference: pinned Rust 58da7bc4693d9aa8f6340e348cba1c5b5fb5a856.

Accepted bounded primitives (2026-11-16):

- globalsim_jax.lane: exact comparison-sort permutations and sequential overlap correction for up to 32 active runners, including source size-2/3 networks and larger equal-key partitions. Prevalidated padded kernels preserve inactive tails. Checked wrappers reject unsupported capacity, keys, flags, nonfinite storage and lanes outside the supported int32 cast domain.
- globalsim_jax.lane_move: standalone supplied-target MoveState.step and checked scalar boundary. State includes lane, clamped target, stored speed, continuation time and direction. Preserve blocked momentum, pre-inward modifier-zero return, mixed precision, after-goal dt-squared displacement and source timer ordering. The approximately helper reconstructs subnormal values to preserve the source comparison under CPU XLA flush-to-zero behavior. Modifier capacity is 16; checked phase/rank use bounded int32 and direction is 0/1. JAX x64 must be explicitly enabled by the caller.

Independent evidence: tools/lane_oracle.rs includes pinned Rust sort and extracted overlap behavior; tools/lane_move_oracle.rs mechanically copies SideRoom, MoveParameters (loader omitted), MoveState, MoveContext, approximately and the full step implementation, while including pinned physics. Frozen fixtures compare exact sort IDs, overlap state and all movement state fields in eager/JIT paths. The copied 700-vector legacy fixture provides additional lane/target/speed/moved evidence; 13 new oracle cases cover full continuation and early/blocked paths. All source/oracle/fixture hashes are recorded in ORACLE_PROVENANCE.json.

Validation: coordinator full suite 43 passed in 83.75s; Luna focused lane suite 9 passed in 29.37s. Coordinator rebuilt the movement oracle, reproduced its fixture byte-for-byte and verified all 30 manifest hashes. See final bounded acceptance in REVIEW_M3B.md.

Next: side_room/is_visible and overtaking predicates, then supplied-target moving-lane field integration. Follow the source observation/gate/overlap/terminal criteria in REVIEW_M3B.md. M3a field itself remains fixed-lane; these standalone primitives are not yet connected to its scan. Automatic TargetState, target selection, pacing and M3c are separate unimplemented work. This acceptance establishes neither full M3b/M3 nor full-race or GPU parity.
