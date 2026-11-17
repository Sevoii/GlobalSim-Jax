# Milestone 3: field behavior

Reference: Rust commit `58da7bc4693d9aa8f6340e348cba1c5b5fb5a856`. M3a is the first bounded portion; it does not complete milestone 3.

## M3a accepted contract
One-based ranks with descending-position/lower-index ties; fixed-lane front/side blocking queries; ordered stored-minimum inheritance; float64 front/side/all blocked timers; and a runner-index-ordered movement pass with live blocker-speed reads. Lanes and M2 numeric inputs are supplied. The blocker index/gap and side result come from the pre-movement snapshot. The query no-blocker sentinel is index -1 paired with NaN gap. Tick metadata labels ranks `pre_movement`; call ranks on the resulting positions for final ranks.

The minimum pass is precomputed sequentially before movement. This is equivalent only in this restricted callback-free profile: no skill/stat/receiver changes can intervene or modify minimum after refresh. Current speed equals observed speed only under this receiver-free profile and its cap-at-least-minimum projection. These optimizations must be reconsidered before broader M3/M4 integration.

Public query and run checked boundaries validate the supported shapes, finite values, types, precision and M2 profile. Compiled kernels are prevalidated. Fresh attachment can initialize minima to3; nonfresh attachment requires explicit prior minima. An explicit synthetic history used by a frozen extraction is not evidence of fresh-game reachability.

A crossing tick completes its ordered pass, then the whole race batch freezes before any post-goal tick. The bounded execution cap also freezes terminal field state. This is a wrapper stopping policy, not a port of retail post-goal behavior. Full post-goal minimum5/deceleration behavior remains unsupported. Front queries independently include finishers, while side queries exclude finished peers.

## Evidence
Frozen Rust query cases cover ties, front/side bounds and finish flags; an independent custom front-lane coefficient vector exposes f32 parameter storage before widening. Three eight-tick, three-runner Rust field cases cover a higher-index blocker, lower-index live-speed cap with partial gate, and fully closed gate. All19 channels per runner are compared eagerly and under JIT scan, and race batches are compared with scalar execution. The order counterexample checks actual follower speed against updated-blocker and frozen-initial-blocker caps, which must differ. Source, adapter and fixture hashes are in ORACLE_PROVENANCE.json. Regeneration must first verify those hashes and pinned commit; generators use absolute source paths and do not enforce revision themselves.

## Remaining milestone 3
M3b: pacing, lane selection/overlaps, overtaking and lane targets.
M3c: rushing, competition, spurt and shared RNG scheduling.
M4: skills/conditions and peer effects. GPU correctness and performance evaluation remain separate work. Geometry is still standalone, not integrated into these flat-loop profiles.

