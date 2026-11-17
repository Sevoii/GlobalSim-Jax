import jax
import jax.numpy as jnp
import numpy as np
import pytest
import json
from pathlib import Path
from typing import NamedTuple

from globalsim_jax.field import BlockParameters
from globalsim_jax.lane_queries import SideRoom
from globalsim_jax.lane_target import TargetCache, TargetState, collect, initialize_target_state, update_targets, update_targets_checked


FIXTURE=json.loads((Path(__file__).parent / "fixtures" / "target_update_oracle.json").read_text())


class _KernelParameters(NamedTuple):
    front_distance: jax.Array
    front_lane: jax.Array


def _f32(bits):
    return np.asarray(bits, np.uint32).view(np.float32)


def _horse(bits):
    return np.asarray(bits, np.uint32).view(np.float32)


def _cache(node):
    count=node["count"]
    peers=np.zeros((32,4),np.float32)
    if count:
        peers[:count]=np.asarray(node["peers"],np.uint32).view(np.float32)
    visible=np.zeros(32,bool); visible[np.asarray(node["visible"],np.int32)]=True
    side=node["side"]
    return TargetCache(
        jnp.asarray(_horse(node["owner"])),jnp.asarray(peers),jnp.int32(count),jnp.asarray(visible),
        SideRoom(jnp.asarray(side[0]),jnp.asarray(side[1]),jnp.asarray(_f32(side[2])),jnp.asarray(_f32(side[3])),jnp.asarray(_f32(side[4])),jnp.int32(side[5])),
        jnp.int32(node["front"]),jnp.asarray(_f32(node["front_distance"])),jnp.asarray(_f32(node["front_lane"])),jnp.asarray(node["force_in"]),
    )


def _state(node):
    targets=np.full(32,-1,np.int32); targets[:len(node["targets"])]=node["targets"]
    present,bits=node["extra_outward"]
    extra=_f32(bits) if present else _f32(bits)
    cache=_cache(node["cache"])
    return TargetState(cache,jnp.asarray(targets),jnp.int32(len(node["targets"])),jnp.asarray(_f32(node["cooldown"])),jnp.asarray(node["force_check"]),jnp.asarray(node["force_in"]),jnp.asarray(extra),jnp.int32(node["previous_can_in"]),jnp.asarray(node["skip_passing"]))


def _live(rows):
    result=np.zeros((32,4),np.float32)
    if rows:
        result[:len(rows)]=np.asarray(rows,np.uint32).view(np.float32)
    return result


def _parameters(update):
    return BlockParameters(front_distance=float(_f32(update["front_distance"])),front_lane=float(_f32(update["front_lane"])))


def _assert_state_matches(state, node):
    expected=_state(node)
    _assert_tree_bits(state,expected)


def _step(state, update):
    return update_targets(state,_horse(update["owner"]),_live(update["live"]),state.cache.count,_f32(update["dt"]),_parameters(update),cooldown=_f32(update["cooldown_param"]))


def cache(owner, peers, visible=True, front=-1):
    peers=np.asarray(peers,np.float32).reshape((-1,4)); c = collect(np.asarray(owner, np.float32), np.pad(peers, ((0, 32-len(peers)), (0, 0))), len(peers), .8, 20., 0, True, .2)
    return c if visible else c.__class__(c.owner,c.peers,c.count,jnp.zeros(32,bool),c.side,jnp.int32(front),c.front_distance,c.front_lane,c.force_in)


def test_retention_visible_front_and_cooldown_eager_jit():
    owner=np.array([0.,.2,24.,25.],np.float32)
    peers=np.array([[10.,.4,20.,26.],[1.,.2,30.,30.]],np.float32)
    # Cached peer 0 is eligible from cached gap/speed but its live position is irrelevant.
    c=cache(owner, peers)
    s=initialize_target_state(c, targets=np.array([31,0],np.int32), cooldown=np.float32(.05), force_check=False, extra_outward=np.float32(.7), previous_can_in=0, skip_passing=True)
    live=np.array([[100.,.4,26.,24.],[1.,.2,30.,30.]],np.float32)
    for fn in (lambda x:update_targets(x,owner,np.pad(live,((0,30),(0,0))),2,.1,cooldown=.5), jax.jit(lambda x:update_targets(x,owner,np.pad(live,((0,30),(0,0))),2,.1,cooldown=.5))):
        out=fn(s); assert int(out.target_count)==2; assert int(out.targets[0])==0 and int(out.targets[1])==1
        assert float(out.cooldown)==pytest.approx(.5); assert bool(out.skip_passing) is False
        assert float(out.extra_outward)==pytest.approx(.7); assert out.cache is not None


def test_checked_stale_and_bad_synthetic_state():
    owner=np.array([0.,.2,24.,25.],np.float32); c=cache(owner,[])
    # Source retains this usize until its first drop pass.  It must not be
    # narrowed/rejected merely because it exceeds the 32-slot peer capacity.
    s=initialize_target_state(c,targets=np.array([99],np.int32),cooldown=np.float32(.01))
    out=update_targets_checked(s,owner,np.zeros((32,4),np.float32),np.float32(.1),cooldown=np.float32(.5))
    assert int(out.target_count)==0 and bool(out.force_check)
    bad=s.__class__(s.cache,s.targets.at[0].set(0),jnp.int32(2),s.cooldown,s.force_check,s.force_in,s.extra_outward,s.previous_can_in,s.skip_passing)
    with pytest.raises(ValueError): update_targets_checked(bad,owner,np.zeros((32,4),np.float32),np.float32(.1),cooldown=np.float32(.5))


@pytest.mark.parametrize("previous", [True, 0.0, np.array([0], np.int32), 2])
def test_initializer_rejects_noncanonical_lifecycle_sentinels(previous):
    c=cache(np.array([0.,.2,24.,25.],np.float32),[])
    with pytest.raises(ValueError):
        initialize_target_state(c, previous_can_in=previous)
    with pytest.raises(ValueError):
        initialize_target_state(c, force_check=np.array([True]))
    with pytest.raises(ValueError):
        initialize_target_state(c, skip_passing=1)
    with pytest.raises(ValueError):
        initialize_target_state(c, extra_outward=np.inf)


def test_checked_rejects_scalar_subnormal_ranges_and_poisoned_padding():
    owner=np.array([0.,.2,24.,25.],np.float32); c=cache(owner,[])
    s=initialize_target_state(c,cooldown=np.float32(np.finfo(np.float32).tiny / 2))
    live=np.zeros((32,4),np.float32)
    with pytest.raises(ValueError, match="subnormal"):
        update_targets_checked(s,owner,live,np.float32(.1),cooldown=np.float32(.5))
    s=initialize_target_state(c)
    with pytest.raises(ValueError, match="subnormal"):
        update_targets_checked(s,owner,live,np.float32(np.finfo(np.float32).tiny / 2),cooldown=np.float32(.5))
    with pytest.raises(ValueError):
        update_targets_checked(s,owner,live,np.float32(-.1),cooldown=np.float32(.5))
    with pytest.raises(ValueError):
        update_targets_checked(s,owner,live,np.float32(.1),cooldown=np.float32(-.5))
    poisoned=live.copy(); poisoned[0,0]=1.
    with pytest.raises(ValueError, match="padding"):
        update_targets_checked(s,owner,poisoned,np.float32(.1),cooldown=np.float32(.5))


def test_checked_rejects_pairwise_minimum_subnormal_blocker_gap():
    owner=np.array([-10.,.2,24.,25.],np.float32)
    low=np.finfo(np.float32).tiny
    peers=np.array([[low,.4,20.,26.],[np.nextafter(low,np.float32(np.inf)),.4,20.,26.]],np.float32)
    c=cache(owner,peers)
    s=initialize_target_state(c,cooldown=np.float32(.5))
    live=np.zeros((32,4),np.float32)
    live[:2]=np.array([[100.,.4,26.,24.],[100.,.4,26.,26.]],np.float32)
    with pytest.raises(ValueError, match="subnormal"):
        update_targets_checked(s,owner,live,np.float32(.1),cooldown=np.float32(.5))


def test_checked_state_leaf_dtypes_sentinels_and_cache_padding():
    owner=np.array([0.,.2,24.,25.],np.float32); c=cache(owner,[])
    s=initialize_target_state(c)
    live=np.zeros((32,4),np.float32)
    bad_dtype=TargetState(s.cache,s.targets.astype(jnp.int64),s.target_count,s.cooldown,s.force_check,s.force_in,s.extra_outward,s.previous_can_in,s.skip_passing)
    with pytest.raises(ValueError, match="state.targets"):
        update_targets_checked(bad_dtype,owner,live,np.float32(.1),cooldown=np.float32(.5))
    bad_previous=TargetState(s.cache,s.targets,s.target_count,s.cooldown,s.force_check,s.force_in,s.extra_outward,jnp.int32(2),s.skip_passing)
    with pytest.raises(ValueError, match="previous"):
        update_targets_checked(bad_previous,owner,live,np.float32(.1),cooldown=np.float32(.5))
    bad_extra=TargetState(s.cache,s.targets,s.target_count,s.cooldown,s.force_check,s.force_in,jnp.float32(np.inf),s.previous_can_in,s.skip_passing)
    with pytest.raises(ValueError, match="extra"):
        update_targets_checked(bad_extra,owner,live,np.float32(.1),cooldown=np.float32(.5))
    bad_cache=c.__class__(c.owner,c.peers.at[0,0].set(1.),c.count,c.visible,c.side,c.front,c.front_distance,c.front_lane,c.force_in)
    bad=TargetState(bad_cache,s.targets,s.target_count,s.cooldown,s.force_check,s.force_in,s.extra_outward,s.previous_can_in,s.skip_passing)
    with pytest.raises(ValueError, match="padding"):
        update_targets_checked(bad,owner,live,np.float32(.1),cooldown=np.float32(.5))


def _assert_tree_bits(actual, expected):
    for got, want in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        got, want=np.asarray(got),np.asarray(want)
        assert got.shape == want.shape and got.dtype == want.dtype
        if got.dtype == np.float32:
            np.testing.assert_array_equal(got.view(np.uint32),want.view(np.uint32))
        else:
            np.testing.assert_array_equal(got,want)


def _scan_kernel(state, owners, lives, dts, cooldowns, distances, lanes):
    def body(carry, values):
        owner, live, dt, cooldown, distance, lane=values
        next_state=update_targets(carry,owner,live,carry.cache.count,dt,
                                  _KernelParameters(distance,lane),cooldown=cooldown)
        return next_state,next_state
    return jax.lax.scan(body,state,(owners,lives,dts,cooldowns,distances,lanes))


def test_frozen_rust_traces_every_leaf_eager_jit_scan_and_checked():
    """Every supported update uses only independently frozen Rust bits."""
    compiled=jax.jit(_scan_kernel)
    for case in FIXTURE["cases"]:
        assert case["supported"] is True
        initial=_state(case["initial"])
        before=jax.tree_util.tree_map(lambda x:np.array(x,copy=True),initial)
        eager=initial
        expected=[]
        for update in case["updates"]:
            eager=_step(eager,update)
            expected.append(_state(update["out"]))
            _assert_state_matches(eager,update["out"])
        # Functional update must never alter the supplied lifecycle/cache tree.
        _assert_tree_bits(initial,before)
        checked=initial
        for update in case["updates"]:
            checked=update_targets_checked(checked,_horse(update["owner"]),_live(update["live"]),_f32(update["dt"]),cooldown=_f32(update["cooldown_param"]),parameters=_parameters(update))
            _assert_state_matches(checked,update["out"])
        owners=jnp.asarray(np.asarray([row["owner"] for row in case["updates"]],np.uint32).view(np.float32))
        lives=jnp.asarray(np.asarray([_live(row["live"]) for row in case["updates"]],np.float32))
        dts=jnp.asarray(np.asarray([_f32(row["dt"]) for row in case["updates"]],np.float32))
        cooldowns=jnp.asarray(np.asarray([_f32(row["cooldown_param"]) for row in case["updates"]],np.float32))
        distances=jnp.asarray(np.asarray([_f32(row["front_distance"]) for row in case["updates"]],np.float32))
        lanes=jnp.asarray(np.asarray([_f32(row["front_lane"]) for row in case["updates"]],np.float32))
        final,trace=compiled(initial,owners,lives,dts,cooldowns,distances,lanes)
        _assert_tree_bits(final,expected[-1])
        expected_trace=jax.tree_util.tree_map(lambda *rows:jnp.stack(rows),*expected)
        _assert_tree_bits(trace,expected_trace)


def test_frozen_rust_vmap_matches_scalar_first_updates():
    cases=FIXTURE["cases"]
    states=tuple(_state(case["initial"]) for case in cases)
    rows=[case["updates"][0] for case in cases]
    stacked=jax.tree_util.tree_map(lambda *values:jnp.stack(values),*states)
    owners=jnp.asarray(np.asarray([row["owner"] for row in rows],np.uint32).view(np.float32))
    lives=jnp.asarray(np.asarray([_live(row["live"]) for row in rows],np.float32))
    dts=jnp.asarray(np.asarray([_f32(row["dt"]) for row in rows],np.float32))
    cooldowns=jnp.asarray(np.asarray([_f32(row["cooldown_param"]) for row in rows],np.float32))
    distances=jnp.asarray(np.asarray([_f32(row["front_distance"]) for row in rows],np.float32))
    lanes=jnp.asarray(np.asarray([_f32(row["front_lane"]) for row in rows],np.float32))
    def one(state,owner,live,dt,cooldown,distance,lane):
        return update_targets(state,owner,live,state.cache.count,dt,_KernelParameters(distance,lane),cooldown=cooldown)
    batched=jax.vmap(one)(stacked,owners,lives,dts,cooldowns,distances,lanes)
    for index,(state,row) in enumerate(zip(states,rows,strict=True)):
        _assert_tree_bits(jax.tree_util.tree_map(lambda value:value[index],batched),_step(state,row))


def test_frozen_subnormal_diagnostic_is_rejected_before_dispatch():
    row=FIXTURE["diagnostics"][0]
    update=row["updates"][0]
    with pytest.raises(ValueError,match="subnormal"):
        update_targets_checked(_state(row["initial"]),_horse(update["owner"]),_live(update["live"]),_f32(update["dt"]),cooldown=_f32(update["cooldown_param"]),parameters=_parameters(update))


def test_prevalidated_scan_jit_and_vmap_match_independent_scalars():
    owners=np.array([[0.,.2,24.,25.],[0.,.4,24.,25.]],np.float32)
    peers=np.array([[[10.,.4,20.,26.]],[[1.,.4,20.,26.]]],np.float32)
    states=tuple(initialize_target_state(cache(owner,peer),cooldown=np.float32(.15)) for owner,peer in zip(owners,peers,strict=True))
    lives=np.zeros((2,32,4),np.float32)
    lives[:,0]=np.array([[100.,.4,26.,24.],[100.,.4,26.,24.]],np.float32)
    def one(state, owner, live):
        return update_targets(state,owner,live,1,np.float32(.1),cooldown=np.float32(.5))
    # The two cases use separate cached snapshots; batching must not merge a
    # target or blocker observation from the neighboring element.
    batched=jax.vmap(one)(jax.tree_util.tree_map(lambda *x:jnp.stack(x),*states),owners,lives)
    scalar=tuple(one(state,owner,live) for state,owner,live in zip(states,owners,lives,strict=True))
    for index in range(2):
        _assert_tree_bits(jax.tree_util.tree_map(lambda x:x[index],batched),scalar[index])
    def scan(state):
        def body(carry, _):
            next_state=one(carry,owners[0],lives[0])
            return next_state,next_state
        return jax.lax.scan(body,state,None,length=4)
    eager=scan(states[0]); compiled=jax.jit(scan)(states[0])
    _assert_tree_bits(eager,compiled)
