"""Frozen source evidence for the bounded TargetState collection snapshot."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from globalsim_jax.field import BlockParameters
from globalsim_jax.lane_target import (
    MAX_PEERS, collect, collect_checked, collect_roster,
    collect_roster_checked,
)


DATA = json.loads((Path(__file__).parent / "fixtures" / "target_collect_oracle.json").read_text())
P = BlockParameters()


def f32(bits):
    return np.asarray(bits, np.uint32).view(np.float32)


def case_args(row):
    owner = f32(row["owner"])
    peers = np.zeros((MAX_PEERS, 4), np.float32)
    raw = np.asarray(row["peers"], np.uint32).reshape((-1, 4)).view(np.float32)
    peers[:len(raw)] = raw
    max_lane, vision, phase, empty, threshold = row["config"]
    return owner, peers, len(raw), f32(max_lane), f32(vision), np.int32(phase), np.bool_(empty), f32(threshold)


def cache_bits(cache):
    return np.asarray([
        cache.side.can_in, cache.side.can_out,
        np.asarray(cache.side.in_space).view(np.uint32),
        np.asarray(cache.side.out_space).view(np.uint32),
        np.asarray(cache.side.inner_blocker_lane).view(np.uint32),
        cache.side.inner_blocker_index,
        cache.front, np.asarray(cache.front_distance).view(np.uint32),
        np.asarray(cache.front_lane).view(np.uint32), cache.force_in,
    ])


def expected_bits(row):
    return np.asarray([*row["side"], row["front"],
                       np.asarray(P.front_distance, np.float32).view(np.uint32),
                       np.asarray(P.front_lane, np.float32).view(np.uint32), row["force_in"]])


def assert_cache(cache, row, peers, count):
    np.testing.assert_array_equal(cache_bits(cache), expected_bits(row))
    np.testing.assert_array_equal(np.flatnonzero(np.asarray(cache.visible)), row["visible"])
    np.testing.assert_array_equal(np.asarray(cache.owner).view(np.uint32), row["owner"])
    np.testing.assert_array_equal(np.asarray(cache.peers[:count]).view(np.uint32), np.asarray(peers[:count]).view(np.uint32))
    np.testing.assert_array_equal(np.asarray(cache.peers[count:]), 0.0)
    assert int(np.asarray(cache.count)) == count


def test_frozen_collect_cache_every_field_eager_and_jit():
    compiled = jax.jit(lambda owner, peers, count, max_lane, vision, phase, empty, threshold:
                       collect(owner, peers, count, max_lane, vision, phase, empty, threshold, P))
    for row in DATA["cases"]:
        args = case_args(row)
        for actual in (collect(*args, P), compiled(*args)):
            assert_cache(actual, row, args[1], args[2])


def test_collect_vmap_isolation_and_actual_owner_removal():
    rows = [DATA["cases"][0], DATA["cases"][10], DATA["cases"][-1]]
    args = [case_args(row) for row in rows]
    batched = jax.jit(jax.vmap(lambda owner, peers, count, max_lane, vision, phase, empty, threshold:
        collect(owner, peers, count, max_lane, vision, phase, empty, threshold, P)))(
        *[np.asarray([a[i] for a in args]) for i in range(8)])
    for i, row in enumerate(rows):
        assert_cache(jax.tree_util.tree_map(lambda x: x[i], batched), row, args[i][1], args[i][2])

    compiled_roster = jax.jit(lambda roster, count, owner, max_lane, vision, phase, empty, threshold:
        collect_roster(roster, count, owner, max_lane, vision, phase, empty, threshold, P))
    for row in DATA["roster_cases"]:
        roster = f32(row["roster"])
        owner = np.int32(row["owner_index"])
        padded = np.zeros((MAX_PEERS, 4), np.float32); padded[:len(roster)] = roster
        config = row["config"]
        roster_args = (padded, len(roster), owner, f32(config[0]), f32(config[1]),
                       np.int32(config[2]), np.bool_(config[3]), f32(config[4]))
        peers = f32(row["peers"])
        expected = {"owner": row["roster"][row["owner_index"]], "peers": row["peers"],
                    "visible": row["visible"], "side": row["side"], "front": row["front"],
                    "force_in": row["force_in"]}
        for actual in (collect_roster(*roster_args, P), compiled_roster(*roster_args)):
            assert_cache(actual, expected, np.pad(peers, ((0, MAX_PEERS-len(peers)), (0, 0))), len(peers))
        checked = collect_roster_checked(roster, owner, max_lane=f32(config[0]), vision=f32(config[1]),
                                         phase=np.int32(config[2]), empty_in_move=np.bool_(config[3]),
                                         force_in_threshold=f32(config[4]), parameters=P)
        assert_cache(checked, expected, np.pad(peers, ((0, MAX_PEERS-len(peers)), (0, 0))), len(peers))

    # Same-capacity first/middle/last removals exercise batched owner indices.
    rows = DATA["roster_cases"][:3]
    roster = np.stack([np.pad(f32(r["roster"]), ((0, MAX_PEERS-len(r["roster"])), (0, 0))) for r in rows])
    batch = jax.jit(jax.vmap(lambda r, owner: collect_roster(r, np.int32(5), owner,
        np.float32(.8), np.float32(20.), np.int32(0), np.bool_(True), np.float32(.2), P)))(
        roster, np.asarray([r["owner_index"] for r in rows], np.int32))
    for i, row in enumerate(rows):
        expected = {"owner": row["roster"][row["owner_index"]], "peers": row["peers"],
                    "visible": row["visible"], "side": row["side"], "front": row["front"], "force_in": row["force_in"]}
        peers = np.asarray(row["peers"], np.uint32).reshape((-1, 4)).view(np.float32)
        assert_cache(jax.tree_util.tree_map(lambda x: x[i], batch), expected,
                     np.pad(peers, ((0, MAX_PEERS-len(peers)), (0, 0))), len(peers))


def test_checked_boundaries_and_capacity_padding():
    owner = np.array([0., .2, 12., 13.], np.float32)
    peers = np.array([[1., .2, 11., 14.]], np.float32)
    # Checked valid calls must agree with every independent cached-source row.
    for row in DATA["cases"]:
        args = case_args(row)
        actual = collect_checked(args[0], args[1][:args[2]], max_lane=args[3], vision=args[4],
                                 phase=args[5], empty_in_move=args[6], force_in_threshold=args[7], parameters=P)
        assert_cache(actual, row, args[1], args[2])
    with pytest.raises(ValueError): collect_checked(np.zeros(3), peers, max_lane=.8, vision=1., phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=.2)
    with pytest.raises(ValueError): collect_checked(owner, np.zeros((33, 4)), max_lane=.8, vision=1., phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=.2)
    with pytest.raises(ValueError): collect_checked(np.array(["x", ".2", "12", "13"]), peers, max_lane=.8, vision=1., phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=.2)
    with pytest.raises(ValueError): collect_checked(np.array([0. + 1j, .2, 12., 13.]), peers, max_lane=.8, vision=1., phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=.2)
    with pytest.raises(ValueError, match="overflow"):
        collect_checked(np.array([1e100, .2, 12., 13.]), peers, max_lane=.8, vision=1., phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=.2)
    with pytest.raises(ValueError): collect_checked(owner, peers, max_lane=.8, vision=1., phase=True, empty_in_move=np.bool_(True), force_in_threshold=.2)
    with pytest.raises(ValueError): collect_checked(owner, peers, max_lane=np.inf, vision=1., phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=.2)
    with pytest.raises(ValueError): collect_checked(owner, peers, max_lane=.8, vision=1., phase=np.int32(0), empty_in_move=np.int32(1), force_in_threshold=.2)
    with pytest.raises(ValueError): collect_checked(owner, peers, max_lane=.8, vision=1., phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=np.finfo(np.float32).tiny / 2)
    # Both stored operands can be normal while a source f32 subtraction is subnormal.
    tiny = np.finfo(np.float32).tiny
    near = np.nextafter(np.float32(tiny), np.float32(np.inf))
    with pytest.raises(ValueError, match="subnormal"):
        collect_checked(np.array([tiny, .2, 12., 13.], np.float32), np.array([[near, .2, 11., 14.]], np.float32), max_lane=.8, vision=1., phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=.2)
    with pytest.raises(ValueError, match="intermediates"):
        collect_checked(np.array([-np.finfo(np.float32).max, .2, 12., 13.], np.float32), np.array([[np.finfo(np.float32).max, .2, 11., 14.]], np.float32), max_lane=.8, vision=1., phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=.2)
    # This row passes Rust's cone and widest-lane gates, so its ratio is
    # source-evaluated; 1 / f32::MAX narrows to a nonzero f32 subnormal.
    with pytest.raises(ValueError, match="subnormal"):
        collect_checked(owner, peers, max_lane=.8, vision=np.finfo(np.float32).max,
                        phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=.2)
    with pytest.raises(ValueError): collect_roster_checked(np.zeros((33, 4), np.float32), np.int32(0), max_lane=.8, vision=1., phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=.2)
    with pytest.raises(ValueError): collect_roster_checked(np.zeros((2, 4), np.float32), np.int32(2), max_lane=.8, vision=1., phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=.2)
    old = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(RuntimeError):
            collect_checked(owner, peers, max_lane=.8, vision=1., phase=np.int32(0), empty_in_move=np.bool_(True), force_in_threshold=.2)
    finally:
        jax.config.update("jax_enable_x64", old)


def test_inactive_padding_is_poisoned_but_never_observed():
    row = DATA["cases"][1]
    args = list(case_args(row))
    poisoned = args[1].copy()
    poisoned[args[2]:] = np.array([.5, .15, 99., 99.], np.float32)
    actual = collect(args[0], poisoned, *args[2:], P)
    assert_cache(actual, row, args[1], args[2])

