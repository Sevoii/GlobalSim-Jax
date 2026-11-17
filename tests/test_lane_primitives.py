"""Frozen pinned-Rust evidence for the first deterministic M3b primitives."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from globalsim_jax.lane import (
    MAX_LANES,
    comparison_sort_indices,
    comparison_sort_indices_checked,
    resolve_overlaps,
    resolve_overlaps_checked,
)


ROOT = Path(__file__).parent / "fixtures"
ORACLE = json.loads((ROOT / "lane_oracle.json").read_text())
PHYSICS = json.loads((ROOT / "retail_physics_port.json").read_text())


def _sort_inputs(row):
    pairs = np.asarray(row["input"], dtype=np.int32)
    keys = np.zeros(MAX_LANES, dtype=np.int32)
    keys[: len(pairs)] = pairs[:, 0] if len(pairs) else []
    return keys, pairs[:, 1] if len(pairs) else np.empty(0, dtype=np.int32)


def _assert_sort(row, actual):
    keys, ids = _sort_inputs(row)
    count = len(ids)
    # retail_physics_port expected_order holds row IDs, not raw permutation
    # indices.  The source's public sort permutes those row IDs.
    np.testing.assert_array_equal(ids[np.asarray(actual)[:count]], row["order"] if "order" in row else row["expected_order"])


def test_comparison_sort_matches_pinned_rust_sizes_eager_and_jit():
    rows = ORACLE["sorts"] + ORACLE["special_sorts"]
    compiled = jax.jit(comparison_sort_indices)
    for row in rows:
        keys, _ = _sort_inputs(row)
        count = len(row["input"])
        _assert_sort(row, comparison_sort_indices(keys, count))
        _assert_sort(row, compiled(keys, np.int32(count)))

    # Reuse all legacy source fixture rows that fit the declared 32 capacity.
    for row in PHYSICS["sort_vectors"]:
        if len(row["input"]) <= MAX_LANES:
            keys, _ = _sort_inputs(row)
            _assert_sort(row, compiled(keys, np.int32(len(row["input"]))))


def test_comparison_sort_batched_and_checked_capacity():
    rows = [row for row in ORACLE["sorts"] if row["size"] in (1, 2, 3, 16, 17, 18, 32)]
    keys = np.stack([_sort_inputs(row)[0] for row in rows])
    counts = np.asarray([row["size"] for row in rows], dtype=np.int32)
    compiled = jax.jit(jax.vmap(comparison_sort_indices))
    for row, actual in zip(rows, compiled(keys, counts), strict=True):
        _assert_sort(row, actual)
    with pytest.raises(ValueError, match="at most 32"):
        comparison_sort_indices_checked(np.zeros(33, dtype=np.int32), 33)
    with pytest.raises(ValueError, match="integer dtype"):
        comparison_sort_indices_checked(np.asarray([1.0, 2.0]), 2)
    with pytest.raises(ValueError, match="representable"):
        comparison_sort_indices_checked(np.asarray([np.iinfo(np.int32).max + 1], dtype=np.int64), 1)


def _overlap_inputs(row):
    count = row["size"]
    position = np.zeros(MAX_LANES, np.uint32)
    lane = np.zeros(MAX_LANES, np.uint32)
    force = np.zeros(MAX_LANES, bool)
    position[:count] = row["position_bits"]
    lane[:count] = row["lane_bits"]
    force[:count] = row["force"]
    return position.view(np.float32), lane.view(np.float32), force


def _assert_overlap(row, actual):
    (lane, force), order = actual
    count = row["size"]
    np.testing.assert_array_equal(np.asarray(order)[:count], row["order"])
    np.testing.assert_array_equal(np.asarray(lane).view(np.uint32)[:count], row["out_lane_bits"])
    np.testing.assert_array_equal(np.asarray(force)[:count], row["out_force"])


def test_overlap_resolution_matches_frozen_rust_eager_jit_and_batches():
    rows = ORACLE["overlaps"]
    compiled = jax.jit(resolve_overlaps)
    for row in rows:
        position, lane, force = _overlap_inputs(row)
        _assert_overlap(row, resolve_overlaps(position, lane, force, row["size"]))
        _assert_overlap(row, compiled(position, lane, force, np.int32(row["size"])))

    selected = [row for row in rows if row["size"] in (0, 3, 17, 32)]
    positions, lanes, forces = zip(*[_overlap_inputs(row) for row in selected], strict=True)
    batched = jax.jit(jax.vmap(resolve_overlaps))(
        np.stack(positions), np.stack(lanes), np.stack(forces), np.asarray([r["size"] for r in selected], np.int32)
    )
    for index, row in enumerate(selected):
        _assert_overlap(row, ((batched[0][0][index], batched[0][1][index]), batched[1][index]))


def test_overlap_checked_padding_flags_and_strict_thresholds():
    # Empty source rows do not clear padded flag zero in the kernel.
    (_, padded_force), _ = resolve_overlaps(
        np.zeros(MAX_LANES, np.float32),
        np.zeros(MAX_LANES, np.float32),
        np.r_[True, np.zeros(MAX_LANES - 1, bool)],
        0,
    )
    assert bool(np.asarray(padded_force)[0])
    lane, force, order = resolve_overlaps_checked([], [], [])
    assert lane.size == force.size == order.size == 0
    # Exactly .4 is excluded, while the next lower float would be included.
    lane, force, _ = resolve_overlaps_checked([0.0, 0.4], [0.0, 0.0], [True, True])
    np.testing.assert_array_equal(force, [False, True])
    np.testing.assert_array_equal(lane.view(np.uint32), np.asarray([0, 0], np.uint32))
    below = np.nextafter(np.float32(0.4), np.float32(0.0))
    lane, force, _ = resolve_overlaps_checked([0.0, below], [0.0, 0.0], [False, False])
    assert bool(force[1])
    assert lane[1] > 0.0
    with pytest.raises(ValueError, match="at most 32"):
        resolve_overlaps_checked(np.zeros(33), np.zeros(33), np.zeros(33, bool))
    with pytest.raises(ValueError, match="boolean dtype"):
        resolve_overlaps_checked([0.0], [0.0], [1])
    with pytest.raises(ValueError, match="remain finite"):
        resolve_overlaps_checked([float(np.finfo(np.float32).max) * 2.0], [0.0], [False])
    with pytest.raises(ValueError, match="representable"):
        resolve_overlaps_checked([0.0], [float(np.iinfo(np.int32).max) * 2.0], [False])
    with pytest.raises(ValueError, match="representable"):
        resolve_overlaps_checked([0.0, 2.0], [float(2**31), 0.0], [False, False])


def test_overlap_nonempty_preserves_full_padded_tails():
    position = np.zeros(MAX_LANES, np.float32)
    lane = np.zeros(MAX_LANES, np.float32)
    force = np.zeros(MAX_LANES, bool)
    position[:2] = [0.0, 0.1]
    lane[:2] = [0.0, 0.0]
    lane[2:] = np.arange(2, MAX_LANES, dtype=np.float32) + 100.0
    force[2:] = True
    (out_lane, out_force), order = resolve_overlaps(position, lane, force, 2)
    np.testing.assert_array_equal(np.asarray(out_lane)[2:].view(np.uint32), lane[2:].view(np.uint32))
    np.testing.assert_array_equal(np.asarray(out_force)[2:], force[2:])
    np.testing.assert_array_equal(np.asarray(order)[2:], np.arange(2, MAX_LANES))
