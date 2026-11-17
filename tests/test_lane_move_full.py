"""Frozen full-state rows emitted by the mechanical pinned-Rust MoveState oracle."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from globalsim_jax.lane_move import MAX_MODIFIERS, MoveContext, MoveParameters, MoveState, step, step_checked


ROWS = json.loads((Path(__file__).parent / "fixtures" / "lane_move_oracle.json").read_text())["rows"]
PARAMETERS = MoveParameters(*[np.float32(value) for value in (0.0010000000474974513, 0.30000001192092896, 0.019999999552965164, 0.05000000074505806, 1.5)])


def _f32(bits):
    return np.asarray(bits, np.uint32).view(np.float32)


def _case(row):
    value = row["input"]
    state = MoveState(*[_f32(bits) for bits in value[:4]], np.int32(value[4]))
    modifiers = np.zeros(MAX_MODIFIERS, np.float32)
    modifiers[: len(value[15])] = [_f32(bits) for bits in value[15]]
    context = MoveContext(
        np.bool_(value[5]), _f32(value[6]), _f32(value[7]), np.int32(value[8]), np.int32(value[9]),
        _f32(value[10]), _f32(value[11]), np.bool_(value[13]), np.bool_(value[14]), modifiers, np.int32(len(value[15])),
    )
    return state, context, _f32(value[12]), PARAMETERS


def _assert_rows(result, moved):
    bits = np.stack([np.asarray(result.lane), np.asarray(result.target), np.asarray(result.speed), np.asarray(result.continue_time)], axis=1).view(np.uint32)
    expected = np.asarray([row["out"][:4] for row in ROWS], np.uint32)
    np.testing.assert_array_equal(bits, expected)
    np.testing.assert_array_equal(np.asarray(result.continue_direction), [row["out"][4] for row in ROWS])
    np.testing.assert_array_equal(np.asarray(moved), [bool(row["out"][5]) for row in ROWS])


def test_frozen_rust_full_state_eager_and_jitted_vmap():
    values = list(zip(*[_case(row) for row in ROWS], strict=True))
    stack = lambda group: jax.tree_util.tree_map(lambda *leaf: jnp.stack(leaf), *group)
    state, context, dt, parameters = stack(values[0]), stack(values[1]), jnp.stack(values[2]), stack(values[3])
    batched = jax.vmap(step)
    for result, moved in (batched(state, context, dt, parameters), jax.jit(batched)(state, context, dt, parameters)):
        _assert_rows(result, moved)


def test_checked_python_scalar_canonicalization_and_int32_rejection():
    state, context, dt, parameters = _case(ROWS[0])
    python_context = MoveContext(True, context.position, context.power, 1, 2, context.first_move_point, context.max_lane, True, False, context.modifiers, 0)
    expected = step(state, MoveContext(np.bool_(True), context.position, context.power, np.int32(1), np.int32(2), context.first_move_point, context.max_lane, np.bool_(True), np.bool_(False), context.modifiers, np.int32(0)), dt, parameters)
    actual = step_checked(state, python_context, dt, parameters)
    for left, right in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    with pytest.raises(ValueError, match="supported integer range"):
        step_checked(state, MoveContext(False, context.position, context.power, 1, 2**31, context.first_move_point, context.max_lane, True, False, context.modifiers, 0), dt, parameters)
    with pytest.raises(ValueError, match="direction flag"):
        step_checked(MoveState(state.lane, state.target, state.speed, state.continue_time, 2), context, dt, parameters)
