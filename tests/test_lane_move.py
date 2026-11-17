"""Copied pinned-Rust `retail_lane_port` MoveState vectors."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from globalsim_jax.lane_move import MAX_MODIFIERS, MoveContext, MoveParameters, MoveState, _approximately, step


DATA = json.loads((Path(__file__).parent / "fixtures" / "retail_lane_port.json").read_text())


def _case(row):
    modifiers = np.zeros(MAX_MODIFIERS, np.float32)
    modifiers[: len(row["modifiers"])] = row["modifiers"]
    state = MoveState(*[np.asarray(value, np.float32) for value in (row["lane"], row["target"], row["speed"], 0.0)], np.asarray(0, np.int32))
    context = MoveContext(
        np.asarray(False), np.asarray(row["position"], np.float32), np.asarray(row["power"], np.float32),
        np.asarray(row["phase"], np.int32), np.asarray(row["rank"], np.int32), np.asarray(row["first_move"], np.float32),
        np.asarray(row["max_lane"], np.float32), np.asarray(row["can_in"]), np.asarray(row["can_out"]),
        modifiers, np.asarray(len(row["modifiers"]), np.int32),
    )
    p = DATA["move_parameters"]
    parameters = MoveParameters(*[np.asarray(p[name], np.float32) for name in ("laneMoveSpeedPowCoef", "laneMoveSpeedAdd", "laneMoveSpeedBase", "laneMoveSpeedOutCoef", "laneMoveAccelMoveSpeedBaseCoef")])
    return state, context, np.asarray(row["dt"], np.float32), parameters


def test_copied_rust_move_vectors_eager_and_jit():
    rows = DATA["moves"]
    states, contexts, dts, parameters = zip(*[_case(row) for row in rows], strict=True)
    stack = lambda values: jax.tree_util.tree_map(lambda *items: jnp.stack(items), *values)
    state, context, dt, parameter = stack(states), stack(contexts), jnp.stack(dts), stack(parameters)
    batched = jax.vmap(step)
    for actual, moved in (batched(state, context, dt, parameter), jax.jit(batched)(state, context, dt, parameter)):
        result, moved = actual, moved
        bits = np.stack((np.asarray(result.lane), np.asarray(result.target), np.asarray(result.speed)), axis=1).view(np.uint32)
        np.testing.assert_array_equal(bits, np.asarray([row["expected_bits"] for row in rows], np.uint32))
        np.testing.assert_array_equal(np.asarray(moved), [row["moved"] for row in rows])


def test_approximately_preserves_rust_subnormal_floor_eager_and_jit():
    values = np.asarray([0, 1, 7, 8, 9], np.uint32).view(np.float32)
    expected = np.asarray([True, True, True, False, False])
    for actual in (_approximately(np.float32(0.0), values), jax.jit(_approximately)(np.float32(0.0), values)):
        np.testing.assert_array_equal(np.asarray(actual), expected)
