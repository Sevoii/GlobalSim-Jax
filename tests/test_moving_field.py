"""Frozen, source-extracted differential rows for supplied-target field movement.

The fixture is emitted by ``tools/moving_field_oracle.rs``.  It is deliberately
not a Python-generated expected trace: every state channel is raw Rust storage
bits, including the three f64 blocking timers.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from globalsim_jax.field import FieldState, prepare_blocking
from globalsim_jax.lane_move import MAX_MODIFIERS, MoveParameters
from globalsim_jax.lane_queries import BlockParameters as SideParameters, side_room
from globalsim_jax.moving_field import MovingFieldState, run, step
from globalsim_jax.physics import Parameters, advance_start_delay
from globalsim_jax.restricted import RestrictedState


TRACE = json.loads(
    (Path(__file__).parent / "fixtures" / "moving_field_trace.json").read_text(
        encoding="utf-8"
    )
)
DT = np.asarray(TRACE["dt_bits"], np.uint32).view(np.float32).item()
MOVE_PARAMETERS = MoveParameters(*[
    np.float32(value)
    for value in (
        0.0010000000474974513,
        0.30000001192092896,
        0.019999999552965164,
        0.05000000074505806,
        1.5,
    )
])
F32_COLUMNS = {
    "time": 0,
    "position": 1,
    "speed": 2,
    "hp": 3,
    "delay": 4,
    "finish_time": 8,
    "minimum": 9,
    "lane": 13,
    "target": 15,
    "lane_speed": 16,
    "continue_time": 17,
}
F64_COLUMNS = {"blocked_front_time": 10, "blocked_side_time": 11, "blocked_all_time": 12}


def _f32(bits):
    return np.asarray(bits, np.uint32).view(np.float32)


def _f64(bits):
    return np.asarray(bits, np.uint64).view(np.float64)


def _u32(value):
    return np.asarray(value, np.float32).view(np.uint32).item()


def _u64(value):
    return np.asarray(value, np.float64).view(np.uint64).item()


def _matrix(rows, column, dtype=jnp.float32):
    converter = _f64 if dtype == jnp.float64 else _f32
    return jnp.asarray([[converter(row[column]) for row in rows]], dtype)


def _initial(case):
    rows = case["initial"]
    runners = RestrictedState(
        _matrix(rows, 0),
        _matrix(rows, 1),
        _matrix(rows, 2),
        _matrix(rows, 3),
        _matrix(rows, 4),
        jnp.asarray([[bool(row[7]) for row in rows]], bool),
        jnp.asarray([[bool(row[22]) for row in rows]], bool),
        _matrix(rows, 8),
        jnp.asarray([[row[21] for row in rows]], jnp.int32),
        jnp.asarray([[row[5] for row in rows]], jnp.int32),
        jnp.asarray([[bool(row[6]) for row in rows]], bool),
    )
    field = FieldState(
        runners,
        _matrix(rows, 13),
        _matrix(rows, 9),
        _matrix(rows, 10, jnp.float64),
        _matrix(rows, 11, jnp.float64),
        _matrix(rows, 12, jnp.float64),
        jnp.asarray([bool(rows[0][14])], bool),
    )
    return MovingFieldState(
        field,
        _matrix(rows, 15),
        _matrix(rows, 16),
        _matrix(rows, 17),
        jnp.asarray([[row[18] for row in rows]], jnp.int32),
        jnp.asarray([[bool(row[19]) for row in rows]], bool),
        jnp.asarray([[bool(row[20]) for row in rows]], bool),
    )


def _inputs(batch=1):
    f32 = lambda value: jnp.full((batch, 3), np.float32(value), jnp.float32)
    return {
        "speed_stat": f32(1000.0),
        "power": f32(1000.0),
        "guts": f32(900.0),
        "strategy": jnp.ones((batch, 3), jnp.int32),
        "aptitude": f32(1.0),
        "ground": f32(1.0),
        "distance": 1200.0,
        "dt": DT,
        "max_lane": 1.0,
        "first_move_point": 375.0,
        "move_parameters": MOVE_PARAMETERS,
    }


def _assert_expected(state, expected):
    """Compare each per-runner fixture column, preserving f32/f64 NaN bits."""
    actual = {
        "time": state.field.runners.time,
        "position": state.field.runners.position,
        "speed": state.field.runners.speed,
        "hp": state.field.runners.hp,
        "delay": state.field.runners.delay,
        "finish_time": state.field.runners.finish_time,
        "minimum": state.field.minimum,
        "lane": state.field.lane,
        "target": state.target,
        "lane_speed": state.lane_speed,
        "continue_time": state.continue_time,
    }
    for runner, row in enumerate(expected):
        for name, column in F32_COLUMNS.items():
            assert _u32(actual[name][0, runner]) == row[column], (name, runner)
        for name, column in F64_COLUMNS.items():
            assert _u64(getattr(state.field, name)[0, runner]) == row[column], (name, runner)
        assert int(state.field.runners.phase[0, runner]) == row[5]
        assert int(state.field.runners.start_dash[0, runner]) == row[6]
        assert int(state.field.runners.finished[0, runner]) == row[7]
        assert int(state.continue_direction[0, runner]) == row[18]
        assert int(state.force_checks[0, runner]) == row[19]
        assert int(state.moved[0, runner]) == row[20]
        assert int(state.field.runners.ticks[0, runner]) == row[21]
        assert int(state.field.runners.timed_out[0, runner]) == row[22]
        assert int(state.field.truncated[0]) == row[14]


def _entry_observation(params, state):
    """The Rust fixture's frozen tick-entry snapshot, before runner 0 moves."""
    field = state.field
    prepared, front, gap, side, rank = prepare_blocking(
        params,
        field,
        guts=jnp.full_like(field.runners.position, 900.0),
        distance=1200.0,
        dt=DT,
    )
    del prepared
    n = field.lane.shape[1]
    roster = jnp.arange(n, dtype=jnp.int32)
    _, gate, fraction = advance_start_delay(field.runners.delay, DT)
    # Rust's adapter serializes Option::None as a NaN diagnostic payload.  The
    # accepted JAX primitive represents the same absent fraction as zero.
    fraction = jnp.where(gate, fraction, jnp.float64(jnp.nan))
    rows = []
    side_parameters = SideParameters(np.float32(1.0499999523162842), np.float32(0.1111110970377922))
    for owner in range(n):
        peer = roster[:-1] + (roster[:-1] >= owner)
        positions = jnp.take(field.runners.position, peer, axis=1)
        lanes = jnp.take(field.lane, peer, axis=1)
        positions = jnp.pad(positions, ((0, 0), (0, 32 - (n - 1))))
        lanes = jnp.pad(lanes, ((0, 0), (0, 32 - (n - 1))))
        room = jax.vmap(
            lambda owner_position, owner_lane, peer_position, peer_lane: side_room(
                owner_position,
                owner_lane,
                np.float32(1.0),
                peer_position,
                peer_lane,
                np.int32(n - 1),
                side_parameters,
            )
        )(
            field.runners.position[:, owner],
            field.lane[:, owner],
            positions,
            lanes,
        )
        rows.append(jnp.stack((
            front[:, owner].astype(jnp.uint64),
            jax.lax.bitcast_convert_type(gap[:, owner], jnp.uint32).astype(jnp.uint64),
            side[:, owner].astype(jnp.uint64),
            rank[:, owner].astype(jnp.uint64),
            room.can_in.astype(jnp.uint64),
            room.can_out.astype(jnp.uint64),
            gate[:, owner].astype(jnp.uint64),
            jax.lax.bitcast_convert_type(fraction[:, owner], jnp.uint64),
        ), axis=-1))
    return jnp.stack(rows, axis=1)


def _state_scan(params, initial, max_ticks, inputs=None):
    if inputs is None:
        inputs = _inputs()

    def body(carry, _):
        next_state = step(params, carry, max_ticks=max_ticks, **inputs)
        return next_state, next_state

    return jax.lax.scan(body, initial, None, length=8)


def _at_tick(trace, tick):
    return jax.tree_util.tree_map(lambda value: value[tick], trace)


def _eager_history(params, initial, cap):
    states = []
    state = initial
    for _ in range(8):
        state = step(params, state, max_ticks=cap, **_inputs())
        states.append(state)
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *states)


def _assert_tree_bytes(left, right):
    for actual, expected in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()


def _observation_bits(rows):
    """Rust uses signed -1 for the absent front sentinel; pack it as u64."""
    return np.asarray(
        [[int(value) & ((1 << 64) - 1) for value in row] for row in rows],
        np.uint64,
    )


def test_frozen_moving_field_rows_eager_and_jitted_scan():
    params = Parameters.default()
    for case in TRACE["cases"]:
        state = _initial(case)
        for tick, expected in enumerate(case["rows"]):
            if not bool(state.field.truncated[0]):
                np.testing.assert_array_equal(
                    np.asarray(_entry_observation(params, state)[0]),
                    _observation_bits(expected["observation"]),
                    err_msg=case["name"],
                )
            state = step(params, state, max_ticks=case["max_ticks"], **_inputs())
            _assert_expected(state, expected["state"])

    initial = jax.tree_util.tree_map(
        lambda *leaves: jnp.concatenate(leaves, axis=0),
        *[_initial(case) for case in TRACE["cases"]],
    )
    caps = jnp.asarray([[case["max_ticks"]] for case in TRACE["cases"]], jnp.int32)
    _, history = jax.jit(
        lambda value: _state_scan(params, value, caps, _inputs(batch=len(TRACE["cases"])))
    )(initial)
    for batch, case in enumerate(TRACE["cases"]):
        for tick, expected in enumerate(case["rows"]):
            expected_state = jax.tree_util.tree_map(
                lambda leaf: leaf[batch : batch + 1], _at_tick(history, tick)
            )
            _assert_expected(expected_state, expected["state"])


def test_two_batch_moving_field_scan_matches_independent_scalar_traces():
    params = Parameters.default()
    cases = [TRACE["cases"][0], TRACE["cases"][-1]]
    scalar = [_eager_history(params, _initial(case), case["max_ticks"]) for case in cases]
    initial = jax.tree_util.tree_map(lambda left, right: jnp.concatenate((left, right), axis=0), *[_initial(case) for case in cases])
    inputs = _inputs(batch=2)
    caps = jnp.asarray([[case["max_ticks"]] for case in cases], jnp.int32)

    def body(carry, _):
        next_state = step(params, carry, max_ticks=caps, **inputs)
        return next_state, next_state

    _, history = jax.jit(lambda value: jax.lax.scan(body, value, None, length=8))(initial)
    for tick in range(8):
        actual = _at_tick(history, tick)
        for batch, expected_trace in enumerate(scalar):
            expected = _at_tick(expected_trace, tick)
            per_batch = jax.tree_util.tree_map(lambda leaf: leaf[batch : batch + 1], actual)
            _assert_tree_bytes(per_batch, expected)


def test_terminal_fixture_rows_are_full_state_bytewise_padding():
    params = Parameters.default()
    for name in ("first_finish", "all_timeout"):
        case = next(value for value in TRACE["cases"] if value["name"] == name)
        history = _eager_history(params, _initial(case), case["max_ticks"])
        flags = [row["state"][0][14] for row in case["rows"]]
        first = flags.index(1)
        terminal = _at_tick(history, first)
        for tick in range(first + 1, 8):
            _assert_tree_bytes(_at_tick(history, tick), terminal)


def test_oracle_counterfactual_witnesses_select_source_schedule():
    """The frozen adapter proves these outputs differ under the named bad schedule."""
    params = Parameters.default()
    for case in TRACE["cases"]:
        witness = case["rows"][0].get("witness")
        if witness is None:
            continue
        result = step(params, _initial(case), max_ticks=case["max_ticks"], **_inputs())
        actual = _u32(result.field.lane[0, 1])
        assert actual == witness["correct_lane_bits"], (case["name"], witness["kind"])
        assert actual != witness["wrong_lane_bits"], (case["name"], witness["kind"])
