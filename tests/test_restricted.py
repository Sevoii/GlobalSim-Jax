import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from globalsim_jax.physics import Parameters
from globalsim_jax.restricted import RestrictedConfig, RestrictedState, initialize, run, run_checked, step


FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "restricted_trace.json").read_text(encoding="utf-8"))


def f32(bits):
    return jnp.asarray(np.asarray(bits, np.uint32).view(np.float32))


def bits(value):
    return np.asarray(value, np.float32).view(np.uint32).item()


def state(row):
    time, pos, speed, hp, delay, phase, dash, finished = row
    scalar = lambda value, dtype=jnp.float32: jnp.asarray([[value]], dtype)
    return RestrictedState(
        scalar(f32(time)), scalar(f32(pos)), scalar(f32(speed)), scalar(f32(hp)), scalar(f32(delay)),
        scalar(bool(finished), bool), scalar(False, bool), scalar(jnp.nan), scalar(0, jnp.int32),
        scalar(phase, jnp.int32), scalar(bool(dash), bool),
    )


def kwargs(case):
    speed, guts, power, strategy, aptitude, ground, distance = case["inputs"]
    return dict(
        speed_stat=jnp.asarray([[speed]], jnp.float32), power=jnp.asarray([[power]], jnp.float32),
        guts=jnp.asarray([[guts]], jnp.float32), strategy=jnp.asarray([[strategy]], jnp.int32),
        aptitude=jnp.asarray([[aptitude]], jnp.float32), ground=jnp.asarray([[ground]], jnp.float32),
        distance=float(distance), dt=FIXTURE["dt"],
    )


def channels(value):
    return [
        bits(value.time[0, 0]), bits(value.position[0, 0]), bits(value.speed[0, 0]),
        bits(value.hp[0, 0]), bits(value.delay[0, 0]), int(value.phase[0, 0]),
        int(value.start_dash[0, 0]), int(value.finished[0, 0]), bits(value.finish_time[0, 0]),
    ]


def trace_state(trace, tick):
    return RestrictedState(*[getattr(trace, name)[tick] for name in trace.__dataclass_fields__])


def equal_state_field(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.dtype == np.float32:
        return np.array_equal(left.view(np.uint32), right.view(np.uint32))
    return np.array_equal(left, right)


def test_frozen_rust_restricted_traces_eager_and_jit_scan():
    parameters = Parameters.default()
    for case in FIXTURE["cases"]:
        eager = state(case["initial"])
        args = kwargs(case)
        for expected in case["rows"]:
            eager = step(parameters, eager, max_ticks=99, **args)
            assert channels(eager) == expected, case["name"]
        _, trace = jax.jit(lambda value: run(parameters, value, max_ticks=8, **args))(state(case["initial"]))
        for tick, expected in enumerate(case["rows"]):
            assert channels(trace_state(trace, tick)) == expected, (case["name"], tick)


def test_timeout_is_sticky_and_closed_gate_zero_dt_freezes_motion_phase_and_dash():
    parameters = Parameters.default()
    delayed = next(case for case in FIXTURE["cases"] if case["name"] == "delayed")
    args = kwargs(delayed)
    zero = step(parameters, state(delayed["initial"]), max_ticks=99, **(args | {"dt": 0.0}))
    initial = state(delayed["initial"])
    for name in ("position", "speed", "hp", "phase", "start_dash", "delay"):
        assert np.array_equal(np.asarray(getattr(zero, name)), np.asarray(getattr(initial, name)))
    phase = next(case for case in FIXTURE["cases"] if case["name"] == "phase")
    timed = step(parameters, state(phase["initial"]), max_ticks=1, **kwargs(phase))
    assert bool(timed.timed_out[0, 0])
    frozen = step(parameters, timed, max_ticks=1, **kwargs(phase))
    for name in timed.__dataclass_fields__:
        assert equal_state_field(getattr(frozen, name), getattr(timed, name))


def test_divergent_batch_matches_scalar_eager_and_jit_scan():
    parameters = Parameters.default()
    cases = [FIXTURE["cases"][0], FIXTURE["cases"][1], FIXTURE["cases"][4], FIXTURE["cases"][5]]
    scalar_states = [state(case["initial"]) for case in cases]
    batched = RestrictedState(*[
        jnp.concatenate((
            jnp.concatenate((getattr(scalar_states[0], name), getattr(scalar_states[1], name)), axis=1),
            jnp.concatenate((getattr(scalar_states[2], name), getattr(scalar_states[3], name)), axis=1),
        ), axis=0)
        for name in scalar_states[0].__dataclass_fields__
    ])
    scalar_args = [kwargs(case) for case in cases]
    batch_args = {
        name: jnp.concatenate((
            jnp.concatenate((scalar_args[0][name], scalar_args[1][name]), axis=1),
            jnp.concatenate((scalar_args[2][name], scalar_args[3][name]), axis=1),
        ), axis=0)
        for name in ("speed_stat", "power", "guts", "strategy", "aptitude", "ground")
    } | {"distance": scalar_args[0]["distance"], "dt": scalar_args[0]["dt"]}
    eager_batch = batched
    for _ in range(4):
        scalar_states = [step(parameters, value, max_ticks=99, **args) for value, args in zip(scalar_states, scalar_args)]
        eager_batch = step(parameters, eager_batch, max_ticks=99, **batch_args)
        for name in eager_batch.__dataclass_fields__:
            actual = np.asarray(getattr(eager_batch, name))
            for index, scalar in enumerate(scalar_states):
                batch, runner = divmod(index, 2)
                assert equal_state_field(actual[batch, runner], np.asarray(getattr(scalar, name))[0, 0])
    _, compiled = jax.jit(lambda value: run(parameters, value, max_ticks=4, **batch_args))(batched)
    scalar_traces = [jax.jit(lambda value, args=args: run(parameters, value, max_ticks=4, **args))(state(case["initial"]))[1] for case, args in zip(cases, scalar_args)]
    for name in compiled.__dataclass_fields__:
        actual = np.asarray(getattr(compiled, name))
        for index, trace in enumerate(scalar_traces):
            batch, runner = divmod(index, 2)
            assert equal_state_field(actual[:, batch, runner], np.asarray(getattr(trace, name))[:, 0, 0])


def test_checked_boundary_rejects_invalid_config_and_runner_inputs(monkeypatch):
    parameters = Parameters.default()
    base = dict(speed=[[1000.]], stamina=[[1000.]], power=[[1000.]], guts=[[900.]], strategy=[[1]], aptitude=[[1.]], ground=[[1.]], delay=[[0.]])
    with pytest.raises(ValueError):
        initialize(parameters, **base, distance=0.0)
    with pytest.raises(ValueError):
        initialize(parameters, **(base | {"ground": [[0.]]}), distance=1200.0)
    with pytest.raises(ValueError):
        initialize(parameters, **(base | {"aptitude": [[1.1]]}), distance=1200.0)
    with pytest.raises(ValueError):
        RestrictedConfig(1200.0, float("nan"), 8)
    with pytest.raises(ValueError):
        RestrictedConfig(1200.0, 0.1, 0)
    valid = initialize(parameters, **base, distance=1200.0)
    config = RestrictedConfig(1200.0, 0.1, 2)
    _, trace = run_checked(parameters, valid, config=config, speed_stat=jnp.asarray(base["speed"]), power=jnp.asarray(base["power"]), guts=jnp.asarray(base["guts"]), strategy=jnp.asarray(base["strategy"]), aptitude=jnp.asarray(base["aptitude"]), ground=jnp.asarray(base["ground"]))
    assert trace.time.shape == (2, 1, 1)
    invalid_finish = replace(valid, finish_time=jnp.zeros_like(valid.finish_time))
    with pytest.raises(ValueError, match="finish_time"):
        run_checked(parameters, invalid_finish, config=config, speed_stat=jnp.asarray(base["speed"]), power=jnp.asarray(base["power"]), guts=jnp.asarray(base["guts"]), strategy=jnp.asarray(base["strategy"]), aptitude=jnp.asarray(base["aptitude"]), ground=jnp.asarray(base["ground"]))
    invalid_dtype = replace(valid, speed=valid.speed.astype(jnp.int32))
    with pytest.raises(ValueError, match="float32"):
        run_checked(parameters, invalid_dtype, config=config, speed_stat=jnp.asarray(base["speed"]), power=jnp.asarray(base["power"]), guts=jnp.asarray(base["guts"]), strategy=jnp.asarray(base["strategy"]), aptitude=jnp.asarray(base["aptitude"]), ground=jnp.asarray(base["ground"]))
    calls = []
    monkeypatch.setattr("globalsim_jax.restricted.require_x64", lambda: calls.append(True))
    run_checked(parameters, valid, config=config, speed_stat=jnp.asarray(base["speed"]), power=jnp.asarray(base["power"]), guts=jnp.asarray(base["guts"]), strategy=jnp.asarray(base["strategy"]), aptitude=jnp.asarray(base["aptitude"]), ground=jnp.asarray(base["ground"]))
    assert calls == [True]


def test_initialize_preserves_fractional_course_distance_with_integer_stamina():
    parameters = Parameters.default()
    state = initialize(parameters, speed=[[1000]], stamina=[[1000]], power=[[1000]], guts=[[900]], strategy=[[1]], aptitude=[[1.]], ground=[[1.]], delay=[[0.]], distance=1200.5)
    expected = parameters.initial_hp(jnp.float32(1000), jnp.int32(1), jnp.float32(1200.5))
    assert bits(state.hp[0, 0]) == bits(expected)
