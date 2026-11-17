"""Checked moving-field boundaries and the public scan, using independent Rust state."""
from dataclasses import replace
import jax
import numpy as np
import pytest
from globalsim_jax.physics import Parameters
from globalsim_jax.restricted import RestrictedConfig
from globalsim_jax.moving_field import initialize, run_checked
from tests.test_moving_field import TRACE, DT, MOVE_PARAMETERS, _initial, _inputs, _assert_expected


def _call(state, **overrides):
    kwargs = {key: value for key, value in _inputs().items() if key not in ('distance', 'dt')}
    kwargs['config'] = RestrictedConfig(distance=1200., dt=float(DT), max_ticks=3)
    kwargs.update(overrides)
    return run_checked(Parameters.default(), state, **kwargs)


def test_checked_scan_matches_timeout_source_rows():
    case = next(row for row in TRACE['cases'] if row['name'] == 'all_timeout')
    final, history = _call(_initial(case))
    for tick in range(3):
        state = jax.tree_util.tree_map(lambda x: x[tick], history)
        _assert_expected(state, case['rows'][tick]['state'])
    _assert_expected(final, case['rows'][2]['state'])


def test_checked_rejects_state_ranges_and_source_query_subnormals():
    state = _initial(TRACE['cases'][0])
    for name in ('lane_speed', 'continue_time'):
        with pytest.raises(ValueError, match='nonnegative'):
            _call(replace(state, **{name: np.full((1, 3), -1., np.float32)}))
    with pytest.raises(ValueError, match='nonnegative'):
        _call(replace(state, field=replace(state.field, minimum=np.full((1, 3), -1., np.float32))))
    tiny = np.finfo(np.float32).tiny
    lane = np.asarray([[np.nextafter(tiny, np.float32(np.inf)), tiny, 1.]], np.float32)
    with pytest.raises(ValueError, match='subnormal'):
        _call(replace(state, field=replace(state.field, lane=lane)))
    with pytest.raises(ValueError, match='subnormal'):
        _call(replace(state, target=np.full((1, 3), tiny / 2, np.float32)))
    with pytest.raises(ValueError):
        _call(state, max_lane=np.asarray([1.]))
    with pytest.raises(ValueError):
        _call(state, max_lane=1e300)
    with pytest.raises(ValueError):
        _call(state, first_move_point=1+2j)


def test_attach_rejects_bad_targets_before_narrowing_and_checked_terminal_policy():
    state = _initial(TRACE['cases'][0])
    for target in (np.full((1, 3), 1e300), np.ones((1, 3), bool),
                   np.ones((1, 3), complex), np.zeros((3,), np.float32)):
        with pytest.raises(ValueError):
            initialize(state.field, target)
    terminal = _initial(next(c for c in TRACE['cases'] if c['name'] == 'first_finish'))
    position = np.asarray(terminal.field.runners.position).copy()
    position[0, 0] = 1201.
    runners = replace(terminal.field.runners, position=position)
    with pytest.raises(ValueError, match='truncated'):
        _call(replace(terminal, field=replace(terminal.field, runners=runners)))
