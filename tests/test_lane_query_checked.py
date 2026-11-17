import jax
import numpy as np
import pytest
from globalsim_jax.lane_queries import BlockParameters, is_target_checked, is_visible_checked, side_room_checked
P = BlockParameters(np.float32(1.05), np.float32(0.11))

def test_checked_rejects_shapes_complex_nonfinite_and_overflow():
    with pytest.raises(ValueError):
        side_room_checked(0.0, 0.0, 1.0, np.zeros((1, 1)), np.zeros(1), P)
    with pytest.raises(ValueError):
        side_room_checked(0.0, 0.0, 1.0, np.zeros(33), np.zeros(33), P)
    with pytest.raises(ValueError):
        is_visible_checked(1 + 2j, 0.0, 1.0)
    with pytest.raises(ValueError):
        is_target_checked(1.0, 1.0, 1.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        is_visible_checked(np.inf, 0.0, 1.0)
    with pytest.raises(ValueError):
        side_room_checked(float(np.finfo(np.float32).max) * 2, 0.0, 1.0, [], [], P)
    with pytest.raises(ValueError):
        is_target_checked(float(np.finfo(np.float32).max) * 2, 1.0, 1.0, 1.0, 1.0, np.bool_(True), np.bool_(False))

def test_checked_subnormal_and_underflow_zero_profile():
    tiny = np.finfo(np.float32).tiny
    with pytest.raises(ValueError):
        is_visible_checked(tiny / 2, 0.0, 1.0)
    with pytest.raises(ValueError):
        side_room_checked(tiny, 0.0, 1.0, [np.nextafter(tiny, np.float32(0))], [0.0], P)
    assert not bool(is_visible_checked(0.0, 0.0, 1e-100))
    assert not bool(is_visible_checked(0.0, 0.0, -1e-100))
    assert not bool(is_target_checked(0.0, 1.0, 1e-100, 1e-100, 1e-100, np.bool_(False), np.bool_(False)))
    owner = np.nextafter(np.float32(np.finfo(np.float32).tiny), np.float32(np.inf))
    with pytest.raises(ValueError, match='subnormal'):
        side_room_checked(0.0, owner, 1.0, [0.0], [np.float32(np.finfo(np.float32).tiny)], P)
    side_room_checked(0.0, np.float32(np.finfo(np.float32).tiny), 1.0, [0.0], [0.0], P)

def test_checked_requires_x64_and_restores():
    old = jax.config.jax_enable_x64
    try:
        jax.config.update('jax_enable_x64', False)
        with pytest.raises(RuntimeError):
            is_visible_checked(0.0, 0.0, 1.0)
    finally:
        jax.config.update('jax_enable_x64', old)

def test_checked_positive_outputs_match_frozen_source_rows():
    from tests.test_lane_queries import D, args, bits, P as frozen_p
    row = D['side'][0]
    a = args(row)
    actual = side_room_checked(a[0], a[1], a[2], a[3][:a[5]], a[4][:a[5]], frozen_p)
    np.testing.assert_array_equal(bits(actual), row['out'])

def test_checked_wrappers_match_every_valid_frozen_rust_row():
    from tests.test_lane_queries import D, args, bits, f
    source_p = BlockParameters(np.float32(1.0499999523162842), np.float32(0.1111110970377922))
    for row in D['side']:
        a = args(row)
        actual = side_room_checked(a[0], a[1], a[2], a[3][:a[5]], a[4][:a[5]], source_p)
        np.testing.assert_array_equal(bits(actual), row['out'])
    for row in D['visible']:
        values = np.asarray(row['in'], np.uint32).view(np.float32)
        assert bool(is_visible_checked(*values)) is bool(row['out'])
    for row in D['overtake']:
        values = row['in']
        numeric = np.asarray(values[:5], np.uint64).view(np.float64)
        assert bool(is_target_checked(*numeric, np.bool_(values[5]), np.bool_(values[6]))) is bool(row['out'])
