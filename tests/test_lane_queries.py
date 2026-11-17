import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from globalsim_jax.lane_queries import BlockParameters, is_target, is_target_checked, is_visible, is_visible_checked, side_room, side_room_checked
D = json.loads((Path(__file__).parent / 'fixtures' / 'lane_query_oracle.json').read_text())
P = BlockParameters(np.float32(1.0499999523162842), np.float32(0.1111110970377922))

def f(x):
    return np.asarray(x, np.uint32).view(np.float32)

def args(r, pad=False):
    a = np.asarray(r['rows'], np.uint32)
    x = np.zeros(32, np.float32)
    y = np.zeros(32, np.float32)
    if len(a):
        x[:len(a)] = a[:, 0].view(np.float32)
        y[:len(a)] = a[:, 1].view(np.float32)
    if pad:
        x[len(a):] = 0.0
        y[len(a):] = -0.01
    o = np.asarray(r['owner'], np.uint32).view(np.float32)
    return (o[0], o[1], f(r['max']), x, y, len(a))

def bits(s):
    return np.asarray([s.can_in, s.can_out, np.asarray(s.in_space).view(np.uint32), np.asarray(s.out_space).view(np.uint32), np.asarray(s.inner_blocker_lane).view(np.uint32), s.inner_blocker_index])

def test_all_frozen_side_rows_eager_jit_vmap_and_batch_scalar():
    jf = jax.jit(side_room)
    for r in D['side']:
        a = args(r, True)
        for s in (side_room(*a, P), jf(*a, P)):
            np.testing.assert_array_equal(bits(s), r['out'])
    rows = [D['side'][0], D['side'][1], D['side'][-1]]
    aa = [args(r, True) for r in rows]
    q = jax.jit(jax.vmap(side_room))(np.array([z[0] for z in aa]), np.array([z[1] for z in aa]), np.array([z[2] for z in aa]), np.stack([z[3] for z in aa]), np.stack([z[4] for z in aa]), np.array([z[5] for z in aa], np.int32), BlockParameters(jnp.full(3, P.side_distance), jnp.full(3, P.side_lane)))
    for i, r in enumerate(rows):
        np.testing.assert_array_equal(bits(jax.tree_util.tree_map(lambda x: x[i], q)), r['out'])

def test_all_visibility_and_overtaking_rows_eager_jit_vmap():
    v = np.asarray([r['in'] for r in D['visible']], np.uint32).view(np.float32)
    for fn in (jax.vmap(is_visible), jax.jit(jax.vmap(is_visible))):
        np.testing.assert_array_equal(np.asarray(fn(v[:, 0], v[:, 1], v[:, 2])), [r['out'] for r in D['visible']])
    o = D['overtake']
    z = np.asarray([r['in'][:5] for r in o], np.uint64).view(np.float64)
    b = np.asarray([r['in'][5:] for r in o], bool)
    for fn in (jax.vmap(is_target), jax.jit(jax.vmap(is_target))):
        np.testing.assert_array_equal(np.asarray(fn(z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4], b[:, 0], b[:, 1])), [r['out'] for r in o])

def test_checked_boundaries_x64_and_numeric_domain():
    with pytest.raises(ValueError):
        side_room_checked(0.0, 0.0, 1.0, np.zeros(33), np.zeros(33), P)
    with pytest.raises(ValueError):
        side_room_checked(0.0, 0.0, 1.0, np.array([np.inf]), np.zeros(1), P)
    with pytest.raises(ValueError):
        is_target_checked(1.0, 1.0, 2e-40, 5.0, 1e-40, np.bool_(True), np.bool_(False))
    with pytest.raises(ValueError):
        is_visible_checked(0.0, 0.0, np.float32(1e-40))
    assert not bool(is_visible_checked(0.0, 0.0, 1e-100))
