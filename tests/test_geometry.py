import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from globalsim_jax.geometry import CourseGeometry


def bits(x): return np.asarray(x,dtype=np.float32).view(np.uint32).item()

def test_course_10101_fixture_eager_and_jit():
    fixture=json.loads((Path(__file__).parent/'fixtures'/'retail_geometry_port.json').read_text())
    course=next(c for c in fixture['courses'] if c['course_id']==10101)
    geometry=CourseGeometry.load(10101,1200.)
    rows=[row for row in course['rows'] if 0 <= row['position'] <= 1200]
    for row in rows:
        actual=geometry.advance(row['position'],row['lane'],row['movement'],row['previous_world'])
        assert [bits(actual[0]),bits(actual[1][0]),bits(actual[1][1]),bits(geometry.lane_max)] == row['expected_bits']
    fn=jax.jit(jax.vmap(geometry.advance))
    actual=fn(jnp.array([r['position'] for r in rows]),jnp.array([r['lane'] for r in rows]),jnp.array([r['movement'] for r in rows]),jnp.array([r['previous_world'] for r in rows]))
    expected=np.asarray([r['expected_bits'][:3] for r in rows],np.uint32)
    got=np.stack((np.asarray(actual[0],np.float32).view(np.uint32),np.asarray(actual[1][:,0],np.float32).view(np.uint32),np.asarray(actual[1][:,1],np.float32).view(np.uint32)),axis=1)
    assert np.array_equal(got,expected)

def test_geometry_endpoint_and_invalid_host_inputs():
    geometry=CourseGeometry.load(10101,1200.)
    endpoint=geometry.position(1200.,0.)
    assert np.all(np.isfinite(np.asarray(endpoint)))
    for args in [(-1.,0.),(1200.1,0.),(float('nan'),0.),(0.,float('inf'))]:
        with pytest.raises(ValueError): geometry.position(*args)
    with pytest.raises(ValueError): geometry.advance(0.,0.,-1.,[0.,0.])
