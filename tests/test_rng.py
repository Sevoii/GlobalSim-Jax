import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from globalsim_jax.rng import seed, next_int, next_double, integer, real, real_distance

FIX=Path(__file__).parent/'fixtures'/'retail_rng_dotnet.json'
def take(rng, fn, n):
    values=[]
    for _ in range(n): rng,value=fn(rng); values.append(value)
    return rng,values
def test_dotnet_fixture_exact_sequences():
    for row in json.loads(FIX.read_text()):
        _,values=take(seed(row['seed']),next_int,len(row['integers']))
        assert [int(x) for x in values] == row['integers']
        _,values=take(seed(row['seed']),next_double,len(row['doubles']))
        assert [float(x) for x in values] == row['doubles']
        rng=seed(row['seed']); actual=[]
        for _ in row['wide']: rng,value=integer(rng,-2147483648,2147483647); actual.append(int(value))
        assert actual==row['wide']
def test_zero_width_and_wrapper_draw_consumption():
    rng=seed(953795537); rng,value=integer(rng,5,5); assert int(value)==5
    rng,value=real(rng,12.,12.); assert float(value)==12.; assert int(rng.draw_count)==2
    distance_rng,distance=real_distance(seed(0),-10000.3,10000.7); simple_rng,simple=real(seed(0),-10000.3,10000.7)
    assert int(distance_rng.draw_count)==int(simple_rng.draw_count)==1
    assert float(distance)!=float(simple)  # experimental wrapper uses a different operation order
    _, reversed_real=real(seed(0),2.,1.); _, reversed_distance=real_distance(seed(0),2.,1.)
    assert float(reversed_real)==float(reversed_distance)==1.  # Rust's ordered clamp comparisons
def test_jit_and_batched_independence_with_divergent_draw_counts():
    @jax.jit
    def step(rng,wide): return jax.lax.cond(wide,lambda r: integer(r,-2147483648,2147483647),lambda r: integer(r,0,10),rng)
    states=jax.vmap(seed)(jnp.array([0,1,-1],jnp.int32)); flags=jnp.array([False,True,False])
    batched=jax.jit(jax.vmap(step))(states,flags)
    after, values=batched; jax.block_until_ready(values)
    scalar=[step(seed(s),bool(w)) for s,w in zip([0,1,-1],flags)]
    assert np.array_equal(np.asarray(values),np.asarray([x[1] for x in scalar]))
    assert np.array_equal(np.asarray(after.draw_count),np.asarray([x[0].draw_count for x in scalar]))
def test_invalid_bounds_are_rejected_eagerly():
    with pytest.raises(ValueError): integer(seed(0),3,2)
