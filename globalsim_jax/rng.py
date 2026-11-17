"""Pure JAX port of Rust `src/retail_random.rs` (System.Random compatible)."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import lax
from .precision import require_x64

_MAX = 2147483647

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RetailRng:
    state: jax.Array
    a: jax.Array
    b: jax.Array
    draw_count: jax.Array

def seed(value: int | jax.Array) -> RetailRng:
    """Initialize exactly as the Rust retail RNG; accepts signed int32 seeds."""
    require_x64()
    value = jnp.asarray(value, jnp.int32)
    magnitude = jnp.where(value == jnp.int32(-2147483648), jnp.int32(_MAX), jnp.abs(value))
    previous = jnp.int32(161803398) - magnitude
    state = jnp.zeros(56, dtype=jnp.int32).at[55].set(previous)
    def fill(n, carry):
        state, previous, current = carry
        slot = (21 * n) % 55
        state = state.at[slot].set(current)
        current = previous - current
        current = jnp.where(current < 0, current + jnp.int32(_MAX), current)
        return state, state[slot], current
    state, _, _ = lax.fori_loop(1, 55, fill, (state, previous, jnp.int32(1)))
    def shuffle(_, state):
        def one(slot, state):
            value = state[slot] - state[1 + (slot + 30) % 55]
            value = jnp.where(value < 0, value + jnp.int32(_MAX), value)
            return state.at[slot].set(value)
        return lax.fori_loop(1, 56, one, state)
    state = lax.fori_loop(0, 4, shuffle, state)
    return RetailRng(state, jnp.int32(0), jnp.int32(21), jnp.int32(0))

def next_int(rng: RetailRng) -> tuple[RetailRng, jax.Array]:
    a = jnp.where(rng.a < 55, rng.a + 1, 1).astype(jnp.int32)
    b = jnp.where(rng.b < 55, rng.b + 1, 1).astype(jnp.int32)
    value = rng.state[a] - rng.state[b]
    value = jnp.where(value == _MAX, value - 1, value)
    value = jnp.where(value < 0, value + _MAX, value).astype(jnp.int32)
    state = rng.state.at[a].set(value)
    return RetailRng(state, a, b, rng.draw_count + 1), value

def next_double(rng: RetailRng) -> tuple[RetailRng, jax.Array]:
    rng, value = next_int(rng)
    return rng, value.astype(jnp.float64) * (1.0 / float(_MAX))

def integer(rng: RetailRng, lower: int | jax.Array, upper: int | jax.Array) -> tuple[RetailRng, jax.Array]:
    """[lower, upper), including the Rust wide-range path and draw ordering."""
    if not isinstance(lower, jax.core.Tracer) and not isinstance(upper, jax.core.Tracer) and int(lower) > int(upper):
        raise ValueError("Random bounds must be ordered")
    lower, upper = jnp.asarray(lower, jnp.int32), jnp.asarray(upper, jnp.int32)
    width = upper.astype(jnp.int64) - lower.astype(jnp.int64)
    def normal(r):
        r, sample = next_double(r)
        return r, (sample * width.astype(jnp.float64)).astype(jnp.int64).astype(jnp.int32) + lower
    def wide(r):
        r, value = next_int(r)
        r, sign = next_int(r)
        value = jnp.where(sign % 2 == 0, -value.astype(jnp.int64), value.astype(jnp.int64))
        sample = (value + (_MAX - 1)) / float(2 * _MAX - 1)
        return r, (sample * width.astype(jnp.float64)).astype(jnp.int64).astype(jnp.int32) + lower
    return lax.cond(width <= _MAX, normal, wide, rng)

def real(rng: RetailRng, lower, upper) -> tuple[RetailRng, jax.Array]:
    rng, sample = next_double(rng)
    lower, upper = jnp.asarray(lower, jnp.float32), jnp.asarray(upper, jnp.float32)
    value = sample.astype(jnp.float32) * jnp.abs(upper - lower) + lower
    # Deliberately sequential comparisons rather than clip: Rust permits
    # reversed float bounds and its first/second branch order is observable.
    return rng, jnp.where(value < lower, lower, jnp.where(value > upper, upper, value))

def real_distance(rng: RetailRng, lower, upper) -> tuple[RetailRng, jax.Array]:
    rng, sample = next_double(rng)
    lower, upper = jnp.asarray(lower, jnp.float32), jnp.asarray(upper, jnp.float32)
    value = (sample.astype(jnp.float32).astype(jnp.float64) * jnp.abs(upper - lower).astype(jnp.float64) + lower.astype(jnp.float64)).astype(jnp.float32)
    return rng, jnp.where(value < lower, lower, jnp.where(value > upper, upper, value))
