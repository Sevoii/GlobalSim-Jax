"""Source-backed restricted initial stat conversions from race_solver_builder.rs."""
import jax.numpy as jnp
from .precision import require_x64


def adjust_overcap(raw):
    """Source overcap adjustment; requires the package mixed-precision mode."""
    require_x64()
    raw = jnp.asarray(raw, jnp.float64)
    return jnp.where(raw > 1200., 1200. + (raw - 1200.) / 2., raw)


def team_rank_bonus(rank):
    rank = jnp.asarray(rank, jnp.int32)
    return jnp.where(rank <= 0, 0., jnp.where(rank < 6, 50., jnp.where(rank < 11, 30., jnp.where(rank < 16, 20., jnp.where(rank < 21, 10., 0.))))).astype(jnp.float32)


def initial_base_wisdom(raw, mood):
    mood = jnp.clip(jnp.asarray(mood, jnp.int32), -2, 2)
    coefficient = (10000 + 200 * mood).astype(jnp.float32) * jnp.float32(.0001)
    return jnp.clip(jnp.asarray(raw, jnp.float32) * coefficient, 1., 2000.)


def ranked_base_stat(raw, mood, team_rank):
    """Raw -> overcap -> rank bonus -> mood -> [1, 2000] f32 clamp."""
    base = adjust_overcap(raw).astype(jnp.float32) + team_rank_bonus(team_rank)
    return initial_base_wisdom(base, mood)
