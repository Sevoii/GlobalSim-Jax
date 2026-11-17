"""Validated flat/no-field movement kernel for the deliberately narrow M2 profile.

``step`` and ``run`` are prevalidated JIT kernels. Applications should use
``run_checked`` (and ``initialize``) at their host boundary before tracing.
"""
from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
import numpy as np
from .physics import Parameters, advance_start_delay, finish_time, target_speed_indices, phase_boundaries
from .precision import require_x64


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RestrictedState:
    time: jax.Array
    position: jax.Array
    speed: jax.Array
    hp: jax.Array
    delay: jax.Array
    finished: jax.Array
    timed_out: jax.Array
    finish_time: jax.Array
    ticks: jax.Array
    phase: jax.Array
    start_dash: jax.Array


MAX_RUNNERS = 32


@dataclass(frozen=True)
class RestrictedConfig:
    """Host-checked frozen scalar configuration for the restricted profile."""
    distance: float
    dt: float
    max_ticks: int

    def __post_init__(self):
        if not math.isfinite(self.distance) or self.distance <= 0:
            raise ValueError("distance must be finite and positive")
        if not math.isfinite(self.dt) or self.dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        if not isinstance(self.max_ticks, int) or self.max_ticks <= 0:
            raise ValueError("max_ticks must be a positive integer")


def _is_tracer(value):
    return isinstance(value, jax.core.Tracer)


def _checked_arrays(*values):
    arrays = [jnp.asarray(value) for value in values]
    shape = arrays[0].shape
    if len(shape) != 2 or not shape[0] or not shape[1]:
        raise ValueError("restricted inputs require nonempty [batch, runner] arrays")
    if shape[1] > MAX_RUNNERS or any(array.shape != shape for array in arrays):
        raise ValueError("restricted inputs require equal [batch, runner] arrays with runner capacity <= 32")
    if any(_is_tracer(value) for value in values):
        raise ValueError("checked host entry does not accept traced runner inputs")
    if not all(bool(jnp.all(jnp.isfinite(array))) for array in arrays):
        raise ValueError("restricted runner inputs must be finite")
    return arrays


def _validate_profile(*, speed, stamina, power, guts, strategy, aptitude, ground, delay):
    arrays = _checked_arrays(speed, stamina, power, guts, strategy, aptitude, ground, delay)
    _, _, _, _, strategy_a, aptitude_a, ground_a, _ = arrays
    if bool(jnp.any((strategy_a < 1) | (strategy_a > 5))) or not bool(jnp.all(strategy_a == jnp.floor(strategy_a))):
        raise ValueError("restricted loop supports integer strategy values 1..5")
    if bool(jnp.any(ground_a <= 0)):
        raise ValueError("ground must be positive")
    # The Rust concepts are separate. This frozen profile only has evidence for 1.0/1.0.
    if not bool(jnp.all(aptitude_a == 1.0)):
        raise ValueError("restricted profile supports only unity speed and acceleration-distance aptitude")
    return arrays

def initialize(params, *, speed, stamina, power, guts, strategy, aptitude, ground, delay, distance):
    """Create a validated [batch, runner] state for a positive flat distance."""
    require_x64()
    config = RestrictedConfig(float(distance), 0.0, 1)
    arrays = _validate_profile(speed=speed, stamina=stamina, power=power, guts=guts,
                              strategy=strategy, aptitude=aptitude, ground=ground, delay=delay)
    speed_a, stamina_a, _, _, strategy_a, _, _, delay_a = arrays
    hp = jax.vmap(jax.vmap(params.initial_hp))(
        stamina_a.astype(jnp.float32), strategy_a.astype(jnp.int32),
        jnp.full(stamina_a.shape, config.distance, jnp.float32),
    )
    shape = speed_a.shape
    return RestrictedState(jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32), hp, delay_a.astype(jnp.float32), jnp.zeros(shape, bool), jnp.zeros(shape, bool), jnp.full(shape, jnp.nan, jnp.float32), jnp.zeros(shape, jnp.int32), jnp.zeros(shape, jnp.int32), jnp.ones(shape, bool))


def _validate_state_and_inputs(state, *, speed_stat, power, guts, strategy, aptitude, ground):
    state_arrays = [getattr(state, name) for name in state.__dataclass_fields__]
    shape = state.time.shape
    if len(shape) != 2 or not shape[0] or not shape[1] or shape[1] > MAX_RUNNERS:
        raise ValueError("state must contain nonempty [batch, runner] arrays within capacity")
    if any(array.shape != shape for array in state_arrays):
        raise ValueError("all state fields must have the same [batch, runner] shape")
    float_names = ("time", "position", "speed", "hp", "delay", "finish_time")
    for name in float_names:
        value = np.asarray(getattr(state, name))
        if value.dtype != np.float32:
            raise ValueError(f"state.{name} must have float32 dtype")
    for name in ("finished", "timed_out", "start_dash"):
        if np.asarray(getattr(state, name)).dtype != np.bool_:
            raise ValueError(f"state.{name} must have bool dtype")
    for name in ("ticks", "phase"):
        if not np.issubdtype(np.asarray(getattr(state, name)).dtype, np.integer):
            raise ValueError(f"state.{name} must have integer dtype")
    finite_fields = ("time", "position", "speed", "hp", "delay")
    if any(not np.all(np.isfinite(np.asarray(getattr(state, name)))) for name in finite_fields):
        raise ValueError("state numeric fields must be finite")
    if (np.any(np.asarray(state.time) < 0) or np.any(np.asarray(state.position) < 0)
            or np.any(np.asarray(state.speed) < 0) or np.any(np.asarray(state.hp) < 0)
            or np.any(np.asarray(state.ticks) < 0) or np.any((np.asarray(state.phase) < 0) | (np.asarray(state.phase) > 2))):
        raise ValueError("state contains out-of-range time, position, speed, HP, ticks, or phase")
    finished = np.asarray(state.finished)
    finish = np.asarray(state.finish_time)
    if np.any(~finished & ~np.isnan(finish)) or np.any(finished & ~np.isfinite(finish)):
        raise ValueError("finish_time must be NaN exactly for unfinished lanes and finite for finished lanes")
    inputs = _validate_profile(speed=speed_stat, stamina=speed_stat, power=power, guts=guts,
                               strategy=strategy, aptitude=aptitude, ground=ground,
                               delay=jnp.zeros_like(speed_stat))
    if inputs[0].shape != shape:
        raise ValueError("runner input arrays must match state shape")


def step(params, state, *, speed_stat, power, guts, strategy, aptitude, ground, distance, dt, max_ticks,
         minimum_override=None, cap_override=None, update_mask=None):
    """One prevalidated ordered tick. Use :func:`run_checked` at host boundary."""
    # Match field scheduling: an exact-goal lane is not selected again even
    # though the isolated Rust solver's strict-crossing flag can remain false.
    active = ~(state.finished | state.timed_out) & (state.position < distance)
    if update_mask is not None:
        active = active & jnp.asarray(update_mask, bool)
    raw_delay, gate, fraction = advance_start_delay(state.delay, dt)
    delay = jnp.where(active, raw_delay, state.delay)
    section, target_phase = target_speed_indices(state.position, distance)
    # Rust selects target/acceleration with the pre-movement phase, then
    # refreshes the phase after HP spending and before moving position.
    phase = state.phase
    base = jax.vmap(jax.vmap(params.base_target_speed))(speed_stat, strategy, target_phase, aptitude, jnp.full_like(state.position, distance))
    ordinary_minimum = jax.vmap(jax.vmap(params.minimum_speed))(guts, jnp.full_like(state.position, distance))
    minimum = jnp.where(state.start_dash, jnp.float32(3.), ordinary_minimum) if minimum_override is None else jnp.asarray(minimum_override, jnp.float32)
    target = jax.vmap(jax.vmap(params.target_speed))(base, jnp.zeros_like(base), jnp.ones_like(base), jnp.zeros_like(base), power, jnp.zeros_like(active), jnp.zeros_like(base), jnp.zeros_like(active), state.hp > 0, minimum, jnp.zeros((base.shape[0], base.shape[1], 0), jnp.float32))
    accel_up = jax.vmap(jax.vmap(params.acceleration))(power, strategy, phase, ground, aptitude, jnp.zeros_like(active), state.start_dash, jnp.zeros((base.shape[0], base.shape[1], 0), jnp.float32))
    decel = jnp.where(state.hp <= 0, -1.2, jnp.where(state.speed > target, jnp.where(phase == 0, -1.2, jnp.where(phase == 1, -.8, -1.0)), accel_up))
    next_speed = state.speed + decel * jnp.asarray(dt, jnp.float32)
    next_speed = jnp.where(state.speed <= target, jnp.minimum(next_speed, target), jnp.maximum(next_speed, target))
    dash_limit = params.start_dash_speed(distance)
    next_speed = jnp.where(state.start_dash, jnp.minimum(next_speed, dash_limit), next_speed)
    if cap_override is not None:
        next_speed = jnp.minimum(next_speed, jnp.asarray(cap_override, jnp.float32))
    next_speed = jnp.maximum(next_speed, minimum)
    moving = active & gate
    previous = state.position
    moved = (next_speed * jnp.asarray(dt, jnp.float32)).astype(jnp.float64) * fraction
    position = jnp.where(moving, state.position + moved.astype(jnp.float32), state.position)
    course_base = jnp.full_like(next_speed, params.course_speed(distance))
    drain = jax.vmap(jax.vmap(params.hp_per_second))(next_speed, course_base, ground, guts, phase, jnp.ones_like(position))
    dash_drain = jnp.full_like(drain, 20.)
    dash_drain = dash_drain * ground
    hp = jnp.where(moving, jnp.maximum(state.hp - jnp.where(state.start_dash, dash_drain, drain) * jnp.asarray(dt, jnp.float32), 0.), state.hp)
    # M2 follows the field scheduling boundary: exact goal ends this wrapper lane.
    crossed = moving & (previous < distance) & (position >= distance)
    result = finish_time(state.time + jnp.asarray(dt, jnp.float32), dt, previous, position, distance)
    finish = jnp.where(crossed, result, state.finish_time)
    ticks = state.ticks + active.astype(jnp.int32)
    timeout = active & ~crossed & (ticks >= max_ticks)
    boundaries = phase_boundaries(distance)
    new_phase = jnp.minimum(jnp.sum(state.position[...,None] >= boundaries, axis=-1), 2).astype(jnp.int32)
    start_dash = state.start_dash & ~(next_speed >= dash_limit)
    next_time = jnp.where(active, state.time + jnp.asarray(dt, jnp.float32), state.time)
    return RestrictedState(next_time, position, jnp.where(moving, next_speed, state.speed), hp, delay, state.finished | crossed, state.timed_out | timeout, finish, ticks, jnp.where(moving,new_phase,state.phase), jnp.where(moving,start_dash,state.start_dash))


def run(params, state, *, max_ticks, **inputs):
    """Fixed-length prevalidated kernel; terminal scan rows are frozen padding."""
    def body(carry, _):
        next_state = step(params, carry, max_ticks=max_ticks, **inputs)
        return next_state, next_state
    return jax.lax.scan(body, state, None, length=max_ticks)


def run_checked(params, state, *, config, speed_stat, power, guts, strategy, aptitude, ground):
    """Validate host inputs, then execute the fixed-shape JIT-compatible kernel."""
    require_x64()
    if not isinstance(config, RestrictedConfig):
        raise TypeError("config must be a frozen RestrictedConfig")
    _validate_state_and_inputs(state, speed_stat=speed_stat, power=power, guts=guts,
                               strategy=strategy, aptitude=aptitude, ground=ground)
    return run(params, state, max_ticks=config.max_ticks,
               speed_stat=jnp.asarray(speed_stat, jnp.float32), power=jnp.asarray(power, jnp.float32),
               guts=jnp.asarray(guts, jnp.float32), strategy=jnp.asarray(strategy, jnp.int32),
               aptitude=jnp.asarray(aptitude, jnp.float32), ground=jnp.asarray(ground, jnp.float32),
               distance=config.distance, dt=config.dt)
