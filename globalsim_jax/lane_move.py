"""M3b standalone deterministic port of ``retail_lane::MoveState::step``.

This module does not select targets, inspect a field, or resolve overlaps.  A
caller supplies a target and the side-permission snapshot already sampled by
the field.  ``step`` is a prevalidated JIT kernel; ``step_checked`` is the host
boundary for one scalar move state.
"""

from dataclasses import dataclass
import json
from importlib.resources import files

import jax
import jax.numpy as jnp
import numpy as np

from .physics import bounded_stat
from .precision import require_x64


MAX_MODIFIERS = 16
f32 = lambda value: jnp.asarray(value, jnp.float32)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MoveState:
    lane: jax.Array
    target: jax.Array
    speed: jax.Array
    continue_time: jax.Array
    continue_direction: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MoveContext:
    after_goal: jax.Array
    position: jax.Array
    power: jax.Array
    phase: jax.Array
    rank: jax.Array
    first_move_point: jax.Array
    max_lane: jax.Array
    can_in: jax.Array
    can_out: jax.Array
    modifiers: jax.Array
    modifier_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MoveParameters:
    power_coefficient: jax.Array
    speed_add: jax.Array
    base_speed: jax.Array
    outer_coefficient: jax.Array
    acceleration: jax.Array

    @classmethod
    def default(cls):
        require_x64()
        root = json.loads(files("globalsim_jax").joinpath("global_parameters.json").read_text())
        values = root["parameters"]
        return cls(
            f32(values["laneMoveSpeedPowCoef"]),
            f32(values["laneMoveSpeedAdd"]),
            f32(values["laneMoveSpeedBase"]),
            f32(values["laneMoveSpeedOutCoef"]),
            f32(values["laneMoveAccelMoveSpeedBaseCoef"]),
        )


def _approximately(left, right):
    """Rust f32 comparison, widened only for subnormal-safe final comparison."""
    left, right = f32(left), f32(right)
    minimum_subnormal = jnp.asarray(2.0**-149, jnp.float64)
    minimum_normal = jnp.asarray(2.0**-126, jnp.float64)

    def widen(value):
        bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
        exponent = bits & jnp.uint32(0x7F800000)
        mantissa = bits & jnp.uint32(0x007FFFFF)
        sign = jnp.where((bits & jnp.uint32(0x80000000)) != 0, -1.0, 1.0)
        subnormal = sign * mantissa.astype(jnp.float64) * minimum_subnormal
        return jnp.where(exponent == 0, subnormal, value.astype(jnp.float64))

    def rounded_f32(value):
        absolute = jnp.abs(value)
        subnormal = jnp.rint(value / minimum_subnormal) * minimum_subnormal
        return jnp.where(absolute < minimum_normal, subnormal, f32(value).astype(jnp.float64))

    left64, right64 = widen(left), widen(right)
    difference = rounded_f32(jnp.abs(left64 - right64))
    magnitude = jnp.maximum(jnp.abs(left64), jnp.abs(right64))
    scale = rounded_f32(rounded_f32(jnp.asarray(1e-6, jnp.float64)) * magnitude)
    threshold = jnp.maximum(scale, 8.0 * minimum_subnormal)
    return difference < threshold


def _apply_modifiers(base, modifiers, count):
    """Source ordered f32 fold; padded values are never read as modifiers."""
    modifiers = f32(modifiers)
    count = jnp.asarray(count, jnp.int32)
    def add(index, value):
        return jnp.where(index < count, value + modifiers[index], value)
    return jax.lax.fori_loop(0, MAX_MODIFIERS, add, f32(base))


def step(state: MoveState, context: MoveContext, dt, parameters: MoveParameters):
    """Prevalidated source-order lateral move; returns ``(state, moved)``."""
    dt = f32(dt)
    target = jnp.clip(f32(state.target), f32(0.0), f32(context.max_lane))
    initial = MoveState(f32(state.lane), target, f32(state.speed), f32(state.continue_time), jnp.asarray(state.continue_direction, jnp.int32))

    def at_target(_: None):
        return MoveState(initial.lane, initial.target, initial.speed, f32(0.0), initial.continue_direction), jnp.asarray(False)

    def moving(_: None):
        previous_direction = initial.continue_direction
        direction = (initial.target >= initial.lane).astype(jnp.int32)

        def after_goal(_: None):
            return jnp.where(initial.target < initial.lane, -dt, dt), initial.speed, jnp.asarray(False)

        def ordinary(_: None):
            rate = ((bounded_stat(context.power).astype(jnp.float64) * parameters.power_coefficient.astype(jnp.float64) + parameters.speed_add.astype(jnp.float64)) * parameters.base_speed.astype(jnp.float64)).astype(jnp.float32)
            phase_factor = f32(1.0) + (context.rank.astype(jnp.float64) - 1.0) * f32(0.01).astype(jnp.float64)
            outer_factor = f32(1.0) + (initial.lane / context.max_lane).astype(jnp.float64) * parameters.outer_coefficient.astype(jnp.float64)
            factor = jnp.where(context.phase >= 2, phase_factor, jnp.where(context.position <= context.first_move_point, outer_factor, f32(1.0)))
            rate = (rate.astype(jnp.float64) * factor).astype(jnp.float32)
            acceleration = parameters.acceleration.astype(jnp.float64) * parameters.base_speed.astype(jnp.float64) * dt.astype(jnp.float64)
            stored_speed = jnp.where(
                initial.speed < rate,
                jnp.minimum((initial.speed.astype(jnp.float64) + acceleration).astype(jnp.float32), rate),
                jnp.maximum((initial.speed.astype(jnp.float64) - acceleration).astype(jnp.float32), rate),
            )
            stored_speed = jnp.maximum(stored_speed, f32(0.0))
            adjusted = jnp.clip(_apply_modifiers(stored_speed, context.modifiers, context.modifier_count), f32(0.0), f32(0.6))
            signed = jnp.where(initial.target < initial.lane, (-(adjusted.astype(jnp.float64)) * (f32(1.0) + initial.lane.astype(jnp.float64))).astype(jnp.float32), adjusted)
            return signed, stored_speed, _approximately(adjusted, f32(0.0))

        speed, stored_speed, zero_speed = jax.lax.cond(context.after_goal, after_goal, ordinary, operand=None)
        next_lane = (initial.lane.astype(jnp.float64) + dt.astype(jnp.float64) * speed.astype(jnp.float64)).astype(jnp.float32)
        next_lane = jnp.where(initial.target < initial.lane, jnp.maximum(next_lane, initial.target), jnp.minimum(next_lane, initial.target))
        next_lane = jnp.clip(next_lane, f32(0.0), context.max_lane)
        blocked = ((next_lane < initial.lane) & ~context.can_in) | ((next_lane > initial.lane) & ~context.can_out)
        accepted = ~zero_speed & ~blocked
        moved = accepted & (next_lane != initial.lane)
        updated = MoveState(
            jnp.where(accepted, next_lane, initial.lane),
            initial.target,
            jnp.where(context.after_goal, initial.speed, stored_speed),
            jnp.where(accepted & (previous_direction == direction), initial.continue_time + dt, initial.continue_time),
            direction,
        )
        reached = _approximately(next_lane, initial.target)
        updated = MoveState(updated.lane, updated.target, jnp.where(accepted & reached, f32(0.0), updated.speed), updated.continue_time, updated.continue_direction)
        return updated, moved

    return jax.lax.cond(_approximately(target, initial.lane), at_target, moving, operand=None)


def step_checked(state: MoveState, context: MoveContext, dt, parameters: MoveParameters):
    """Validate one scalar supplied-target lateral move without global config mutation."""
    require_x64()
    if not isinstance(state, MoveState) or not isinstance(context, MoveContext) or not isinstance(parameters, MoveParameters):
        raise TypeError("state, context, and parameters must use MoveState/MoveContext/MoveParameters")
    scalar_f32 = (state.lane, state.target, state.speed, state.continue_time, context.position, context.power, context.first_move_point, context.max_lane, dt, *parameters.__dict__.values())
    for value in scalar_f32:
        array = np.asarray(value)
        if array.shape != () or not np.isfinite(array) or array.dtype != np.dtype(np.float32):
            raise ValueError("scalar state, context, dt, and parameters must be finite float32 values")
    if float(np.asarray(context.max_lane)) <= 0 or float(np.asarray(dt)) < 0:
        raise ValueError("max_lane must be positive and dt must be nonnegative")
    for value, name, minimum in ((context.phase, "phase", 0), (context.rank, "rank", 1), (state.continue_direction, "continue_direction", 0), (context.modifier_count, "modifier_count", 0)):
        array = np.asarray(value)
        if array.shape != () or not np.issubdtype(array.dtype, np.integer) or int(array) < minimum or int(array) > np.iinfo(np.int32).max:
            raise ValueError(f"{name} is outside the supported integer range")
    if int(np.asarray(state.continue_direction)) > 1:
        raise ValueError("continue_direction must be a source direction flag (0 or 1)")
    modifiers = np.asarray(context.modifiers)
    if modifiers.shape != (MAX_MODIFIERS,) or modifiers.dtype != np.dtype(np.float32) or not np.isfinite(modifiers).all():
        raise ValueError("modifiers must be a finite float32 padded row")
    if int(np.asarray(context.modifier_count)) > MAX_MODIFIERS:
        raise ValueError("modifier_count exceeds fixed capacity")
    for value in (context.after_goal, context.can_in, context.can_out):
        if np.asarray(value).shape != () or np.asarray(value).dtype != np.dtype(np.bool_):
            raise ValueError("permission and after_goal values must be scalar booleans")
    # Canonicalize every accepted leaf before it reaches the prevalidated JIT
    # kernel; host scalars such as Python bool/int are deliberately supported.
    normalized_state = MoveState(
        f32(state.lane), f32(state.target), f32(state.speed), f32(state.continue_time),
        jnp.asarray(state.continue_direction, jnp.int32),
    )
    normalized_context = MoveContext(
        jnp.asarray(context.after_goal, jnp.bool_), f32(context.position), f32(context.power),
        jnp.asarray(context.phase, jnp.int32), jnp.asarray(context.rank, jnp.int32),
        f32(context.first_move_point), f32(context.max_lane), jnp.asarray(context.can_in, jnp.bool_),
        jnp.asarray(context.can_out, jnp.bool_), f32(context.modifiers),
        jnp.asarray(context.modifier_count, jnp.int32),
    )
    normalized_parameters = MoveParameters(*[f32(value) for value in parameters.__dict__.values()])
    return step(normalized_state, normalized_context, f32(dt), normalized_parameters)
