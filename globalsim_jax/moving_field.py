"""Restricted constant-width, supplied-target field; no target policy or pacing.

Callers enable JAX x64 explicitly. Checked entry points validate finite real f32
storage and exclude nonzero f32 subnormals/side-room differences; f64 blocking
state remains f64. Low-level step/run require those prevalidated conditions.
The whole crossing pass completes, then every state leaf freezes as padding.
"""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from .precision import require_x64
from .field import FieldState, BlockParameters, prepare_blocking
from .lane import resolve_overlaps
from .lane_move import MoveState, MoveContext, MoveParameters, MAX_MODIFIERS, step as move_step
from .lane_queries import side_room, BlockParameters as SideParameters, _real_scalar, _reject_nonzero_subnormals
from .restricted import RestrictedConfig, MAX_RUNNERS, step as restricted_step
from .physics import advance_start_delay

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MovingFieldState:
    field: FieldState
    target: jax.Array
    lane_speed: jax.Array
    continue_time: jax.Array
    continue_direction: jax.Array
    force_checks: jax.Array
    moved: jax.Array  # Per-pass wrapper telemetry; lane-speed receiver state is excluded.

def initialize(field, target):
    """Attach supplied targets; custom momentum/flags may use the explicit dataclass."""
    require_x64()
    if not isinstance(field, FieldState):
        raise TypeError("field must be FieldState")
    raw = np.asarray(target)
    if (np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number)
            or raw.shape != field.lane.shape or not np.isfinite(raw).all()):
        raise ValueError("target must be a finite real [batch, runner] row")
    with np.errstate(over="ignore", under="ignore"):
        target = np.asarray(raw, np.float32)
    if not np.isfinite(target).all():
        raise ValueError("target must remain finite after float32 storage")
    _reject_nonzero_subnormals(target)
    return MovingFieldState(
        field, jnp.asarray(target), jnp.zeros_like(field.lane), jnp.zeros_like(field.lane),
        jnp.zeros_like(field.lane, jnp.int32), jnp.zeros_like(field.lane, bool),
        jnp.zeros_like(field.lane, bool),
    )


def step(params, state, *, speed_stat, power, guts, strategy, aptitude, ground, distance, dt, max_ticks, max_lane, first_move_point, move_parameters, block_parameters=BlockParameters()):
    """Prevalidated source schedule: prepass, indexed lateral/longitudinal, postpass overlap."""
    old = state.field
    n = old.lane.shape[1]
    positions, entry_lane = old.runners.position, old.lane
    prepared, front, gap, _, rank = prepare_blocking(params, old, guts=guts, distance=distance, dt=dt, parameters=block_parameters)
    _, gate, _ = advance_start_delay(old.runners.delay, dt)
    roster = jnp.arange(n, dtype=jnp.int32)
    def cached_side(owner):
        peers = roster[:-1] + (roster[:-1] >= owner)
        pp, pl = jnp.take(positions, peers, axis=1), jnp.take(entry_lane, peers, axis=1)
        pp = jnp.pad(pp, ((0, 0), (0, MAX_RUNNERS - (n - 1))))
        pl = jnp.pad(pl, ((0, 0), (0, MAX_RUNNERS - (n - 1))))
        return jax.vmap(lambda op, ol, x, y: side_room(op, ol, max_lane, x, y, n - 1, SideParameters(block_parameters.side_distance, block_parameters.side_lane)))(positions[:, owner], entry_lane[:, owner], pp, pl)
    sides = jax.vmap(cached_side)(roster)
    def one(index, carry):
        runners, lane, target, speed, cont, direction, moved = carry
        side = jax.tree_util.tree_map(lambda x: x[index], sides)
        blocker = front[:, index]
        live_speed = jnp.take_along_axis(runners.speed, jnp.maximum(blocker, 0)[:, None], axis=1)[:, 0]
        cap = jnp.where(blocker >= 0, params.blocking_speed_cap(prepared.minimum[:, index], live_speed, gap[:, index]), jnp.float32(30.0))
        ms = MoveState(lane[:, index], target[:, index], speed[:, index], cont[:, index], direction[:, index])
        ctx = MoveContext(jnp.zeros(lane.shape[0], bool), positions[:, index], power[:, index], old.runners.phase[:, index], rank[:, index], jnp.full(lane.shape[0], first_move_point, jnp.float32), jnp.full(lane.shape[0], max_lane, jnp.float32), side.can_in, side.can_out, jnp.zeros((lane.shape[0], MAX_MODIFIERS), jnp.float32), jnp.zeros(lane.shape[0], jnp.int32))
        out, did = jax.vmap(move_step)(ms, ctx, jnp.full(lane.shape[0], dt, jnp.float32), jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (lane.shape[0],)), move_parameters))
        active = (~old.truncated) & gate[:, index] & ~old.runners.finished[:, index] & ~old.runners.timed_out[:, index] & (positions[:, index] < distance)
        lane = lane.at[:, index].set(jnp.where(active, out.lane, lane[:, index])); target = target.at[:, index].set(jnp.where(active, out.target, target[:, index])); speed = speed.at[:, index].set(jnp.where(active, out.speed, speed[:, index])); cont = cont.at[:, index].set(jnp.where(active, out.continue_time, cont[:, index])); direction = direction.at[:, index].set(jnp.where(active, out.continue_direction, direction[:, index])); moved = moved.at[:, index].set(active & did)
        update = (roster[None, :] == index) & ~old.truncated[:, None]
        runners = restricted_step(params, runners, speed_stat=speed_stat, power=power, guts=guts, strategy=strategy, aptitude=aptitude, ground=ground, distance=distance, dt=dt, max_ticks=max_ticks, minimum_override=prepared.minimum, cap_override=jnp.broadcast_to(cap[:, None], runners.speed.shape), update_mask=update)
        return runners, lane, target, speed, cont, direction, moved
    runners, lane, target, speed, cont, direction, moved = jax.lax.fori_loop(0, n, one, (prepared.runners, prepared.lane, state.target, state.lane_speed, state.continue_time, state.continue_direction, jnp.zeros_like(state.moved)))
    def post(pos, ln, force):
        pad = MAX_RUNNERS - n
        pp = jnp.pad(pos, (0, pad)); pl = jnp.pad(ln, (0, pad)); pf = jnp.pad(force, (0, pad))
        (out_lane, out_force), _ = resolve_overlaps(pp, pl, pf, n)
        return out_lane[:n], out_force[:n]
    lane, force = jax.vmap(post)(runners.position, lane, state.force_checks)
    truncated = old.truncated | jnp.any(runners.finished, axis=1) | jnp.all(runners.timed_out, axis=1)
    field = FieldState(runners, lane, prepared.minimum, prepared.blocked_front_time, prepared.blocked_side_time, prepared.blocked_all_time, truncated)
    result = MovingFieldState(field, target, speed, cont, direction, force, moved)
    def freeze(new, prior):
        mask = old.truncated.reshape((old.truncated.shape[0],) + (1,) * (new.ndim - 1))
        return jnp.where(mask, prior, new)
    return jax.tree_util.tree_map(freeze, result, state)

def run(params, state, *, max_ticks, **inputs):
    """Prevalidated fixed-length scan; terminal rows are whole-state padding."""
    def body(carry, _):
        next_state = step(params, carry, max_ticks=max_ticks, **inputs)
        return next_state, next_state
    return jax.lax.scan(body, state, None, length=max_ticks)

def run_checked(params, state, *, config, speed_stat, power, guts, strategy, aptitude, ground, max_lane, first_move_point, move_parameters):
    """Host validation for supplied-target moving fields; kernels are prevalidated."""
    require_x64()
    from .restricted import _validate_state_and_inputs
    if not isinstance(config, RestrictedConfig) or not isinstance(move_parameters, MoveParameters):
        raise TypeError('config and move_parameters have required types')
    _validate_state_and_inputs(state.field.runners, speed_stat=speed_stat, power=power, guts=guts, strategy=strategy, aptitude=aptitude, ground=ground)
    shape = state.field.runners.position.shape
    def f32field(value, name):
        a=np.asarray(value)
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all(): raise ValueError(f'{name} must be finite float32 [batch, runner]')
        return a
    for name in ('lane','minimum') : f32field(getattr(state.field,name), 'field.'+name)
    for name in ('target','lane_speed','continue_time'): f32field(getattr(state,name), name)
    if (np.any(np.asarray(state.field.minimum) < 0)
            or np.any(np.asarray(state.lane_speed) < 0)
            or np.any(np.asarray(state.continue_time) < 0)):
        raise ValueError('minimum, lane_speed and continue_time must be nonnegative')
    for name in ('continue_direction',):
        a=np.asarray(getattr(state,name))
        if a.shape!=shape or not np.issubdtype(a.dtype,np.integer) or np.any((a<0)|(a>1)): raise ValueError('continue_direction must be 0/1 [batch, runner]')
    for name in ('force_checks','moved'):
        a=np.asarray(getattr(state,name))
        if a.shape!=shape or a.dtype!=np.bool_: raise ValueError(f'{name} must be bool [batch, runner]')
    for name in ('blocked_front_time','blocked_side_time','blocked_all_time'):
        a=np.asarray(getattr(state.field,name))
        if a.shape!=shape or a.dtype!=np.float64 or not np.isfinite(a).all() or np.any(a<0): raise ValueError(f'field.{name} invalid')
    tr=np.asarray(state.field.truncated)
    if tr.shape!=(shape[0],) or tr.dtype!=np.bool_: raise ValueError('truncated must be bool [batch]')
    terminal=np.any(np.asarray(state.field.runners.finished)|(np.asarray(state.field.runners.position)>=config.distance),axis=1)|np.all(np.asarray(state.field.runners.timed_out),axis=1)
    if np.any(terminal & ~tr): raise ValueError('terminal batch must be truncated')
    _real_scalar(max_lane, 'max_lane')
    _real_scalar(first_move_point, 'first_move_point')
    with np.errstate(over='ignore', under='ignore'):
        scalars = np.asarray([max_lane, first_move_point], np.float32)
    if not np.isfinite(scalars).all() or scalars[0] <= 0:
        raise ValueError('max_lane/first_move_point invalid after float32 storage')
    for value in move_parameters.__dict__.values():
        a=np.asarray(value)
        if a.shape!=() or a.dtype!=np.float32 or not np.isfinite(a): raise ValueError('MoveParameters must be finite float32 scalars')
    # The query profile excludes f32 subnormals, including stored side deltas.
    # Keep the f64 blocking timers in their source dtype; never narrow them here.
    position = np.asarray(state.field.runners.position)
    lane = np.asarray(state.field.lane)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        derived = (position[:, :, None] - position[:, None, :],
                   lane[:, :, None] - lane[:, None, :], scalars[0] - lane)
    if any(not np.isfinite(value).all() for value in derived):
        raise ValueError('side-room differences must remain finite in float32')
    _reject_nonzero_subnormals(
        scalars, position, lane, np.asarray(state.field.minimum),
        np.asarray(state.target), np.asarray(state.lane_speed),
        np.asarray(state.continue_time), np.asarray(state.field.runners.speed),
        np.asarray(state.field.runners.hp), np.asarray(state.field.runners.delay),
        np.asarray(state.field.runners.time), np.asarray(speed_stat), np.asarray(power),
        np.asarray(guts), np.asarray(aptitude), np.asarray(ground), np.float32(config.dt),
        *derived, *move_parameters.__dict__.values(),
    )
    return run(params,state,max_ticks=config.max_ticks,speed_stat=jnp.asarray(speed_stat,jnp.float32),power=jnp.asarray(power,jnp.float32),guts=jnp.asarray(guts,jnp.float32),strategy=jnp.asarray(strategy,jnp.int32),aptitude=jnp.asarray(aptitude,jnp.float32),ground=jnp.asarray(ground,jnp.float32),distance=config.distance,dt=config.dt,max_lane=float(scalars[0]),first_move_point=float(scalars[1]),move_parameters=move_parameters)
