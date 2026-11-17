"""M3a fixed-lane field queries and ordered blocking pass.

This deliberately excludes lane selection, skills, pacing, and post-goal retail
updates.  Integrated runs truncate before the first finisher.
"""
from dataclasses import dataclass
import math
import numpy as np
import jax
import jax.numpy as jnp
from .precision import require_x64
from .restricted import MAX_RUNNERS, RestrictedState, RestrictedConfig, _validate_state_and_inputs
from .restricted import step as restricted_step


@dataclass(frozen=True)
class BlockParameters:
    front_distance: float = 2.0
    front_lane: float = 0.0416666716337204
    side_distance: float = 1.0499999523162842
    side_lane: float = 0.1111110970377922

    def __post_init__(self):
        for name, value in self.__dict__.items():
            rounded = float(np.float32(value))
            object.__setattr__(self, name, rounded)
        if any(not math.isfinite(v) or v <= 0 for v in self.__dict__.values()):
            raise ValueError("block parameters must be finite and positive")


def front_lane_limit(gap, parameters):
    """Rust `front_lane_limit`: f32 narrowing then f64 interpolation store."""
    gap = jnp.asarray(gap, jnp.float32)
    narrowing = gap / jnp.float32(parameters.front_distance) * jnp.float32(0.6)
    front_lane = jnp.asarray(parameters.front_lane, jnp.float32)
    return ((jnp.float64(1.0) - narrowing.astype(jnp.float64)) * front_lane.astype(jnp.float64)).astype(jnp.float32)


def ranks(position):
    """Descending position rank; exact ties are resolved by lower runner index."""
    position = jnp.asarray(position, jnp.float32)
    n = position.shape[-1]
    index = jnp.arange(n, dtype=jnp.int32)
    ahead = (position[..., None, :] > position[..., :, None])
    tie_lower = (position[..., None, :] == position[..., :, None]) & (index[None, :] < index[:, None])
    return (jnp.sum(ahead | tie_lower, axis=-1) + 1).astype(jnp.int32)


def _checked_query_arrays(position, lane, finished):
    require_x64()
    arrays = [np.asarray(value) for value in (position, lane, finished)]
    if any(value.ndim != 2 or value.shape != arrays[0].shape for value in arrays) or not arrays[0].shape[0] or not arrays[0].shape[1] or arrays[0].shape[1] > MAX_RUNNERS:
        raise ValueError("field query inputs require nonempty equal [batch, runner] arrays within capacity")
    if arrays[0].dtype != np.float32 or arrays[1].dtype != np.float32 or arrays[2].dtype != np.bool_ or not np.all(np.isfinite(arrays[0])) or not np.all(np.isfinite(arrays[1])):
        raise ValueError("field query requires finite float32 position/lane and bool finished arrays")


def ranks_checked(position):
    position = np.asarray(position)
    _checked_query_arrays(position, np.zeros_like(position, np.float32), np.zeros_like(position, bool))
    return ranks(position)


def blocking_snapshot(position, lane, finished, parameters=BlockParameters()):
    """Post-skill fixed-lane snapshot: front index/gap and side bool per owner."""
    position, lane = jnp.asarray(position, jnp.float32), jnp.asarray(lane, jnp.float32)
    finished = jnp.asarray(finished, bool)
    n = position.shape[-1]
    owner = jnp.arange(n, dtype=jnp.int32)
    gap = position[..., None, :] - position[..., :, None]
    lane_gap = lane[..., None, :] - lane[..., :, None]
    other = owner[None, :] != owner[:, None]
    eligible = other & (gap > 0) & (gap < jnp.float32(parameters.front_distance)) & (jnp.abs(lane_gap) <= front_lane_limit(gap, parameters))
    # Lexicographic minimum gap then runner index, exactly preserving equal-gap ties.
    candidate_gap = jnp.where(eligible, gap, jnp.inf)
    nearest = jnp.min(candidate_gap, axis=-1)
    equal = eligible & (gap == nearest[..., None])
    front = jnp.min(jnp.where(equal, owner, n), axis=-1).astype(jnp.int32)
    has_front = jnp.isfinite(nearest)
    front = jnp.where(has_front, front, -1)
    side = other & ~finished[..., None, :] & (jnp.abs(gap) < jnp.float32(parameters.side_distance)) & (jnp.abs(lane_gap) < jnp.float32(parameters.side_lane)) & (lane_gap != 0)
    return front, jnp.where(has_front, nearest, jnp.float32(jnp.nan)), jnp.any(side, axis=-1)


def blocking_snapshot_checked(position, lane, finished, parameters=BlockParameters()):
    _checked_query_arrays(position, lane, finished)
    return blocking_snapshot(position, lane, finished, parameters)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FieldState:
    runners: RestrictedState
    lane: jax.Array
    minimum: jax.Array
    blocked_front_time: jax.Array
    blocked_side_time: jax.Array
    blocked_all_time: jax.Array
    truncated: jax.Array


def initialize_field(runners, lane, *, prior_minimum=None):
    """Host checked fixed-lane field state; no lane selection is performed."""
    require_x64()
    lane = np.asarray(lane, np.float32)
    if lane.shape != runners.position.shape or lane.ndim != 2 or not lane.shape[0] or not lane.shape[1] or lane.shape[1] > MAX_RUNNERS or not np.all(np.isfinite(lane)):
        raise ValueError("lane must be finite nonempty [batch, runner] within capacity")
    fresh = (np.all(np.asarray(runners.time) == 0) and np.all(np.asarray(runners.ticks) == 0)
             and np.all(np.asarray(runners.position) == 0) and np.all(np.asarray(runners.speed) == 0)
             and np.all(np.asarray(runners.phase) == 0) and np.all(np.asarray(runners.start_dash))
             and not np.any(np.asarray(runners.finished | runners.timed_out))
             and np.all(np.isnan(np.asarray(runners.finish_time))))
    if prior_minimum is None and not fresh:
        raise ValueError("nonfresh field attachment requires explicit prior_minimum")
    minimum = jnp.full(lane.shape, 3.0, jnp.float32) if prior_minimum is None else jnp.asarray(prior_minimum, jnp.float32)
    if minimum.shape != lane.shape or not np.all(np.isfinite(np.asarray(minimum))) or np.any(np.asarray(minimum) < 0):
        raise ValueError("prior_minimum must be finite nonnegative float32 [batch, runner]")
    zeros64 = jnp.zeros(lane.shape, jnp.float64)
    truncated = jnp.all(runners.timed_out, axis=1)
    return FieldState(runners, jnp.asarray(lane), minimum, zeros64, zeros64, zeros64, truncated)


def prepare_blocking(params, state, *, guts, distance, dt, parameters=BlockParameters()):
    """M3a post-skill snapshot and ordered stored-minimum pass.

    This is deliberately separated from M2 movement while M3b lane/pace state
    is absent.  It pins the source observation boundary for the later field
    movement port: front/side comes from one pre-movement snapshot, but stored
    minimum inheritance is read in runner order.
    """
    front, gap, side = blocking_snapshot(state.runners.position, state.lane, state.runners.position >= distance, parameters)
    dt64 = jnp.asarray(dt, jnp.float32).astype(jnp.float64)
    front_time = jnp.where(front >= 0, state.blocked_front_time + dt64, 0.0)
    side_time = jnp.where(side, state.blocked_side_time + dt64, 0.0)
    all_time = jnp.where((front >= 0) & side, state.blocked_all_time + dt64, 0.0)
    base = jax.vmap(jax.vmap(params.minimum_speed))(jnp.asarray(guts, jnp.float32), jnp.full_like(state.runners.position, distance))
    initial = jnp.where(state.runners.start_dash, 3.0, base)

    def one_runner(index, minimum):
        blocker = front[:, index]
        inherited = jnp.take_along_axis(minimum, jnp.maximum(blocker, 0)[:, None], axis=1)[:, 0]
        value = jnp.where((blocker >= 0) & ~state.runners.start_dash[:, index] & ~state.runners.finished[:, index],
                          jnp.minimum(initial[:, index], inherited), initial[:, index])
        return minimum.at[:, index].set(value)

    minimum = jax.lax.fori_loop(0, state.lane.shape[1], one_runner, state.minimum)
    return FieldState(state.runners, state.lane, minimum, front_time, side_time, all_time, state.truncated), front, gap, side, ranks(state.runners.position)


def step(params, state, *, speed_stat, power, guts, strategy, aptitude, ground, distance, dt, max_ticks,
         parameters=BlockParameters()):
    """One M3a ordered fixed-lane movement pass.

    The blocker identity/gap is frozen before the pass. Every runner then sees
    prior indices' live speed through the cap while retaining that snapshot ID.
    A batch stops only on the *next* call after any lane finishes.
    """
    prepared, front, gap, side, rank = prepare_blocking(params, state, guts=guts, distance=distance, dt=dt, parameters=parameters)
    # The bounded first-finish rule freezes the complete FieldState, including
    # snapshot timers and stored minima, before any later field pass.
    if_truncated = state.truncated[:, None]
    prepared = FieldState(
        state.runners, state.lane, jnp.where(if_truncated, state.minimum, prepared.minimum),
        jnp.where(if_truncated, state.blocked_front_time, prepared.blocked_front_time),
        jnp.where(if_truncated, state.blocked_side_time, prepared.blocked_side_time),
        jnp.where(if_truncated, state.blocked_all_time, prepared.blocked_all_time), state.truncated,
    )
    n = state.lane.shape[1]

    def move(index, carry):
        runners = carry.runners
        blocker = front[:, index]
        safe = jnp.maximum(blocker, 0)
        speed = jnp.take_along_axis(runners.speed, safe[:, None], axis=1)[:, 0]
        inherited_min = prepared.minimum[:, index]
        cap = params.blocking_speed_cap(inherited_min, speed, gap[:, index])
        cap = jnp.where(blocker >= 0, cap, jnp.float32(30.0))
        mask = (jnp.arange(n)[None, :] == index) & ~carry.truncated[:, None]
        next_runners = restricted_step(
            params, runners, speed_stat=speed_stat, power=power, guts=guts, strategy=strategy,
            aptitude=aptitude, ground=ground, distance=distance, dt=dt, max_ticks=max_ticks,
            minimum_override=prepared.minimum, cap_override=jnp.broadcast_to(cap[:, None], runners.speed.shape),
            update_mask=mask,
        )
        return FieldState(next_runners, carry.lane, prepared.minimum, carry.blocked_front_time,
                          carry.blocked_side_time, carry.blocked_all_time, carry.truncated)

    moved = jax.lax.fori_loop(0, n, move, prepared)
    # Bounded policy: permit this crossing pass, then freeze the whole batch.
    return FieldState(moved.runners, moved.lane, moved.minimum, moved.blocked_front_time,
                      moved.blocked_side_time, moved.blocked_all_time,
                      moved.truncated | jnp.any(moved.runners.finished, axis=1) | jnp.all(moved.runners.timed_out, axis=1)), front, gap, side, rank


def run(params, state, *, max_ticks, **inputs):
    """Fixed-length M3a scan; rows after a batch's first finish are frozen."""
    def body(carry, _):
        value = step(params, carry, max_ticks=max_ticks, **inputs)[0]
        return value, value
    return jax.lax.scan(body, state, None, length=max_ticks)


def run_checked(params, state, *, config, speed_stat, power, guts, strategy, aptitude, ground,
                parameters=BlockParameters()):
    """Host-validated M3a scan; low-level ``step``/``run`` are prevalidated."""
    require_x64()
    if not isinstance(config, RestrictedConfig):
        raise TypeError("config must be RestrictedConfig")
    _validate_state_and_inputs(state.runners, speed_stat=speed_stat, power=power, guts=guts,
                               strategy=strategy, aptitude=aptitude, ground=ground)
    shape = state.runners.position.shape
    lane = np.asarray(state.lane)
    minimum = np.asarray(state.minimum)
    if lane.shape != shape or lane.dtype != np.float32 or not np.all(np.isfinite(lane)):
        raise ValueError("field lane must be finite float32 [batch, runner]")
    if minimum.shape != shape or minimum.dtype != np.float32 or not np.all(np.isfinite(minimum)) or np.any(minimum < 0):
        raise ValueError("field minimum must be finite nonnegative float32 [batch, runner]")
    for name in ("blocked_front_time", "blocked_side_time", "blocked_all_time"):
        value = np.asarray(getattr(state, name))
        if value.shape != shape or value.dtype != np.float64 or not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError(f"{name} must be finite nonnegative float64 [batch, runner]")
    truncated = np.asarray(state.truncated)
    if truncated.shape != (shape[0],) or truncated.dtype != np.bool_:
        raise ValueError("truncated must be bool [batch]")
    timed_out = np.all(np.asarray(state.runners.timed_out), axis=1)
    if np.any(timed_out & ~truncated):
        raise ValueError("all-timed-out batch must be truncated")
    live_post_goal = np.any((np.asarray(state.runners.position) >= config.distance) | np.asarray(state.runners.finished), axis=1)
    if np.any(live_post_goal & ~truncated):
        raise ValueError("M3a does not attach live post-goal runners; mark batch truncated")
    return run(params, state, max_ticks=config.max_ticks, speed_stat=jnp.asarray(speed_stat, jnp.float32),
               power=jnp.asarray(power, jnp.float32), guts=jnp.asarray(guts, jnp.float32),
               strategy=jnp.asarray(strategy, jnp.int32), aptitude=jnp.asarray(aptitude, jnp.float32),
               ground=jnp.asarray(ground, jnp.float32), distance=config.distance, dt=config.dt,
               parameters=parameters)
