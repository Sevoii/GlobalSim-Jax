"""Standalone M3b side-room, visibility, and overtaking query kernels.

Checked wrappers accept real finite values after source f32 storage and reject
overflow plus nonzero subnormal operands; side-room also rejects subnormal
stored deltas/spaces. Kernels require callers to prevalidate those conditions,
shape-32 peer rows/count 0..32, and JAX x64 configuration.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from .precision import require_x64


MAX_PEERS = 32
f32 = lambda value: jnp.asarray(value, jnp.float32)


def _reject_nonzero_subnormals(*arrays):
    """CPU XLA flushes these in comparisons; checked APIs exclude that domain."""
    tiny = np.finfo(np.float32).tiny
    for array in arrays:
        value = np.asarray(array, np.float32)
        if np.any((value != 0.0) & (np.abs(value) < tiny)):
            raise ValueError("nonzero float32 subnormal inputs are outside the checked query profile")


def _real_scalar(value, name):
    array = np.asarray(value)
    if array.shape != () or np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite real scalar")
    return array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BlockParameters:
    side_distance: jax.Array
    side_lane: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SideRoom:
    can_in: jax.Array
    can_out: jax.Array
    in_space: jax.Array
    out_space: jax.Array
    inner_blocker_lane: jax.Array
    inner_blocker_index: jax.Array


def side_room(owner_position, owner_lane, max_lane, peer_position, peer_lane, count, parameters: BlockParameters):
    """Prevalidated <=32 kernel: finite, non-subnormal f32 inputs/deltas only."""
    owner_position, owner_lane, max_lane = f32(owner_position), f32(owner_lane), f32(max_lane)
    peer_position, peer_lane = f32(peer_position), f32(peer_lane)
    count = jnp.asarray(count, jnp.int32)
    state = SideRoom(owner_lane > f32(0.0001), owner_lane < max_lane, owner_lane, max_lane - owner_lane, jnp.asarray(jnp.nan, jnp.float32), jnp.asarray(-1, jnp.int32))
    def body(index, result):
        active = index < count
        nearby = active & (jnp.abs(peer_position[index] - owner_position) < parameters.side_distance)
        gap = peer_lane[index] - owner_lane
        inner = nearby & (gap < f32(0.0))
        outer = nearby & (gap > f32(0.0))
        inner_block = inner & (-gap < parameters.side_lane)
        outer_block = outer & (gap < parameters.side_lane)
        return SideRoom(
            jnp.where(inner_block, False, result.can_in),
            jnp.where(outer_block, False, result.can_out),
            jnp.where(inner, jnp.minimum(result.in_space, -gap), result.in_space),
            jnp.where(outer, jnp.minimum(result.out_space, gap), result.out_space),
            jnp.where(inner_block, peer_lane[index], result.inner_blocker_lane),
            jnp.where(inner_block, index, result.inner_blocker_index),
        )
    return jax.lax.fori_loop(0, MAX_PEERS, body, state)


def is_visible(gap, lane_gap, vision):
    """Prevalidated finite/non-subnormal f32 cone kernel; x64 caller policy."""
    gap, lane_gap, vision = f32(gap), f32(lane_gap), f32(vision)
    valid = (vision > f32(0.0)) & (gap >= f32(0.0)) & (gap <= vision)
    widest = ((jnp.asarray(1.0, jnp.float64) * f32(0.6388888955116272).astype(jnp.float64) + f32(0.1111111119389534).astype(jnp.float64)) * jnp.asarray(0.5, jnp.float64)).astype(jnp.float32)
    ratio = gap / vision
    width = ((ratio.astype(jnp.float64) * f32(0.6388888955116272).astype(jnp.float64) + f32(0.1111111119389534).astype(jnp.float64)) * jnp.asarray(0.5, jnp.float64)).astype(jnp.float32)
    return valid & (jnp.abs(lane_gap) <= widest) & (jnp.abs(lane_gap) <= width)


def is_target(gap, speed_gap, owner_target, other_target, other_speed, directly_blocking, other_blocked):
    """Prevalidated finite/non-subnormal f32 public Boolean predicate; x64 policy."""
    gap, speed_gap = jnp.asarray(gap, jnp.float64).astype(jnp.float32), jnp.asarray(speed_gap, jnp.float64).astype(jnp.float32)
    owner_target = jnp.asarray(owner_target, jnp.float64).astype(jnp.float32)
    other_target = jnp.asarray(other_target, jnp.float64).astype(jnp.float32)
    other_speed = jnp.asarray(other_speed, jnp.float64).astype(jnp.float32)
    directly_blocking, other_blocked = jnp.asarray(directly_blocking, bool), jnp.asarray(other_blocked, bool)
    direct = other_speed < owner_target
    range_ok = (gap >= f32(1.0)) & (gap <= f32(20.0)) & (speed_gap > f32(0.0)) & (gap / speed_gap <= f32(15.0))
    blocked_result = other_speed < owner_target
    unblocked_result = other_target < owner_target
    indirect = jnp.where(blocked_result == unblocked_result, blocked_result, jnp.where(other_blocked, blocked_result, unblocked_result))
    return jnp.where(gap < f32(-1.0), False, jnp.where(directly_blocking, direct, jnp.where(range_ok, indirect, False)))


def side_room_checked(owner_position, owner_lane, max_lane, peer_position, peer_lane, parameters: BlockParameters):
    require_x64()
    positions, lanes = np.asarray(peer_position), np.asarray(peer_lane)
    if positions.ndim != 1 or lanes.shape != positions.shape or positions.size > MAX_PEERS:
        raise ValueError("peer positions and lanes must be equal one-dimensional rows of at most 32")
    scalars = tuple(_real_scalar(x, "side-room scalar") for x in (owner_position, owner_lane, max_lane, parameters.side_distance, parameters.side_lane))
    if (not np.issubdtype(positions.dtype, np.number) or not np.issubdtype(lanes.dtype, np.number) or np.iscomplexobj(positions) or np.iscomplexobj(lanes) or not np.isfinite(positions).all() or not np.isfinite(lanes).all()):
        raise ValueError("side-room inputs must be finite scalars and rows")
    with np.errstate(over="ignore"):
        narrowed_scalars = np.asarray([owner_position, owner_lane, max_lane, parameters.side_distance, parameters.side_lane], np.float32)
    if not np.isfinite(narrowed_scalars).all():
        raise ValueError("side-room scalars must remain finite after float32 storage")
    if float(narrowed_scalars[2]) <= 0 or float(narrowed_scalars[3]) <= 0 or float(narrowed_scalars[4]) <= 0:
        raise ValueError("max_lane and block parameters must be positive")
    pos = np.zeros(MAX_PEERS, np.float32); lane = np.zeros(MAX_PEERS, np.float32)
    pos[:positions.size], lane[:lanes.size] = positions, lanes
    if not np.isfinite(pos).all() or not np.isfinite(lane).all():
        raise ValueError("side-room rows must remain finite after float32 storage")
    _reject_nonzero_subnormals(narrowed_scalars, pos, lane, pos - narrowed_scalars[0], lane - narrowed_scalars[1], narrowed_scalars[2] - narrowed_scalars[1])
    return side_room(f32(owner_position), f32(owner_lane), f32(max_lane), pos, lane, np.int32(positions.size), BlockParameters(f32(parameters.side_distance), f32(parameters.side_lane)))


def is_visible_checked(gap, lane_gap, vision):
    require_x64()
    _real_scalar(gap, "visibility input"); _real_scalar(lane_gap, "visibility input"); _real_scalar(vision, "visibility input")
    with np.errstate(over="ignore"):
        narrowed = np.asarray([gap, lane_gap, vision], np.float32)
    if not np.isfinite(narrowed).all():
        raise ValueError("visibility inputs must remain finite after float32 storage")
    _reject_nonzero_subnormals(narrowed)
    return is_visible(*[f32(value) for value in narrowed])


def is_target_checked(gap, speed_gap, owner_target, other_target, other_speed, directly_blocking, other_blocked):
    require_x64()
    values = (gap, speed_gap, owner_target, other_target, other_speed)
    for value in values: _real_scalar(value, "overtaking numeric input")
    if not all(np.asarray(x).shape == () and np.asarray(x).dtype == np.dtype(np.bool_) for x in (directly_blocking, other_blocked)):
        raise ValueError("overtaking blocker inputs must be scalar booleans")
    with np.errstate(over="ignore"):
        narrowed = np.asarray(values, np.float32)
    if not np.isfinite(narrowed).all():
        raise ValueError("overtaking inputs must remain finite after float32 storage")
    _reject_nonzero_subnormals(narrowed)
    return is_target(*[float(value) for value in narrowed], directly_blocking, other_blocked)
