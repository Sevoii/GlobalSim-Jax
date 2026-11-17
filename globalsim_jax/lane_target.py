"""Bounded Rust ``TargetState`` cache collection and persistent updates.

This is a snapshot primitive, not automatic target selection. Stable peer slots
encode the Rust visible Vec order.  ``update_targets`` adds only the
default-feature persistent retention transition: it has no PassingSearch,
final target generation, RNG, pacing, field integration, or experimental gap
refresh.  Low-level kernels require prevalidated f32 rows and caller x64.

Checked update padding is deliberately canonical: inactive cache/live horse
rows must be zero, inactive visibility slots false, and inactive target slots
``-1``.  This is stricter than the source Vec representation because the JAX
kernel evaluates fixed-size vector expressions before masks select an active
row; it keeps poisoned padding out of those expressions.  The unchecked
kernel is for callers that have already established that invariant.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .field import BlockParameters, front_lane_limit
from .lane_queries import (
    MAX_PEERS, SideRoom, BlockParameters as SideParameters,
    _reject_nonzero_subnormals, is_visible, side_room,
)
from .precision import require_x64


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TargetCache:
    owner: jax.Array
    peers: jax.Array
    count: jax.Array
    visible: jax.Array
    side: SideRoom
    front: jax.Array
    front_distance: jax.Array
    front_lane: jax.Array
    force_in: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TargetState:
    """Bounded persistent portion of Rust ``TargetState`` after collection.

    ``cache`` is supplied by ``collect``; target generation/search, RNG and
    collection attachment/lifecycle are intentionally outside this API.
    ``extra_outward`` uses NaN for Rust's ``None``.
    """
    cache: TargetCache
    targets: jax.Array              # padded peer indices, -1 sentinel
    target_count: jax.Array
    cooldown: jax.Array
    force_check: jax.Array
    force_in: jax.Array
    extra_outward: jax.Array
    previous_can_in: jax.Array      # -1 none, 0 false, 1 true
    skip_passing: jax.Array


def initialize_target_state(cache, *, targets=(), cooldown=0., force_check=False,
                            extra_outward=jnp.nan, previous_can_in=-1,
                            skip_passing=False):
    """Host convenience constructor; callers attach a collected snapshot."""
    raw = np.asarray(targets)
    if raw.ndim != 1 or len(raw) > MAX_PEERS or (len(raw) and raw.dtype.kind not in "iu"):
        raise ValueError("targets must be an integer row of at most 32 entries")
    # Rust keeps usize values until its first retain/drop pass.  A synthetic
    # state may therefore contain an int32-representable stale nonnegative
    # index (for example 99), which must be dropped rather than rejected.
    if len(raw) and (np.any(raw < 0) or np.any(raw > np.iinfo(np.int32).max)):
        raise ValueError("targets must be nonnegative int32-representable indices")
    for value, name in ((cooldown, "cooldown"), (extra_outward, "extra_outward")):
        a = np.asarray(value)
        if a.shape != () or np.iscomplexobj(a) or not np.issubdtype(a.dtype, np.number):
            raise ValueError(f"{name} must be a scalar numeric value")
    narrowed_cooldown = np.asarray(cooldown, np.float32)
    narrowed_extra = np.asarray(extra_outward, np.float32)
    if not np.isfinite(narrowed_cooldown):
        raise ValueError("cooldown must remain finite float32")
    if np.isinf(narrowed_extra) or (not np.isfinite(narrowed_extra) and not np.isnan(narrowed_extra)):
        raise ValueError("extra_outward must be finite float32 or the NaN None sentinel")
    previous = np.asarray(previous_can_in)
    def bool_scalar(value):
        array = np.asarray(value)
        return array.shape == () and array.dtype == np.dtype(np.bool_)
    if (previous.shape != () or previous.dtype.kind not in "iu" or previous.dtype == np.dtype(np.bool_)
            or int(previous) not in (-1, 0, 1)
            or not bool_scalar(force_check) or not bool_scalar(skip_passing)):
        raise ValueError("invalid lifecycle flag/sentinel")
    padded = np.full(MAX_PEERS, -1, np.int32); padded[:len(raw)] = raw
    return TargetState(cache, jnp.asarray(padded), jnp.int32(len(raw)),
                       jnp.asarray(cooldown, jnp.float32), jnp.asarray(force_check, bool),
                       cache.force_in, jnp.asarray(narrowed_extra, jnp.float32),
                       jnp.asarray(int(previous), jnp.int32), jnp.asarray(skip_passing, bool))


def _front_index(owner, cache, parameters):
    gap = cache.peers[:, 0] - owner[0]
    lane_gap = cache.peers[:, 1] - owner[1]
    ok = ((jnp.arange(MAX_PEERS) < cache.count) & (gap > 0)
          & (gap < jnp.float32(parameters.front_distance))
          & (jnp.abs(lane_gap) <= front_lane_limit(gap, parameters)))
    nearest = jnp.min(jnp.where(ok, gap, jnp.inf))
    index = jnp.min(jnp.where(ok & (gap == nearest), jnp.arange(MAX_PEERS), MAX_PEERS))
    return jnp.where(index == MAX_PEERS, -1, index).astype(jnp.int32)


def _has_front_blocker(owner, peers, count, parameters):
    gap = peers[:, 0] - owner[0]
    return jnp.any((jnp.arange(MAX_PEERS) < count) & (gap > 0)
                   & (gap < jnp.float32(parameters.front_distance))
                   & (jnp.abs(peers[:, 1] - owner[1]) <= front_lane_limit(gap, parameters)))


def update_targets(state, owner, live, count, dt, parameters=BlockParameters(), *, cooldown=0.):
    """Prevalidated default-feature Rust ``update_targets`` port.

    ``live`` is aligned to the cache's owner-excluded peers.  The experimental
    gap-refresh feature is deliberately absent (and therefore never sets
    ``skip_passing`` after its mandatory reset).
    """
    owner = jnp.asarray(owner, jnp.float32); live = jnp.asarray(live, jnp.float32)
    count = jnp.asarray(count, jnp.int32); dt = jnp.asarray(dt, jnp.float32)
    same = ((state.cache.owner[0].view(jnp.uint32) == owner[0].view(jnp.uint32))
            & (state.cache.owner[1].view(jnp.uint32) == owner[1].view(jnp.uint32))
            & (state.cache.front_distance.view(jnp.uint32) == jnp.asarray(parameters.front_distance, jnp.float32).view(jnp.uint32))
            & (state.cache.front_lane.view(jnp.uint32) == jnp.asarray(parameters.front_lane, jnp.float32).view(jnp.uint32)))
    front = jnp.where(same, state.cache.front, _front_index(owner, state.cache, parameters))
    active = jnp.arange(MAX_PEERS) < count
    gap = state.cache.peers[:, 0] - owner[0]
    speed_gap = owner[2] - state.cache.peers[:, 2]
    direct = live[:, 2] < owner[3]
    range_ok = (gap >= 1) & (gap <= 20) & (speed_gap > 0) & (gap / speed_gap <= 15)
    blocked_result = direct; unblocked_result = live[:, 3] < owner[3]
    # Source passes every cached peer followed by the current owner to the
    # private blocker iterator.  Keep both observations even though current
    # owner inclusion cannot itself flip a finite indirect-eligibility case.
    candidate = state.cache.peers[:, None, :2]
    cached_rows = state.cache.peers[None, :, :2]
    peer_gap = cached_rows[:, :, 0] - candidate[:, :, 0]
    peer_lane_gap = cached_rows[:, :, 1] - candidate[:, :, 1]
    peer_blocked = ((jnp.arange(MAX_PEERS)[None, :] < count)
                    & (peer_gap > 0)
                    & (peer_gap < jnp.float32(parameters.front_distance))
                    & (jnp.abs(peer_lane_gap) <= front_lane_limit(peer_gap, parameters)))
    owner_gap = owner[0] - state.cache.peers[:, 0]
    owner_blocked = ((owner_gap > 0) & (owner_gap < jnp.float32(parameters.front_distance))
                     & (jnp.abs(owner[1] - state.cache.peers[:, 1])
                        <= front_lane_limit(owner_gap, parameters)))
    blocked = jnp.any(peer_blocked, axis=1) | owner_blocked
    eligible = jnp.where(gap < -1, False, jnp.where(front == jnp.arange(MAX_PEERS), direct,
        jnp.where(range_ok, jnp.where(blocked_result == unblocked_result, blocked_result, jnp.where(blocked, blocked_result, unblocked_result)), False))) & active
    old_valid = (jnp.arange(MAX_PEERS) < state.target_count) & (state.targets >= 0) & (state.targets < count)
    old = state.targets
    # Rust first drops only stale indices, appends visible newcomers against
    # that retained list, then revalidates *only* the old prefix.
    def pack(mask, values):
        key = jnp.where(mask, jnp.arange(MAX_PEERS), MAX_PEERS)
        order = jnp.argsort(key)
        result = values[order]
        n = jnp.sum(mask, dtype=jnp.int32)
        return jnp.where(jnp.arange(MAX_PEERS) < n, result, -1), n
    targets, n = pack(old_valid, old)
    existing_count = n
    def add_visible(i, carry):
        values, size = carry
        add = state.cache.visible[i] & eligible[i] & ~jnp.any((jnp.arange(MAX_PEERS) < size) & (values == i))
        values = values.at[size].set(jnp.where(add, i, values[size]))
        return values, size + add.astype(jnp.int32)
    targets, n = jax.lax.fori_loop(0, MAX_PEERS, add_visible, (targets, n))
    old_prefix = jnp.arange(MAX_PEERS) < existing_count
    keep = (jnp.arange(MAX_PEERS) < n) & (~old_prefix | eligible[jnp.clip(targets, 0, MAX_PEERS - 1)])
    targets, n = pack(keep, targets)
    add_front = (front >= 0) & ~jnp.any((jnp.arange(MAX_PEERS) < n) & (targets == front))
    targets = targets.at[n].set(jnp.where(add_front, front, targets[n])); n = n + add_front.astype(jnp.int32)
    cooldown_value = jnp.where(n > 0, jnp.asarray(cooldown, jnp.float32), state.cooldown)
    decrement = (n == 0) & (state.cooldown > 0)
    cooldown_value = jnp.where(decrement, state.cooldown - dt, cooldown_value)
    force = state.force_check | (decrement & (cooldown_value <= 0))
    can_in = state.cache.side.can_in
    return TargetState(state.cache, targets, n, cooldown_value, force, state.force_in,
                       state.extra_outward, can_in.astype(jnp.int32), jnp.asarray(False))


def _checked_scalar(array, dtype, name, *, finite=True):
    value = np.asarray(array)
    if value.shape != () or value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must be a scalar {np.dtype(dtype).name}")
    if finite and not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _checked_array(array, shape, dtype, name, *, finite=False):
    value = np.asarray(array)
    if value.shape != shape or value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must have shape {shape} and dtype {np.dtype(dtype).name}")
    if finite and not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _checked_target_state(state):
    """Validate every serialized fixed-capacity state/cache leaf on host."""
    if not isinstance(state, TargetState) or not isinstance(state.cache, TargetCache):
        raise ValueError("state must be a TargetState with a TargetCache")
    cache = state.cache
    count = int(_checked_scalar(cache.count, np.int32, "cache.count"))
    if not 0 <= count <= MAX_PEERS:
        raise ValueError("cache.count must be within fixed capacity")
    owner = _checked_array(cache.owner, (4,), np.float32, "cache.owner", finite=True)
    peers = _checked_array(cache.peers, (MAX_PEERS, 4), np.float32, "cache.peers", finite=True)
    visible = _checked_array(cache.visible, (MAX_PEERS,), np.bool_, "cache.visible")
    front = int(_checked_scalar(cache.front, np.int32, "cache.front"))
    if not -1 <= front < count:
        raise ValueError("cache.front must be -1 or an active peer index")
    front_distance = _checked_scalar(cache.front_distance, np.float32, "cache.front_distance")
    front_lane = _checked_scalar(cache.front_lane, np.float32, "cache.front_lane")
    if front_distance <= 0 or front_lane <= 0:
        raise ValueError("cached front parameters must be positive")
    _checked_scalar(cache.force_in, np.bool_, "cache.force_in", finite=False)
    side = cache.side
    if not isinstance(side, SideRoom):
        raise ValueError("cache.side must be SideRoom")
    _checked_scalar(side.can_in, np.bool_, "cache.side.can_in", finite=False)
    _checked_scalar(side.can_out, np.bool_, "cache.side.can_out", finite=False)
    _checked_scalar(side.in_space, np.float32, "cache.side.in_space")
    _checked_scalar(side.out_space, np.float32, "cache.side.out_space")
    inner_lane = _checked_scalar(side.inner_blocker_lane, np.float32, "cache.side.inner_blocker_lane", finite=False)
    if np.isinf(inner_lane) or (not np.isfinite(inner_lane) and not np.isnan(inner_lane)):
        raise ValueError("cache.side.inner_blocker_lane must be finite or NaN")
    inner_index = int(_checked_scalar(side.inner_blocker_index, np.int32, "cache.side.inner_blocker_index"))
    if not -1 <= inner_index < count:
        raise ValueError("cache.side.inner_blocker_index must be -1 or active")
    target_count = int(_checked_scalar(state.target_count, np.int32, "state.target_count"))
    if not 0 <= target_count <= MAX_PEERS:
        raise ValueError("state.target_count must be within fixed capacity")
    targets = _checked_array(state.targets, (MAX_PEERS,), np.int32, "state.targets")
    active_targets = targets[:target_count]
    if np.any(active_targets < 0):
        raise ValueError("active target indices must be nonnegative")
    if len(np.unique(active_targets)) != len(active_targets):
        raise ValueError("active target indices must be unique")
    if np.any(targets[target_count:] != -1):
        raise ValueError("inactive target padding must be -1")
    _checked_scalar(state.cooldown, np.float32, "state.cooldown")
    _checked_scalar(state.force_check, np.bool_, "state.force_check", finite=False)
    _checked_scalar(state.force_in, np.bool_, "state.force_in", finite=False)
    if bool(np.asarray(state.force_in)) != bool(np.asarray(cache.force_in)):
        raise ValueError("state.force_in must preserve the collected cache value")
    extra = _checked_scalar(state.extra_outward, np.float32, "state.extra_outward", finite=False)
    if np.isinf(extra) or (not np.isfinite(extra) and not np.isnan(extra)):
        raise ValueError("state.extra_outward must be finite or NaN None sentinel")
    previous = int(_checked_scalar(state.previous_can_in, np.int32, "state.previous_can_in"))
    if previous not in (-1, 0, 1):
        raise ValueError("state.previous_can_in must be -1, 0, or 1")
    _checked_scalar(state.skip_passing, np.bool_, "state.skip_passing", finite=False)
    # Canonical inactive cache padding protects the vectorized, prevalidated
    # kernel from evaluating arbitrary values Rust's Vec iterator never sees.
    if np.any(peers[count:] != 0) or np.any(visible[count:]):
        raise ValueError("inactive cache padding must be zero horses and false visibility")
    _reject_nonzero_subnormals(front_distance, front_lane)
    return count, owner, peers


def update_targets_checked(state, owner, live, dt, *, cooldown, parameters=BlockParameters()):
    """Checked host boundary for default-feature persistent updates.

    Stale nonnegative target indices are accepted for Rust's drop pass;
    duplicates are rejected as unreachable synthetic state.
    """
    require_x64()
    count, cached_owner, cached_peers = _checked_target_state(state)
    owner = _checked_array(owner, (4,), np.float32, "owner", finite=True)
    live = _checked_array(live, (MAX_PEERS, 4), np.float32, "live", finite=True)
    dt = _checked_scalar(dt, np.float32, "dt")
    cooldown = _checked_scalar(cooldown, np.float32, "cooldown")
    if dt < 0 or cooldown < 0:
        raise ValueError("dt and cooldown parameter must be nonnegative")
    if np.any(live[count:] != 0):
        raise ValueError("inactive live padding must be zero horses")
    if not isinstance(parameters, BlockParameters):
        raise ValueError("parameters must be default BlockParameters")
    parameter_front = np.asarray([parameters.front_distance, parameters.front_lane], np.float32)
    if not np.isfinite(parameter_front).all() or np.any(parameter_front <= 0):
        raise ValueError("front parameters must be finite positive float32 values")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cached_gap = cached_peers[:count, :2] - cached_owner[:2]
        current_gap = cached_peers[:count, :2] - owner[:2]
        speed_gap = owner[2] - cached_peers[:count, 2]
        cooldown_gap = np.asarray(state.cooldown) - dt
        all_owner_gap = owner[0] - cached_peers[:count, 0]
        pair_gap = cached_peers[None, :count, :2] - cached_peers[:count, None, :2]
        fd = np.float32(parameters.front_distance)
        pair_cone = pair_gap[:, :, 0] / fd * np.float32(.6)
        owner_cone = all_owner_gap / fd * np.float32(.6)
        current_cone = current_gap[:, 0] / fd * np.float32(.6)
        # The overtaking ratio is source-evaluated only after its strict range
        # and positive-speed checks.  Guard that actual domain, not dead rows.
        ratio_mask = ((current_gap[:, 0] >= 1) & (current_gap[:, 0] <= 20) & (speed_gap > 0))
        ratios = current_gap[:, 0][ratio_mask] / speed_gap[ratio_mask]
        if (not np.isfinite(cached_gap).all() or not np.isfinite(current_gap).all()
                or not np.isfinite(speed_gap).all() or not np.isfinite(cooldown_gap)
                or not np.isfinite(pair_gap).all() or not np.isfinite(pair_cone).all()
                or not np.isfinite(owner_cone).all() or not np.isfinite(current_cone).all()
                or not np.isfinite(ratios).all()):
            raise ValueError("update derived intermediates overflow float32")
    _reject_nonzero_subnormals(owner, live[:count], cached_owner, cached_peers[:count],
                               np.asarray(dt), np.asarray(cooldown), np.asarray(state.cooldown),
                               parameter_front,
                               cached_gap, current_gap, speed_gap, cooldown_gap,
                               pair_gap, pair_cone, owner_cone, current_cone, ratios)
    return update_targets(state, owner, live, count, dt, parameters, cooldown=cooldown)


def collect(owner, peers, count, max_lane, vision, phase, empty_in_move,
            force_in_threshold, parameters=BlockParameters()):
    """Collect owner-excluded, stable peer rows [32,4]: position/lane/speed/target.

    `visible` is a padded mask, whose ascending true slots are the source Vec.
    All indices refer to this peer row, including front and inner blocker.
    """
    owner = jnp.asarray(owner, jnp.float32)
    peers = jnp.asarray(peers, jnp.float32)
    count = jnp.asarray(count, jnp.int32)
    active = jnp.arange(MAX_PEERS) < count
    gap = peers[:, 0] - owner[0]
    lane_gap = peers[:, 1] - owner[1]
    visible = active & is_visible(gap, lane_gap, vision)
    eligible = (active & (gap > jnp.float32(0))
                & (gap < jnp.float32(parameters.front_distance))
                & (jnp.abs(lane_gap) <= front_lane_limit(gap, parameters)))
    nearest = jnp.min(jnp.where(eligible, gap, jnp.inf))
    front = jnp.min(jnp.where(eligible & (gap == nearest),
                             jnp.arange(MAX_PEERS), MAX_PEERS))
    front = jnp.where(front == MAX_PEERS, -1, front).astype(jnp.int32)
    side = side_room(owner[0], owner[1], max_lane, peers[:, 0], peers[:, 1],
                     count, SideParameters(jnp.float32(parameters.side_distance),
                                           jnp.float32(parameters.side_lane)))
    inward_gap = owner[1] - peers[:, 1]
    inner_nearby = jnp.any(active & (jnp.abs(gap) < jnp.float32(parameters.side_distance))
                           & (inward_gap > jnp.float32(0))
                           & (inward_gap < jnp.float32(parameters.side_lane)))
    force_in = (jnp.asarray(empty_in_move, bool) & (phase < 1)
                & (owner[1] >= jnp.float32(force_in_threshold)) & inner_nearby)
    # Canonical padding is part of this wrapper representation, not source state.
    return TargetCache(owner, jnp.where(active[:, None], peers, jnp.float32(0)),
                       count, visible, side, front,
                       jnp.float32(parameters.front_distance),
                       jnp.float32(parameters.front_lane), force_in)


def collect_roster(roster, count, owner_index, max_lane, vision, phase,
                   empty_in_move, force_in_threshold, parameters=BlockParameters()):
    """Equivalent to collect_roster's unsorted path, capacity 32 total runners.

    The Rust sorted-nearby optimization is omitted. Owner removal preserves
    roster order and the resulting owner-excluded peer-index convention.
    """
    roster = jnp.asarray(roster, jnp.float32)
    slots = jnp.arange(MAX_PEERS)
    indices = slots + (slots >= owner_index).astype(jnp.int32)
    peers = roster[jnp.minimum(indices, MAX_PEERS - 1)]
    return collect(roster[owner_index], peers, count - 1, max_lane, vision,
                   phase, empty_in_move, force_in_threshold, parameters)


def _canonical_collection_inputs(owner, peers, *, max_lane, vision, phase,
                                 empty_in_move, force_in_threshold, parameters):
    """Validate/narrow one owner plus an owner-excluded peer row on the host."""
    # This is intentionally host-only: roster validation reuses it without
    # dispatching a throwaway JAX collection kernel.

    owner, peers = np.asarray(owner), np.asarray(peers)
    if owner.shape != (4,) or peers.ndim != 2 or peers.shape[1] != 4 or len(peers) > MAX_PEERS:
        raise ValueError("owner must be [4] and peers [0..32,4]")
    values = (max_lane, vision, force_in_threshold, parameters.front_distance,
              parameters.front_lane, parameters.side_distance, parameters.side_lane)
    for value in (owner, peers, *values):
        a = np.asarray(value)
        if (not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a)
                or not np.isfinite(a).all()):
            raise ValueError("collection inputs must be finite real numbers")
    if any(np.asarray(v).shape != () for v in values):
        raise ValueError("collection configuration must be scalar")
    if np.asarray(phase).shape != () or np.asarray(phase).dtype.kind not in 'iu' or not -(2**31) <= int(phase) < 2**31:
        raise ValueError("phase must be an int32-representable integer")
    if np.asarray(empty_in_move).shape != () or np.asarray(empty_in_move).dtype != np.bool_:
        raise ValueError("empty_in_move must be a scalar boolean")
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        owner, peers, config = owner.astype(np.float32), peers.astype(np.float32), np.asarray(values, np.float32)
        if not all(np.isfinite(a).all() for a in (owner, peers, config)):
            raise ValueError("collection inputs overflow float32 storage")
        if config[0] <= 0 or np.any(config[3:] <= 0):
            raise ValueError("max_lane and block parameters must be positive")
        gaps = peers[:, :2] - owner[:2]
        space = config[0] - owner[1]
        narrowing = (gaps[:, 0] / config[3]) * np.float32(.6)
        if not np.isfinite(gaps).all() or not np.isfinite(space) or not np.isfinite(narrowing).all():
            raise ValueError("collection query intermediates overflow float32")
        # Rust computes gap / vision only after its valid-cone and widest-lane
        # early exits.  Check that actual source-evaluated ratio domain, rather
        # than rejecting an exceptional division on a row the source skips.
        widest = np.float32((float(np.float32(.6388888955116272))
                             + float(np.float32(.1111111119389534))) * .5)
        source_ratio = ((np.arange(len(peers)) >= 0) & (config[1] > 0)
                        & (gaps[:, 0] >= 0) & (gaps[:, 0] <= config[1])
                        & (np.abs(gaps[:, 1]) <= widest))
        ratio = np.divide(gaps[:, 0], config[1], dtype=np.float32)
        if not np.isfinite(ratio[source_ratio]).all():
            raise ValueError("source-evaluated visibility ratio overflows float32")
    _reject_nonzero_subnormals(owner, peers, config, gaps, space, narrowing,
                               ratio[source_ratio])
    padded = np.zeros((MAX_PEERS, 4), np.float32)
    padded[:len(peers)] = peers
    p = BlockParameters(*map(float, config[3:]))
    return owner, padded, len(peers), config, np.int32(phase), empty_in_move, p


def collect_checked(owner, peers, *, max_lane, vision, phase, empty_in_move,
                    force_in_threshold, parameters=BlockParameters()):
    """Host boundary for a standalone snapshot; accepts 0..32 supplied peers.

    Rejects nonfinite/overflow/subnormal stored values and derived query
    operands. Negative vision remains valid (no visible peers), as in Rust.
    """
    require_x64()
    owner, peers, count, config, phase, empty_in_move, p = _canonical_collection_inputs(
        owner, peers, max_lane=max_lane, vision=vision, phase=phase,
        empty_in_move=empty_in_move, force_in_threshold=force_in_threshold,
        parameters=parameters)
    return collect(owner, peers, count,
                   config[0], config[1], phase, empty_in_move, config[2], p)


def collect_roster_checked(roster, owner_index, *, max_lane, vision, phase,
                           empty_in_move, force_in_threshold,
                           parameters=BlockParameters()):
    """Checked unsorted ``collect_roster`` boundary, for 1..32 runners.

    This is deliberately the owner-removal path only.  It does not accept a
    position sort or expose Rust's nearby-roster optimization.
    """
    require_x64()
    roster = np.asarray(roster)
    owner = np.asarray(owner_index)
    if (roster.ndim != 2 or roster.shape[1:] != (4,) or not 1 <= len(roster) <= MAX_PEERS):
        raise ValueError("roster must be [1..32,4]")
    if owner.shape != () or owner.dtype.kind not in "iu" or not 0 <= int(owner) < len(roster):
        raise ValueError("owner_index must be an in-range integer")
    # Share float/domain checks and canonicalization without dispatching a
    # throwaway single-owner collection.
    owner_row = roster[int(owner)]
    peer_rows = np.concatenate((roster[:int(owner)], roster[int(owner) + 1:]), axis=0)
    _, _, _, config, phase, empty_in_move, p = _canonical_collection_inputs(
        owner_row, peer_rows, max_lane=max_lane, vision=vision, phase=phase,
        empty_in_move=empty_in_move, force_in_threshold=force_in_threshold,
        parameters=parameters)
    padded = np.zeros((MAX_PEERS, 4), np.float32)
    padded[:len(roster)] = roster.astype(np.float32)
    return collect_roster(padded, len(roster), np.int32(owner), config[0],
                          config[1], phase, empty_in_move, config[2], p)
