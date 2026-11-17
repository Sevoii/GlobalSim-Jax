"""M3b deterministic lane primitives (fixed-capacity prevalidated kernels)."""

import jax
import jax.numpy as jnp
import numpy as np


MAX_LANES = 32
_STACK_CAPACITY = 64
OVERLAP_POSITION = jnp.float32(0.4)
OVERLAP_LANE = jnp.float32(0.4 / 18.0)
_DEPTH = jnp.asarray([0] + [2 * n.bit_length() for n in range(1, 33)], jnp.int32)


def _swap(order, left, right, enabled):
    first, second = order[left], order[right]
    order = order.at[left].set(jnp.where(enabled, second, first))
    return order.at[right].set(jnp.where(enabled, first, second))


def _swap_if_greater(order, keys, left, right):
    return _swap(order, left, right, (left != right) & (keys[order[left]] > keys[order[right]]))


def _insertion_sort(order, keys, lo, hi):
    """Source's strict-less insertion branch (partitions no larger than 16)."""
    def outer(offset, order):
        index = lo + offset
        active = index < hi
        value = order[jnp.minimum(index + 1, MAX_LANES - 1)]

        def shift(_, state):
            order, cursor, shifting = state
            safe = jnp.maximum(cursor, 0)
            move = shifting & (cursor >= lo) & (keys[value] < keys[order[safe]])
            order = order.at[safe + 1].set(jnp.where(move, order[safe], order[safe + 1]))
            return order, cursor - move.astype(jnp.int32), move

        order, cursor, _ = jax.lax.fori_loop(0, 16, shift, (order, index, active))
        slot = jnp.clip(cursor + 1, lo, hi)
        return order.at[slot].set(jnp.where(active, value, order[slot]))

    return jax.lax.fori_loop(0, 16, outer, order)


def _small_sort(order, keys, lo, hi, size):
    """Exact source small cases: networks for two/three, insertion otherwise."""
    def two(value):
        return _swap_if_greater(value, keys, lo, hi)

    def three(value):
        value = _swap_if_greater(value, keys, lo, hi - 1)
        value = _swap_if_greater(value, keys, lo, hi)
        return _swap_if_greater(value, keys, hi - 1, hi)

    return jax.lax.cond(
        size == 2,
        two,
        lambda value: jax.lax.cond(size == 3, three, lambda other: _insertion_sort(other, keys, lo, hi), value),
        order,
    )


def _down_heap(order, keys, lo, start, size):
    """One-indexed source heap sift for introsort's depth fallback."""
    value = order[lo + start - 1]

    def sift(_, state):
        order, index, moving = state
        child = 2 * index
        can_move = moving & (index <= size // 2)
        child_slot, next_slot = lo + child - 1, lo + child
        choose_right = (child < size) & (keys[order[child_slot]] < keys[order[next_slot]])
        child = child + choose_right.astype(jnp.int32)
        child_slot = lo + child - 1
        move = can_move & (keys[value] < keys[order[child_slot]])
        target = lo + index - 1
        order = order.at[target].set(jnp.where(move, order[child_slot], order[target]))
        return order, jnp.where(move, child, index), move

    order, index, _ = jax.lax.fori_loop(0, MAX_LANES, sift, (order, start, True))
    return order.at[lo + index - 1].set(value)


def _heap_sort(order, keys, lo, hi):
    size = hi - lo + 1

    def build(offset, order):
        index = size // 2 - offset
        return jax.lax.cond(index >= 1, lambda x: _down_heap(x, keys, lo, index, size), lambda x: x, order)

    order = jax.lax.fori_loop(0, MAX_LANES, build, order)

    def extract(offset, order):
        index = size - offset
        def active(value):
            return _down_heap(_swap(value, lo, lo + index - 1, True), keys, lo, jnp.int32(1), index - 1)
        return jax.lax.cond(index >= 2, active, lambda x: x, order)

    return jax.lax.fori_loop(0, MAX_LANES, extract, order)


def _partition(order, keys, lo, hi):
    middle = lo + (hi - lo) // 2
    order = _swap_if_greater(order, keys, lo, middle)
    order = _swap_if_greater(order, keys, lo, hi)
    order = _swap_if_greater(order, keys, middle, hi)
    pivot = order[middle]
    order = _swap(order, middle, hi - 1, True)

    def partition_step(_, state):
        order, left, right, done = state
        running = ~done
        left = jnp.where(running, left + 1, left)

        def advance_left(_, state):
            cursor, moving = state
            safe = jnp.minimum(cursor, hi)
            move = moving & (keys[order[safe]] < keys[pivot])
            return cursor + move.astype(jnp.int32), move

        left, _ = jax.lax.fori_loop(0, MAX_LANES, advance_left, (left, running))
        right = jnp.where(running, right - 1, right)

        def advance_right(_, state):
            cursor, moving = state
            safe = jnp.maximum(cursor, lo)
            move = moving & (keys[pivot] < keys[order[safe]])
            return cursor - move.astype(jnp.int32), move

        right, _ = jax.lax.fori_loop(0, MAX_LANES, advance_right, (right, running))
        crossed = left >= right
        order = _swap(order, left, right, running & ~crossed)
        return order, left, right, done | crossed

    order, left, _, _ = jax.lax.fori_loop(0, MAX_LANES, partition_step, (order, lo, hi - 1, False))
    return _swap(order, left, hi - 1, True), left


def comparison_sort_indices(keys, count):
    """Rust ``comparison_sort`` index permutation for up to 32 int32 keys.

    The explicit 64-entry stack reproduces source recursion order: left is
    pushed before right so LIFO executes the recursive right partition first.
    Equal keys are intentionally unstable beyond the <=16 insertion branch.
    """
    keys = jnp.asarray(keys, jnp.int32)
    order = jnp.arange(MAX_LANES, dtype=jnp.int32)
    los = jnp.zeros(_STACK_CAPACITY, jnp.int32).at[0].set(0)
    his = jnp.zeros(_STACK_CAPACITY, jnp.int32).at[0].set(count - 1)
    depths = jnp.zeros(_STACK_CAPACITY, jnp.int32).at[0].set(_DEPTH[count])

    def process(_, state):
        order, los, his, depths, pointer = state
        active = pointer > 0
        top = jnp.maximum(pointer - 1, 0)
        lo, hi, depth = los[top], his[top], depths[top]
        pointer = pointer - active.astype(jnp.int32)
        size = hi - lo + 1
        usable = active & (hi > lo)

        def small(value):
            order, *tail = value
            return (_small_sort(order, keys, lo, hi, size), *tail)

        def heap(value):
            order, *tail = value
            return (_heap_sort(order, keys, lo, hi), *tail)

        def split(value):
            order, stack_lo, stack_hi, stack_depth, point = value
            order, pivot = _partition(order, keys, lo, hi)
            child_depth = depth - 1
            stack_lo = stack_lo.at[point].set(lo)
            stack_hi = stack_hi.at[point].set(pivot - 1)
            stack_depth = stack_depth.at[point].set(child_depth)
            stack_lo = stack_lo.at[point + 1].set(pivot + 1)
            stack_hi = stack_hi.at[point + 1].set(hi)
            stack_depth = stack_depth.at[point + 1].set(child_depth)
            return order, stack_lo, stack_hi, stack_depth, point + 2

        value = (order, los, his, depths, pointer)
        return jax.lax.cond(
            usable,
            lambda x: jax.lax.cond(size <= 16, small, lambda y: jax.lax.cond(depth == 0, heap, split, y), x),
            lambda x: x,
            value,
        )

    return jax.lax.fori_loop(0, _STACK_CAPACITY, process, (order, los, his, depths, jnp.int32(1)))[0]


def comparison_sort_indices_checked(keys, count):
    """Host boundary for the 32-entry sort kernel; rejects rather than clips."""
    values = np.asarray(keys)
    if values.ndim != 1 or values.shape[0] > MAX_LANES:
        raise ValueError("comparison sort accepts one key row of at most 32 entries")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("comparison sort keys must have an integer dtype")
    if values.size and (
        np.any(values < np.iinfo(np.int32).min)
        or np.any(values > np.iinfo(np.int32).max)
    ):
        raise ValueError("comparison sort keys must be representable as int32")
    if not isinstance(count, (int, np.integer)) or not 0 <= count <= values.shape[0]:
        raise ValueError("count must be an integer within the supplied key row")
    padded = np.zeros(MAX_LANES, dtype=np.int32)
    padded[: values.shape[0]] = values.astype(np.int32, copy=False)
    return comparison_sort_indices(jnp.asarray(padded), jnp.int32(count))[:count]


def resolve_overlaps(position, lane, force_checks, count):
    """Source `resolve_overlaps`: strict thresholds, chained pushes, no rails.

    Empty rows leave padded flags unchanged.  A nonempty row clears only the
    first sorted flag; later non-overlap flags keep their prior values.
    """
    position = jnp.asarray(position, jnp.float32)
    lane = jnp.asarray(lane, jnp.float32)
    force_checks = jnp.asarray(force_checks, bool)
    order = comparison_sort_indices(lane.astype(jnp.int32), count)
    first = order[0]
    force_checks = jax.lax.cond(count > 0, lambda x: x.at[first].set(False), lambda x: x, force_checks)

    def resolve(offset, state):
        lane, force = state
        active = offset < count
        previous, current = order[offset - 1], order[offset]
        overlaps = active & (jnp.abs(position[current] - position[previous]) < OVERLAP_POSITION) & (jnp.abs(lane[current] - lane[previous]) < OVERLAP_LANE)
        lane = lane.at[current].set(jnp.where(overlaps, lane[previous] + OVERLAP_LANE, lane[current]))
        force = force.at[current].set(jnp.where(overlaps, True, force[current]))
        return lane, force

    lane, force_checks = jax.lax.fori_loop(1, MAX_LANES, resolve, (lane, force_checks))
    return (lane, force_checks), order


def resolve_overlaps_checked(position, lane, force_checks):
    """Host-checked wrapper for a non-padded lane row of at most 32 runners."""
    position, lane, force = np.asarray(position), np.asarray(lane), np.asarray(force_checks)
    if position.ndim != 1 or lane.shape != position.shape or force.shape != position.shape:
        raise ValueError("position, lane, and force_checks must be equal one-dimensional rows")
    if force.size and force.dtype != np.dtype(np.bool_):
        raise ValueError("force_checks must have boolean dtype")
    if position.size > MAX_LANES:
        raise ValueError("overlap resolution accepts at most 32 runners")
    if not np.isfinite(position).all() or not np.isfinite(lane).all():
        raise ValueError("position and lane must be finite")
    count = position.size
    padded_position = np.zeros(MAX_LANES, dtype=np.float32)
    padded_lane = np.zeros(MAX_LANES, dtype=np.float32)
    padded_force = np.zeros(MAX_LANES, dtype=np.bool_)
    with np.errstate(over="ignore"):
        narrowed_position = position.astype(np.float32)
        narrowed_lane = lane.astype(np.float32)
    if not np.isfinite(narrowed_position).all() or not np.isfinite(narrowed_lane).all():
        raise ValueError("position and lane must remain finite after float32 storage")
    narrowed_lane_f64 = narrowed_lane.astype(np.float64)
    if np.any(narrowed_lane_f64 < np.iinfo(np.int32).min) or np.any(narrowed_lane_f64 > np.iinfo(np.int32).max):
        raise ValueError("lane values must be representable by the source int32 sort key")
    padded_position[:count], padded_lane[:count], padded_force[:count] = narrowed_position, narrowed_lane, force
    (result_lane, result_force), order = resolve_overlaps(padded_position, padded_lane, padded_force, count)
    return result_lane[:count], result_force[:count], order[:count]
