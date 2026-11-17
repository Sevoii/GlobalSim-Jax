"""Fixture-backed numerical primitives only; this is not a race simulator."""
from .rng import RetailRng, integer, next_double, next_int, real, real_distance, seed
from .physics import Parameters, advance_start_delay, bounded_stat, finish_time, phase_boundaries, target_speed_indices
from .initialization import adjust_overcap, initial_base_wisdom, ranked_base_stat, team_rank_bonus

__all__ = ["RetailRng", "seed", "next_int", "next_double", "integer", "real", "real_distance", "Parameters", "phase_boundaries", "target_speed_indices", "advance_start_delay", "finish_time", "bounded_stat", "adjust_overcap", "team_rank_bonus", "initial_base_wisdom", "ranked_base_stat"]
