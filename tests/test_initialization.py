import jax
import jax.numpy as jnp
import pytest
from globalsim_jax.initialization import initial_base_wisdom, ranked_base_stat, team_rank_bonus


def test_source_backed_base_stat_order():
    assert float(initial_base_wisdom(1000., -1)) == 979.9999389648438
    assert [float(team_rank_bonus(x)) for x in (0, 1, 5, 6, 10, 11, 16, 21)] == [0., 50., 50., 30., 30., 20., 10., 0.]
    assert float(ranked_base_stat(1201., 0, 1)) == 1250.5
    assert float(ranked_base_stat(9999., 2, 1)) == 2000.


def test_adjust_overcap_rejects_disabled_x64_without_mutating_configuration():
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="x64"):
            ranked_base_stat(1201., 0, 1)
    finally:
        jax.config.update("jax_enable_x64", True)
