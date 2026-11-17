"""Precision policy: callers, not imports, choose JAX's process-wide x64 setting."""
import jax

def require_x64() -> None:
    if not jax.config.jax_enable_x64:
        raise RuntimeError("globalsim_jax reference primitives require JAX x64. Before importing globalsim_jax, call jax.config.update('jax_enable_x64', True).")
