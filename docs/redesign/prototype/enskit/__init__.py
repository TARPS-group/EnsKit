"""Throwaway prototype of the redesigned package (placeholder name ``enskit``).

Reuses ``pyeki.linalg`` unchanged as the operator layer.
"""
import jax

jax.config.update("jax_enable_x64", True)

from . import linalg, distribution, maps, kalman  # noqa: E402,F401
