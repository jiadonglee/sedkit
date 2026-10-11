"""JAX port of the stellar forward model and the `fit` route (prototype)."""

import jax

jax.config.update("jax_enable_x64", True)
