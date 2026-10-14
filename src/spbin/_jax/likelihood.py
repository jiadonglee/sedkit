"""Masked -2 ln L on fixed 168-channel arrays and the interpolated dust prior."""

import jax.numpy as jnp
from jax.scipy.special import log_ndtr


def neg2_log_likelihood(sed_flux, sed_error, mask, parallax_mas, flux_10pc,
                        columns, variance, transmission):
    """-2 ln L (constant n ln(2pi) omitted), chi2 and observed-scale model.

    Equivalent to masking first: unobserved channels get unit diagonal,
    zero residual and zero columns, so they contribute nothing to chi2 or
    the determinant while every array keeps its full 168 shape. Their
    values are never read, so NaN there (an unobserved band) is harmless.
    """
    scale = (parallax_mas / 100.0)**2
    flux = flux_10pc * scale * transmission
    observed = mask > 0
    diagonal = jnp.where(observed, sed_error**2 + variance * transmission**2, 1.0)
    residual = jnp.where(observed, flux - sed_flux, 0.0)
    masked_columns = jnp.where(observed[:, None], columns * transmission[:, None], 0.0)
    small = jnp.eye(masked_columns.shape[1]) + masked_columns.T @ (
        masked_columns / diagonal[:, None])
    nuisance = jnp.linalg.solve(small, masked_columns.T @ (residual / diagonal))
    whitened = residual - masked_columns @ nuisance
    chi2 = jnp.sum(whitened**2 / diagonal) + nuisance @ nuisance
    value = chi2 + jnp.sum(mask * jnp.log(diagonal)) + jnp.linalg.slogdet(small)[1]
    return value, chi2, flux


def dust_penalty(extinction, parallax_mas, dust_par, dust_mean, dust_sigma, dust_fail):
    """-2 log p(E|distance) by linear interpolation of the per-star grid.

    `dust_fail` marks grid points where the map has no support; interpolating
    it gives a smooth transition into the rejected region.
    """
    mean = jnp.interp(parallax_mas, dust_par, dust_mean)
    sigma = jnp.interp(parallax_mas, dust_par, dust_sigma)
    failed = jnp.interp(parallax_mas, dust_par, dust_fail)
    value = ((extinction - mean) / sigma)**2 + 2 * jnp.log(sigma) + 2 * log_ndtr(mean / sigma)
    return jnp.where(failed > 0, 1e30, value)
