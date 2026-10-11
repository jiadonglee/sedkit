"""Masked -2 ln L on fixed 168-channel arrays and the interpolated dust prior."""

import jax.numpy as jnp
from jax.scipy.special import log_ndtr


def neg2_log_likelihood(sed_flux, sed_error, mask, parallax_mas, flux_10pc,
                        columns, variance, transmission):
    """-2 ln L (constant n ln(2pi) omitted), chi2 and observed-scale model.

    Equivalent to masking first: unobserved channels get unit diagonal,
    zero residual and zero columns, so they contribute nothing to chi2 or
    the determinant while every array keeps its full 168 shape.
    """
    scale = (parallax_mas / 100.0)**2
    flux = flux_10pc * scale * transmission
    diagonal = jnp.where(mask > 0, sed_error**2 + variance * transmission**2, 1.0)
    residual = mask * (flux - sed_flux)
    masked_columns = columns * transmission[:, None] * mask[:, None]
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
