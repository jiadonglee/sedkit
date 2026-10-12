"""Pure-JAX stellar forward model: PARSEC tracks, network, hot table, likelihood atoms.

Every function here is compiled with fixed shapes: a coeval pair is two
components whose fluxes add with weight zero for a single star (a fit whose
q is fixed at 0 compiles the one-component layout instead), channel masks
are 0/1 weights, and the low-rank model-error basis always has all 12
columns (4 per error term). A fit with no SPHEREx channel compiles the
first 66 channels (XP61, J/H/Ks, W1/W2) only.
Outside their domain the components return safe finite values and a `valid`
flag; the objective turns the flag into one large finite value.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..model import ERROR_BLEND_K, HOT_BLEND_K, HOT_CHANNELS, SOLAR_LOGG, SOLAR_TEFF


def smoothstep(x):
    u = jnp.clip(x, 0, 1)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def gelu(values):
    return 0.5 * values * (1 + jnp.tanh(
        jnp.sqrt(2 / jnp.pi) * (values + 0.044715 * values**3)))


@jax.custom_jvp
def sqrt_weight(weight):
    """sqrt with a zero gradient at zero weight, so unused error terms stay NaN-free."""
    return jnp.sqrt(weight)


@sqrt_weight.defjvp
def _sqrt_weight_jvp(primals, tangents):
    (weight,) = primals
    (dweight,) = tangents
    safe = jnp.where(weight > 0, weight, 1.0)
    return jnp.sqrt(weight), jnp.where(weight > 0, 0.5 / jnp.sqrt(safe), 0.0) * dweight


def search(grid, x, side="left"):
    """jnp.searchsorted by one comparison against every node (the grids are short)."""
    return jnp.searchsorted(grid, x, side=side, method="compare_all")


def interp_clamped(grid, values, x):
    """np.interp of x on grid (columns of values), by gather on a traced grid."""
    index = jnp.clip(search(grid, x) - 1, 0, grid.shape[0] - 2)
    x0, x1 = grid[index], grid[index + 1]
    t = jnp.clip((x - x0) / (x1 - x0), 0, 1)
    return values[index] + t * (values[index + 1] - values[index])


def bilinear(teff_ax, logg_ax, values, teff, logg):
    """Bilinear interpolation of values (n_teff, n_logg[, n]) at a scalar point."""
    i = jnp.clip(search(teff_ax, teff) - 1, 0, teff_ax.shape[0] - 2)
    j = jnp.clip(search(logg_ax, logg) - 1, 0, logg_ax.shape[0] - 2)
    x = (teff - teff_ax[i]) / (teff_ax[i + 1] - teff_ax[i])
    y = (logg - logg_ax[j]) / (logg_ax[j + 1] - logg_ax[j])
    corners = values[i, j], values[i + 1, j], values[i, j + 1], values[i + 1, j + 1]
    weights = (1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y
    terms = [w[...] * c for w, c in zip(weights, corners)]
    return terms[0] + terms[1] + terms[2] + terms[3]


def track(arr, masses, age_gyr, feh):
    """Labels [Teff, M_Ks, G-Ks, [M/H]], log g and radius from the PARSEC grid.

    masses is a fixed-length (n,) of component masses; returns per-component
    labels, logg, radius and a validity mask equal to the numpy NaN case.
    """
    k = jnp.clip(search(arr.metals, feh, side="right"), 1, arr.metals.shape[0] - 1)
    metal_weight = jnp.clip(
        (feh - arr.metals[k - 1]) / (arr.metals[k] - arr.metals[k - 1]), 0, 1)
    log_age = jnp.log10(age_gyr) + 9
    h = jnp.clip(search(arr.log_ages, log_age, side="right"), 1, arr.log_ages.shape[0] - 1)
    age_weight = jnp.clip(
        (log_age - arr.log_ages[h - 1]) / (arr.log_ages[h] - arr.log_ages[h - 1]), 0, 1)

    def one_metal(metal):
        lo = jnp.maximum(arr.track_lo[metal, h - 1], arr.track_lo[metal, h])
        hi = jnp.minimum(arr.track_hi[metal, h - 1], arr.track_hi[metal, h])
        valid = (masses >= lo) & (masses <= hi)

        def one_age(age):
            grid = arr.track_mass[metal, age]
            values = arr.track_values[metal, age]
            return jax.vmap(lambda m: interp_clamped(grid, values, m))(masses)

        return ((1 - age_weight) * one_age(h - 1) + age_weight * one_age(h), valid)

    low_pars, low_valid = one_metal(k - 1)
    high_pars, high_valid = one_metal(k)
    pars = ((1 - metal_weight) * jnp.where(low_valid[:, None], low_pars, 0.0)
            + metal_weight * jnp.where(high_valid[:, None], high_pars, 0.0))
    valid = jnp.where(metal_weight < 1e-9, low_valid,
                      jnp.where(metal_weight > 1 - 1e-9, high_valid,
                                low_valid & high_valid))
    teff = 10**pars[:, 0]
    log_radius = 0.5 * pars[:, 1] - 2 * jnp.log10(teff / SOLAR_TEFF)
    logg = SOLAR_LOGG + jnp.log10(masses) - 2 * log_radius
    labels = jnp.stack([teff, pars[:, 2], pars[:, 3] - pars[:, 2],
                        jnp.full(masses.shape, feh)], axis=1)
    return labels, logg, 10**log_radius, valid


def predict_labels(arr, labels, channels=None):
    """Absolute 10-pc channel fluxes for one labels row [Teff, M_Ks, G-Ks, [M/H]].

    channels=66 (static) returns the XP-head channels only; the default is all 168.
    """
    hidden = (labels[jnp.array([0, 2, 3])] - arr.x_center) / arr.x_scale
    for index, (weight, bias) in enumerate(zip(arr.trunk_weights, arr.trunk_biases)):
        hidden = hidden @ weight.T + bias
        if index < len(arr.trunk_weights) - 1:
            hidden = gelu(hidden)
    values = (hidden @ arr.xp_weight.T + arr.xp_bias) * arr.y_scale[:66] + arr.y_center[:66]
    if channels != 66:
        values = jnp.concatenate([
            values, (hidden @ arr.spherex_weight.T + arr.spherex_bias) * arr.y_scale[66:]
            + arr.y_center[66:]])
    log_flux = values - values[63] + arr.ks0_log - 0.4 * labels[1]
    temperature = (labels[0] - 2600) / 500
    color = labels[2] - 5.5
    gate = 1 - smoothstep((labels[0] - 2800) / 600)
    features = jnp.array([1.0, temperature, color, temperature * color,
                          temperature**2, color**2])
    basis = arr.cal_spectral_basis[:values.shape[0]]
    log_flux += gate * ((features @ arr.cal_coefficients) @ basis.T)
    return 10**log_flux


def predict_hot(arr, teff, logg, radius):
    """Absolute 10-pc XP61 + J/H/Ks fluxes (first 64 channels) from the hot table."""
    ln_flux = bilinear(arr.hot_teff_ax, arr.hot_logg_ax, arr.hot_ln_flux, teff, logg)
    w = bilinear(arr.hot_teff_ax, arr.hot_logg_ax, arr.hot_balmer_w, teff, logg)
    low, high = arr.hot_cool_edge_k
    cool = 1 - smoothstep((teff - low) / (high - low))
    ln_flux = ln_flux + arr.hot_delta_a + w * arr.hot_delta_b + cool * arr.hot_delta_c
    return jnp.exp(ln_flux) * radius**2


def in_domain(arr, labels):
    """Network training-domain check for one labels row (or across boxes)."""
    inside = labels >= arr.boxes[:, :, 0]
    inside &= labels <= arr.boxes[:, :, 1]
    return jnp.any(jnp.all(inside, axis=-1))


def in_hot_domain(arr, teff, logg, feh):
    """Hot-table support check for scalar component coordinates."""
    teff_lo, teff_hi = arr.hot_box[0]
    logg_lo, logg_hi = arr.hot_box[1]
    feh_lo, feh_hi = arr.hot_box[2]
    return ((teff >= teff_lo) & (teff <= teff_hi) & (logg >= logg_lo)
            & (logg <= logg_hi) & (feh >= feh_lo) & (feh <= feh_hi))


class Evaluation(NamedTuple):
    """Forward model of one coeval system on the n-component layout (n = 2, or 1)."""

    labels: jax.Array  # (n, 4)
    logg: jax.Array  # (n,)
    radius: jax.Array  # (n,)
    mg: jax.Array  # (n,)
    hot_weight: jax.Array  # (n,)
    components: jax.Array  # (n, c), c = 168 channels or 66 (XP head)
    flux_10pc: jax.Array  # (c,)
    beta_g: jax.Array
    a_phot_over_a1: jax.Array
    valid: jax.Array


def evaluate(arr, m1, q, age_gyr, feh, pair=True, channels=None):
    """One star (q=0) or a coeval pair on the fixed two-component layout.

    The second component participates only when q > 0; its flux and its
    domain checks are ignored for a single star, matching numpy's
    one-component branch. pair=False (static) evaluates the primary alone
    and ignores q, which gives the same result as q=0 without the second
    track lookup and network pass. channels=66 (static) predicts the
    XP-head channels only. Returns safe finite values with `valid` False
    wherever numpy would return None.
    """
    masses = jnp.array([m1, m1 * jnp.maximum(q, 1e-30)] if pair else [m1])
    labels, logg, radius, track_valid = track(arr, masses, age_gyr, feh)
    weight = (smoothstep((labels[:, 0] - HOT_BLEND_K[0])
                         / (HOT_BLEND_K[1] - HOT_BLEND_K[0])) if arr.hot
              else jnp.zeros(masses.shape))
    domain = jax.vmap(lambda row: in_domain(arr, row))(labels)
    component_ok = track_valid & ((weight == 1) | domain)
    if arr.hot:
        hot_domain = jax.vmap(lambda t, g: in_hot_domain(arr, t, g, feh))(labels[:, 0], logg)
        component_ok &= (weight == 0) | hot_domain
    participate = (q > 0) & pair
    ok = component_ok[0] & (~participate | component_ok[-1])

    safe_labels = jnp.where(track_valid[:, None], labels, jnp.array([5000.0, 4.0, 1.5, 0.0]))
    cool = jax.vmap(lambda row: predict_labels(arr, row, channels))(safe_labels)
    if arr.hot:
        safe_teff = jnp.clip(labels[:, 0], arr.hot_box[0, 0], arr.hot_box[0, 1])
        safe_logg = jnp.clip(logg, arr.hot_box[1, 0], arr.hot_box[1, 1])
        safe_radius = jnp.where(track_valid, radius, 1.0)
        hot64 = jax.vmap(lambda t, g, r: predict_hot(arr, t, g, r))(
            safe_teff, safe_logg, safe_radius)
        hot = jnp.zeros(cool.shape).at[:, :HOT_CHANNELS].set(hot64)
        components = (1 - weight[:, None]) * cool + weight[:, None] * hot
    else:
        components = cool
    components = jnp.where(component_ok[:, None], components, 0.0)
    flux_10pc = components.sum(axis=0)
    mg = labels[:, 1] + labels[:, 2]
    beta = jnp.where(participate, 10**(-0.4 * (mg[-1] - mg[0])), 0.0)
    denominator = jnp.where(participate, q * (1 + beta), 1.0)
    a_phot = jnp.where(participate, (q - beta) / denominator, 0.0)
    return Evaluation(labels, logg, radius, mg, weight, components, flux_10pc,
                      beta, a_phot, ok)


def error_factors(arr, teff, flux_10pc, scale):
    """Low-rank model-error columns (n, 12) and diagonal variance for the first n channels.

    The primary's Teff selects the term weights: v2.1_cold below 3800 K,
    v2.1_warm above 4200 K, smoothstep blend in between, and with hot=True a
    further handover to the hot term over 7000--7498 K. Zero-weight terms
    leave zero columns, so the 12x12 Woodbury matrix stays fixed-shaped.
    """
    warm = smoothstep((teff - ERROR_BLEND_K[0]) / (ERROR_BLEND_K[1] - ERROR_BLEND_K[0]))
    hot = (smoothstep((teff - HOT_BLEND_K[0]) / (HOT_BLEND_K[1] - HOT_BLEND_K[0]))
           if arr.hot else 0.0)
    cold, warm_term, hot_term = 1 - warm, warm * (1 - hot), hot
    column_scale = jnp.concatenate([
        jnp.full((columns,), sqrt_weight(jnp.asarray(weight)))
        for weight, columns in zip((cold, warm_term, hot_term), arr.error_columns)])
    weights = jnp.array([cold, warm_term, hot_term])
    flux = flux_10pc * scale
    n = flux.shape[0]
    columns = flux[:, None] * arr.error_basis[:n] * column_scale
    variance = jnp.sum(weights * (flux[:, None] * arr.error_diag[:n])**2, axis=1)
    return columns, variance
