"""All model tables packed into fixed-shape arrays forming one pytree.

`ModelArrays.build(model)` converts a `StellarModel` into a frozen pytree of
JAX arrays. Variable-length PARSEC tracks are padded with increasing sentinel
masses beyond each track's real range, so the padded grid reproduces
`np.interp` inside the range while the per-track `track_lo`/`track_hi` arrays
reproduce the range check that numpy implements with NaN.
"""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..extinction import extinction_curve
from ..model import DATA, KS_ZERO_FLAMBDA, StellarModel

# Columns of the packed model-error basis: cold, warm and hot terms.
ERROR_TERMS = ("v2.1_cold", "v2.1_warm", "hot")
DUST_GRID_POINTS = 161


class ModelArrays(eqx.Module):
    """Static model data as a pytree; `hot` selects the hot-route behaviour."""

    hot: bool = eqx.field(static=True)

    track_mass: jax.Array  # (n_metal, n_age, n_pad)
    track_values: jax.Array  # (n_metal, n_age, n_pad, 4): logTe, logL, Ks, G
    track_lo: jax.Array  # (n_metal, n_age)
    track_hi: jax.Array  # (n_metal, n_age)
    log_ages: jax.Array  # (n_age,) log10(age/yr) of the PARSEC age grid
    metals: jax.Array  # (n_metal,)

    trunk_weights: tuple
    trunk_biases: tuple
    xp_weight: jax.Array
    xp_bias: jax.Array
    spherex_weight: jax.Array
    spherex_bias: jax.Array
    x_center: jax.Array  # (3,)
    x_scale: jax.Array  # (3,)
    y_center: jax.Array  # (168,)
    y_scale: jax.Array  # (168,)
    ks0_log: jax.Array  # scalar log10(KS_ZERO_FLAMBDA)

    cal_coefficients: jax.Array  # (6, 3)
    cal_spectral_basis: jax.Array  # (168, 3)

    boxes: jax.Array  # (n_box, 4, 2) network prediction-domain boxes
    hot_box: jax.Array  # (3, 2) hot-table support: teff, logg, feh

    hot_teff_ax: jax.Array  # (n_teff,)
    hot_logg_ax: jax.Array  # (n_logg,)
    hot_ln_flux: jax.Array  # (n_teff, n_logg, 64)
    hot_balmer_w: jax.Array  # (n_teff, n_logg)
    hot_delta_a: jax.Array  # (64,)
    hot_delta_b: jax.Array  # (64,)
    hot_delta_c: jax.Array  # (64,)
    hot_cool_edge_k: jax.Array  # (2,)

    error_basis: jax.Array  # (168, n_err) cold, warm and hot term columns
    error_diag: jax.Array  # (168, 3) per-term diagonal standard deviations
    error_columns: tuple = eqx.field(static=True)  # per-term column counts
    support: jax.Array  # (168,) bool, channels the model predicts
    wavelength_um: jax.Array  # (168,)
    extinction_curve: jax.Array  # (168,) optical depth per ZGR23 E

    @classmethod
    def build(cls, model):
        """Convert a numpy `StellarModel` (hot or not) into packed arrays."""
        n_metal, n_age = len(model.metals), len(next(iter(model.tracks.values())))
        n_pad = max(len(track) for metal in model.tracks.values() for track in metal.values())
        mass = np.zeros((n_metal, n_age, n_pad))
        values = np.zeros((n_metal, n_age, n_pad, 4))
        lo = np.zeros((n_metal, n_age))
        hi = np.zeros((n_metal, n_age))
        for i, metal in enumerate(model.metals):
            tracks = model.tracks[metal]
            for j, age in enumerate(sorted(tracks)):
                track = tracks[age]
                mass[i, j, :len(track)] = track[:, 0]
                values[i, j, :len(track)] = track[:, 1:]
                lo[i, j], hi[i, j] = track[0, 0], track[-1, 0]
                mass[i, j, len(track):] = hi[i, j] + 1.0 + np.arange(n_pad - len(track))
                values[i, j, len(track):] = track[-1, 1:]
        ages = np.array(sorted(next(iter(model.tracks.values()))))
        for metal in model.tracks.values():
            if np.any(np.array(sorted(metal)) != ages):
                raise ValueError("PARSEC metals do not share one age grid")
            for track in metal.values():
                if np.any(np.diff(track[:, 0]) <= 0):
                    raise ValueError("PARSEC track masses must increase")

        params = model.params
        trunk = [(params[f"trunk_weight_{i}"], params[f"trunk_bias_{i}"])
                 for i in range(sum(k.startswith("trunk_weight_") for k in params))]
        errors = model.errors
        if not model.hot:
            hot_error = json.loads((DATA / "hot" / "summary.json").read_text())["error_term"]
            errors = dict(errors, **_load(DATA / "hot" / "model_error.npz"))
        else:
            hot_error = model.hot_summary["error_term"]
        bases, diags, widths = [], [], []
        for term in ERROR_TERMS:
            name = hot_error if term == "hot" else term
            # The hot term is NaN beyond the channels the hot route predicts;
            # zero it so the fixed-width packing stays finite at zero weight.
            bases.append(np.nan_to_num(errors[name + "/basis"]))
            diags.append(np.nan_to_num(errors[name + "/diag"]))
            widths.append(errors[name + "/basis"].shape[1])
        norm = model.norm
        boxes = np.array(model.summary["prediction_domain_boxes"])
        box = np.array([[model.hot_summary["support"][key][0],
                         model.hot_summary["support"][key][1]]
                        for key in ("teff", "logg", "feh")]) if model.hot else np.zeros((3, 2))
        table = model.hot_table if model.hot else _load(DATA / "hot" / "hot_table.npz")
        return cls(
            hot=model.hot,
            track_mass=jnp.asarray(mass), track_values=jnp.asarray(values),
            track_lo=jnp.asarray(lo), track_hi=jnp.asarray(hi),
            log_ages=jnp.asarray(ages), metals=jnp.asarray(model.metals),
            trunk_weights=tuple(jnp.asarray(w) for w, _ in trunk),
            trunk_biases=tuple(jnp.asarray(b) for _, b in trunk),
            xp_weight=jnp.asarray(params["xp_head_weight_0"]),
            xp_bias=jnp.asarray(params["xp_head_bias_0"]),
            spherex_weight=jnp.asarray(params["spherex_head_weight_0"]),
            spherex_bias=jnp.asarray(params["spherex_head_bias_0"]),
            x_center=jnp.asarray(norm["x_center"]), x_scale=jnp.asarray(norm["x_scale"]),
            y_center=jnp.asarray(np.r_[norm["xp_y_center"], norm["spherex_y_center"]]),
            y_scale=jnp.asarray(np.r_[norm["xp_y_scale"], norm["spherex_y_scale"]]),
            ks0_log=jnp.log10(KS_ZERO_FLAMBDA),
            cal_coefficients=jnp.asarray(model.calibration["coefficients"]),
            cal_spectral_basis=jnp.asarray(model.calibration["spectral_basis"]),
            boxes=jnp.asarray(boxes), hot_box=jnp.asarray(box),
            hot_teff_ax=jnp.asarray(table["teff_ax"]), hot_logg_ax=jnp.asarray(table["logg_ax"]),
            hot_ln_flux=jnp.asarray(table["ln_flux"]), hot_balmer_w=jnp.asarray(table["balmer_w"]),
            hot_delta_a=jnp.asarray(table["delta_a"]), hot_delta_b=jnp.asarray(table["delta_b"]),
            hot_delta_c=jnp.asarray(table["delta_c"]),
            hot_cool_edge_k=jnp.asarray(table["cool_edge_k"]),
            error_basis=jnp.asarray(np.concatenate(bases, axis=1)),
            error_diag=jnp.asarray(np.stack(diags, axis=1)),
            error_columns=tuple(widths),
            support=jnp.asarray(model.support),
            wavelength_um=jnp.asarray(model.wavelength_um),
            extinction_curve=jnp.asarray(extinction_curve(model.wavelength_um)),
        )


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
