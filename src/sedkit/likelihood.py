"""SED likelihood atoms for composition with external orbital models."""

import numpy as np

from .fit import neg2_log_likelihood
from .model import StellarModel


def loglike_sed(sed, *, m1, q=0.0, age_gyr=5.0, feh=0.0,
                parallax_mas=None, model=None, use_wise=False):
    """Return log L_SED without priors or a catalogue parallax constraint.

    Mass is in solar units, age in Gyr and parallax in mas. q=M2/M1;
    q=0 is a single star. Fluxes remain on the absolute observed scale.
    Model covariance and its determinant are included. The constant
    -N/2 log(2pi) is omitted: compare on one shared data mask.
    Unsupported stellar components or nonpositive parallax return -inf.
    Age/metallicity validation follows StellarModel.evaluate.
    """
    parallax = sed.parallax_mas if parallax_mas is None else parallax_mas
    if not np.isfinite(parallax) or parallax <= 0:
        return -np.inf
    model = StellarModel() if model is None else model
    prediction = model.evaluate(m1, q, age_gyr, feh)
    if prediction is None:
        return -np.inf
    mask = sed.fit_mask(use_wise=use_wise)
    if not mask.any():
        raise ValueError("SED likelihood requires at least one fitted channel")
    return -0.5 * neg2_log_likelihood(sed, prediction, model, mask, parallax)[0]
