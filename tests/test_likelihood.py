import numpy as np
import pytest

from sedkit import SED, StellarModel, loglike_sed
from sedkit.fit import neg2_log_likelihood


def test_composable_likelihood_has_no_catalogue_parallax_constraint():
    model = StellarModel()
    prediction = model.evaluate(.75, .8, 5, 0)
    flux = prediction['flux_10pc']*(20/100)**2
    sed = SED(flux, .03*flux, np.ones(168,bool), 17, .1)
    original = sed.flux.copy(), sed.error.copy(), sed.mask.copy()
    actual = loglike_sed(sed,m1=.75,q=.8,parallax_mas=20,model=model)
    expected = -.5*neg2_log_likelihood(sed,prediction,model,sed.fit_mask(),20)[0]
    assert actual == pytest.approx(expected)
    other_catalogue = SED(flux,.03*flux,np.ones(168,bool),25,.01)
    assert actual == loglike_sed(other_catalogue,m1=.75,q=.8,parallax_mas=20,model=model)
    for before,after in zip(original,(sed.flux,sed.error,sed.mask)):
        np.testing.assert_array_equal(before,after)


def test_composable_likelihood_rejects_unsupported_stars_and_distance():
    model = StellarModel()
    sed = SED(np.ones(168),np.ones(168),np.ones(168,bool),20)
    assert loglike_sed(sed,m1=.75,q=.01,model=model) == -np.inf
    assert loglike_sed(sed,m1=.75,q=.8,parallax_mas=0,model=model) == -np.inf
