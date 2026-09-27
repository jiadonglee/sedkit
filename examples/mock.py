"""Fit a coeval stellar-binary mock and save its SED diagnostic."""

from pathlib import Path
import json

import numpy as np

from sedkit import SED, StellarModel, fit, plot


def main():
    model = StellarModel()
    truth = model.evaluate(m1=0.75, q=0.8, age_gyr=5.0, feh=0.0)
    mean = truth["flux_10pc"] * (50 / 100)**2
    error = 0.03 * mean
    observed = mean + np.random.default_rng(42).normal(size=168) * error
    observed[66:], error[66:] = np.nan, np.nan
    mask = np.zeros(168, bool)
    mask[:66] = True
    sed = SED(observed, error, mask, parallax_mas=50.0, source_id="",
              metadata={"scenario": "matched stellar-binary mock"})
    result = fit(sed, model=model, age_gyr=5, feh=0, fit_parallax=False)
    output = Path(__file__).parent
    fig = plot(sed, result, path=output / "mock.png")
    fig.savefig(output / "mock.pdf")
    summary = {"truth": {"m1": 0.75, "q": 0.8, "age_gyr": 5, "feh": 0},
               "binary_fit": {key: result["binary"][key] for key in
                              ("m1", "q", "chi2", "beta_g", "converged")},
               "delta": result["delta"], "measurement_noise_fraction": 0.03,
               "interpretation": "matched-model algorithm check"}
    (output / "mock.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
