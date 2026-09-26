"""Download and fit Gaia DR3 1521154374020165376."""

from pathlib import Path
import json

from sedlet import download, fit, plot


def main():
    output = Path(__file__).parent
    sed = download("1521154374020165376", cache_dir=output / "data")
    result = fit(sed, age_gyr=5.0, feh=0.0)
    fig = plot(sed, result, path=output / "real_source.png")
    fig.savefig(output / "real_source.pdf")
    summary = {"source_id": sed.source_id, "age_gyr_fixed": 5, "feh_fixed": 0,
               "delta": result["delta"]}
    for hypothesis in ("single", "binary"):
        summary[hypothesis] = {key: result[hypothesis][key] for key in
                              ("m1", "m2", "q", "beta_g", "chi2", "n_fit", "converged", "at_bounds")}
    (output / "real_source.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
