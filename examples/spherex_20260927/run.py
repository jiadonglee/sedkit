"""Download or import a real SB2 SPHEREx spectrum, fit and plot it."""

from pathlib import Path
import argparse
import json
import shutil

from sedkit import SED, download_spherex, load_spherex, fit, plot

HERE = Path(__file__).parent
SOURCE_ID = "858860697467058688"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="use the online downloader")
    args = parser.parse_args()
    sed = SED.load(HERE.parent / "sb2_20260927" / (SOURCE_ID + ".npz"))
    if args.download:
        sed = download_spherex(sed, cache_dir=HERE / "data")
        products = HERE / "data/spherex" / SOURCE_ID
        for name in ("aperture_spectrum.csv", "aperture_exposures.csv", "metadata.json"):
            shutil.copy2(products / name, HERE / name)
    else:
        sed = load_spherex(sed, HERE / "aperture_spectrum.csv", method="aperture")
        sed.metadata["spherex"].update(json.loads((HERE / "metadata.json").read_text()))
    sed.metadata["spherex"]["spectrum_path"] = "aperture_spectrum.csv"
    sed.save(HERE / "sed.npz")
    result = fit(sed, age_gyr=None, feh=None)
    fig = plot(sed, result, path=HERE / "fit.png")
    fig.savefig(HERE / "fit.pdf", bbox_inches="tight", pad_inches=.02)
    fields = ("m1", "q", "age_gyr", "feh", "chi2", "n_fit", "at_bounds", "converged")
    summary = dict(source_id=SOURCE_ID, n_spherex=int(sed.mask[66:].sum()),
                   single={key: result["single"][key] for key in fields},
                   binary={key: result["binary"][key] for key in fields},
                   delta=result["delta"], q_rv=0.8679092693033352,
                   interpretation="observed aperture spectrum; age reaches the model boundary")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
