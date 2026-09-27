"""Optional SPHEREx acquisition and import of XphereX spectra."""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def load_spherex(sed, path, *, method=None):
    """Attach an XphereX channel-binned Jy CSV to a copy of an SED.

    Columns are wavelength_um, flux_jy and error_jy. No interpolation,
    normalization, zero-point correction or uncertainty adjustment is applied.
    The wavelengths must match the bundled SPHEREx grid. Negative fluxes are
    retained, and missing channels stay NaN and masked. Both
    aperture_spectrum.csv and psf_spectrum.csv are accepted.
    """
    path = Path(path).expanduser()
    table = np.genfromtxt(path, delimiter=",", names=True)
    required = {"wavelength_um", "flux_jy", "error_jy"}
    if not required.issubset(table.dtype.names or ()):
        raise ValueError("SPHEREx CSV needs wavelength_um, flux_jy and error_jy columns")
    table = np.atleast_1d(table)
    order = np.argsort(table["wavelength_um"])
    wave = table["wavelength_um"][order]
    expected = sed.wavelength_um[66:].astype(float)
    if not len(wave):
        raise ValueError("SPHEREx CSV has no spectral channels")
    channels = np.argmin(np.abs(wave[:, None] - expected), axis=1)
    if (not np.isfinite(wave).all()
            or np.any(np.abs(wave - expected[channels]) > 1e-6)
            or len(np.unique(channels)) != len(channels)):
        raise ValueError("SPHEREx wavelengths must match unique channels on the bundled 102-channel grid")
    # F_nu [Jy] -> F_lambda [1e-18 W m^-2 nm^-1], at native distance.
    factor = 299792458.0 * 1e-5 / wave**2
    flux, error = np.full(102, np.nan), np.full(102, np.nan)
    flux[channels] = table["flux_jy"][order] * factor
    error[channels] = table["error_jy"][order] * factor
    result = sed.with_spherex(expected, flux, error)
    result.metadata["spherex"] = dict(
        spectrum_path=str(path.resolve()), input_unit="Jy",
        method=method or path.stem.removesuffix("_spectrum"),
        extraction_source="https://github.com/jiadonglee/XphereX")
    return result


def download_spherex(sed, *, cache_dir="data", refresh=False, radius_arcmin=0.8):
    """Download QR2 pixels and attach an aperture spectrum to a copy of sed.

    Requires sedkit[spherex] and network access on the first run. Uses the
    Gaia position, proper motion and reference epoch stored by download().
    An isolated Python 3.12 worker runs the XphereX aperture extraction with
    TallTable and SPExPI. Existing spectra and calibration files are reused.
    refresh=True re-queries pixels and re-extracts this source. Aperture
    centering follows XphereX: one position propagated to epoch 2025.5.
    Exposures with fatal aperture flags or incomplete unmasked area are retained
    in the raw exposure CSV but excluded from the combined spectrum.
    """
    if not sed.source_id:
        raise ValueError("SPHEREx downloads require a Gaia source_id")
    source_id = str(int(sed.source_id))
    if not np.isfinite(radius_arcmin) or radius_arcmin <= 0:
        raise ValueError("radius_arcmin must be positive and finite")
    root = Path(cache_dir).expanduser().resolve() / "spherex"
    directory = root / source_id
    spectrum = directory / "aperture_spectrum.csv"
    if refresh or not spectrum.exists():
        keys = ("ra", "dec", "pmra", "pmdec", "ref_epoch")
        values = [sed.metadata.get(key) for key in keys]
        if any(value is None for value in values) or not np.isfinite(values).all():
            raise ValueError("SPHEREx downloads need finite ra, dec, pmra, pmdec and ref_epoch in SED metadata")
        uv = shutil.which("uv")
        if uv is None:
            try:
                import uv as uv_package
            except ImportError as exc:
                raise ImportError("install the optional downloader: pip install 'sedkit[spherex]'") from exc
            uv = uv_package.find_uv_bin()
        worker = Path(__file__).with_name("_spherex_aperture.py")
        command = [str(uv), "run", "--python", "3.12", str(worker),
                   "--outdir", str(directory), "--cache-root", str(root),
                   "--radius", str(radius_arcmin)]
        for key, value in zip(keys, values):
            command.extend(["--" + key.replace("_", "-"), str(value)])
        if refresh:
            command.append("--refresh")
        subprocess.run(command, check=True)
    result = load_spherex(sed, spectrum, method="aperture")
    provenance = directory / "metadata.json"
    if provenance.exists():
        result.metadata["spherex"].update(json.loads(provenance.read_text()))
    if not result.mask[66:].any():
        raise RuntimeError("no usable SPHEREx channels in the downloaded spectrum")
    result.save(directory / "sed.npz")
    return result
