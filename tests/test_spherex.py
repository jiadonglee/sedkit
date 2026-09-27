import json
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from sedkit import SED, StellarModel, download_spherex, load_spherex
import sedkit.spherex as acquisition


def observation():
    return SED(np.ones(168), np.full(168, .1), np.ones(168, bool), 20,
               source_id='858860697467058688',
               metadata=dict(ra=176.077419, dec=60.3216, pmra=-10.02,
                             pmdec=-109.85, ref_epoch=2016.0))


def write_csv(path, flux=None, error=None, reverse=False):
    wave = StellarModel().wavelength_um[66:]
    flux = np.ones(102) if flux is None else flux
    error = np.full(102, .1) if error is None else error
    values = np.column_stack([wave, flux, error])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values[::-1] if reverse else values, delimiter=',',
               header='wavelength_um,flux_jy,error_jy', comments='')


def test_import_units_mask_and_unchanged_observations(tmp_path):
    sed = observation()
    flux, error = np.ones(102), np.full(102, .1)
    flux[0], error[1] = -.5, 0
    path = tmp_path / 'psf_spectrum.csv'
    write_csv(path, flux, error, reverse=True)
    result = load_spherex(sed, path)
    expected = (flux * u.Jy).to_value(u.W/u.m**2/u.nm,
        equivalencies=u.spectral_density(sed.wavelength_um[66:].astype(float) * u.um)) * 1e18
    np.testing.assert_allclose(result.flux[66:], expected)
    np.testing.assert_array_equal(result.flux[:66], sed.flux[:66])
    np.testing.assert_array_equal(result.error[:66], sed.error[:66])
    np.testing.assert_array_equal(sed.flux, np.ones(168))
    assert result.flux[66] < 0 and result.mask[66] and not result.mask[67]
    assert result.metadata['spherex']['method'] == 'psf'
    assert 'spherex' not in sed.metadata


def test_import_rejects_different_channel_grid(tmp_path):
    path = tmp_path / 'aperture_spectrum.csv'
    write_csv(path)
    values = np.loadtxt(path, delimiter=',', skiprows=1)
    values[0, 0] += .005
    np.savetxt(path, values, delimiter=',',
               header='wavelength_um,flux_jy,error_jy', comments='')
    with pytest.raises(ValueError, match='wavelengths must match'):
        load_spherex(observation(), path)


def test_download_passes_astrometry_and_reuses_spectrum(tmp_path, monkeypatch):
    commands = []
    def run(command, check):
        assert check
        commands.append(command)
        directory = Path(command[command.index('--outdir') + 1])
        write_csv(directory / 'aperture_spectrum.csv')
        (directory / 'metadata.json').write_text(json.dumps({'data_release': 'qr2'}))
    monkeypatch.setattr(acquisition.shutil, 'which', lambda name: '/test/uv')
    monkeypatch.setattr(acquisition.subprocess, 'run', run)
    sed = observation()
    first = download_spherex(sed, cache_dir=tmp_path)
    cached = download_spherex(sed, cache_dir=tmp_path)
    assert len(commands) == 1
    for name, expected in (('--pmra', '-10.02'), ('--pmdec', '-109.85'), ('--ref-epoch', '2016.0')):
        assert commands[0][commands[0].index(name) + 1] == expected
    np.testing.assert_array_equal(first.flux, cached.flux)
    assert first.metadata['spherex']['data_release'] == 'qr2'
    download_spherex(sed, cache_dir=tmp_path, refresh=True)
    assert len(commands) == 2 and '--refresh' in commands[1]


def test_import_keeps_missing_channels_masked(tmp_path):
    path = tmp_path / 'aperture_spectrum.csv'
    write_csv(path)
    values = np.loadtxt(path, delimiter=',', skiprows=1)[::3]
    np.savetxt(path, values, delimiter=',',
               header='wavelength_um,flux_jy,error_jy', comments='')
    result = load_spherex(observation(), path)
    assert result.mask[66:].sum() == len(values)
    assert np.isnan(result.flux[67])
    assert np.isnan(result.error[67])
