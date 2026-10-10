"""DA + dwarf fitting and a conditional WD-light/orbit envelope.

This example constructs synthetic XP data. It demonstrates API conventions;
real-data validation is in docs/validation-whitedwarf.md.
"""

import numpy as np
from sedkit import SED, StellarModel, WhiteDwarfModel, fit_whitedwarf_companion, whitedwarf_light_limit
from sedkit.orbit import photocentre_a0, solve_dark_companion

wd = WhiteDwarfModel()
stellar = StellarModel()
mask = np.arange(168) < 61
white = wd.predict(18000, .6)["flux"]
red = stellar.evaluate(.35, 0., 5., 0.)["flux_10pc"]
flux = .01 * (white + red)  # parallax 10 mas: (10 pc / 100 pc)^2
sed = SED(flux, .02 * flux, mask, 10., .02, "synthetic_wd_m")
fit = fit_whitedwarf_companion(sed)
print("Composite labels:", fit["hypotheses"]["wd+dwarf"]["whitedwarf"])
print("WD total G-light fraction:", fit["hypotheses"]["wd+dwarf"]["fractions"]["beta_G"])

primary = stellar.evaluate(.9, 0., 5., 0.)["flux_10pc"]
sed = SED(.01 * primary, .0002 * primary, mask, 10., .02, "synthetic_primary")
limit = whitedwarf_light_limit(sed, masses=[.6],
    temperatures=[6000, 10000, 15000, 20000, 30000, 50000, 80000], luminous_mass=.6)
a0 = photocentre_a0(.9, .6, 1., 500., 10.)
zero = solve_dark_companion(a0, 10., 500., .9)
light = solve_dark_companion(a0, 10., 500., .9, beta=limit["flux_ratio_G_upper"])
print("Conditional beta_G envelope:", limit["beta_G_upper"])
print("Orbit mass at zero light / allowed light:", zero["m2"], light["m2"])
print("Specified coeval MS hypothesis delta:", limit["luminous_companion"]["delta"])
