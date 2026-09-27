# Group meeting: three observed-data notebooks

A 20-minute introduction to sedkit, from public-data acquisition to composition
with orblet. The notebooks use English text, small code cells and saved figures.
Default execution reads included observed snapshots; live downloads are optional.

| Notebook | Question | Time |
| --- | --- | --- |
| [01: observed SB2](../examples/group_meeting/01_observed_sb2.ipynb) | How do we download and compare single/binary SED fits? | 6 min |
| [02: observed SPHEREx](../examples/group_meeting/02_observed_spherex.ipynb) | What changes when measured infrared channels are added? | 6 min |
| [03: orblet joint fit](../examples/group_meeting/03_orblet_joint_fit.ipynb) | How do real XP and SB2 RVs share a physical binary model? | 8 min |

## Run

Use Python 3.11 or 3.12 for all three notebooks. From a sedkit checkout:

```bash
pip install -e ".[download,notebook]"
pip install "orblet @ git+https://github.com/saharsh1/orblet.git@7aa297df4520d1e93c8a7a5a763a1f2d928bfa51"
jupyter lab examples/group_meeting
```

Select the kernel for that environment. Each notebook runs independently from
top to bottom; no variables or output files from an earlier notebook are required.
Live SPHEREx acquisition additionally needs `pip install -e ".[spherex]"`.
Leave `LIVE_DOWNLOAD=False` for the presentation; the first online extraction
can take several minutes. The optional exercise in notebook 01 is also off.

The checked-in notebooks include executed outputs. Matching `.py` scripts
contain their code cells and generate the same figures. New figures and the
HD 195987 fit are written under `examples/group_meeting/outputs/`, keeping
original observation snapshots and previous experiment results unchanged.

## Speaking sequence

01: begin with the observed SED, then show the single/binary comparison.
Explain absolute flux and the physical component sum before showing q_RV.
Binary preference is a diagnostic, not a binary probability.

02: begin with the exposure-selection histogram, then attach the measured CSV.
The additional channels move q away from the RV value in this example.
Show that discrepancy and the extraction limitations, without correcting fluxes.

03: show both observed RV curves and their residuals. Run the conditional
joint fit, expose the three likelihood calls and the distance constraint,
then show the photocentre response. The response curve is a model calculation;
there is no observed Gaia epoch-astrometry term in this example.

## Scientific boundaries

The SB2 and SPHEREx fits reach an age boundary. HD 195987 has excess secondary
RV scatter; its joint fit fixes orbital shape and has no posterior mass errors.
Its distance constraint and comparison masses share a historical interferometric
analysis, so their agreement is not fully independent. The observed residuals
and these limits are visible in the notebooks.

Data provenance: [SB2 observations](sb2.md), [SPHEREx extraction](spherex.md),
[HD 195987](real-orblet.md). Interface details: [API](api.md), [orblet](orblet.md).
