# MREG Plot Scripts

This folder contains presentation and diagnostic plotting scripts for MREG ISM
outputs. Keep model-running, manifest-preparation and analysis scripts one level
up in `scripts/ism/examples/mreg/`.

## Scripts

```text
plot_mreg_three_region_chromatin.py
  Original three-region figure bundle: locus schematic, REF/ALT/delta panels,
  DIC-to-TSS effects, response matrices and adjusted summaries.

plot_mreg_raw_track_profiles_pdf.py
  Diagnostic multi-page PDF. One page per track, with REF/ALT/delta panels
  across mutation and readout windows.

plot_mreg_browser_style_tracks.py
  Genome-browser-style stacked REF/ALT track plots for MCF-7 exact-track runs.
  This is the preferred presentation style when discussing raw predicted
  track-shape changes. It writes one combined PDF: three local mutation-window
  pages followed by two DIC-to-MREG-TSS readout pages.

plot_mreg_ctcf_multires_topology.py
  Legacy topology plots for the old MREG CTCF multi-resolution MVP.

plot_mreg_ctcf_contact_multiscale.py
  Legacy contact-map visualization for the old MREG CTCF contact analysis.
```

## Current MCF-7 Browser-Style Command

```bash
cd /home/fqijun/python/gReLU
source activate.sh

python scripts/ism/examples/mreg/plot/plot_mreg_browser_style_tracks.py \
  --profiles agent-doc/ism_context/20260618_2019_mreg_mcf7_exact_track_profiles/run/mreg_chromatin_profiles.parquet \
  --resolved-tracks agent-doc/ism_context/20260618_2019_mreg_mcf7_exact_track_profiles/run/resolved_tracks.tsv \
  --intervals agent-doc/ism_context/20260618_2019_mreg_mcf7_exact_track_profiles/run/scored_intervals.tsv \
  --output-dir agent-doc/ism_context/20260618_2019_mreg_mcf7_exact_track_profiles/figures \
  --biosample MCF-7
```
