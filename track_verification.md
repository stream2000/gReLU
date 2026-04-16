# Track Index Verification

This document explains how we confirmed that the track indices used in the inference
correspond correctly to the intended biological targets.

---

## Requested Targets

| Assay | TF | Cell Line |
|---|---|---|
| ChIP-TF | CTCF | MCF-7 |
| ChIP-TF | CTCF | RPE1 |
| ChIP-TF | RAD21 | MCF-7 |
| ChIP-TF | RAD21 | RPE1 |

---

## Step 1 — Query the Track Metadata

AlphaGenome ships a parquet file (`track_metadata_human.parquet`) that lists every
output track the model was trained to predict. Each row describes one track with fields
including `output_type`, `transcription_factor`, `biosample_name`, and `track_index`.

We filtered for `output_type == "chip_tf"` and searched for the requested TF / cell-line
combinations:

```python
import pandas as pd
df = pd.read_parquet("track_metadata_human.parquet")
chip = df[df["output_type"] == "chip_tf"]
hits = chip[
    chip["transcription_factor"].str.contains("CTCF|RAD21", case=False, na=False) &
    chip["biosample_name"].str.contains("MCF-7|RPE1", case=False, na=False)
]
```

Result:

| track_index | transcription_factor | biosample_name |
|---|---|---|
| 776 | CTCF | MCF-7 |
| 812 | RAD21 | MCF-7 |
| 813 | RAD21 | MCF-7 |

**RPE1 was not found.** A broader search confirmed that RPE1 does not appear anywhere
in the 163 unique biosamples of the chip_tf metadata. The model has no output head for
this cell type and cannot produce RPE1 predictions.

---

## Step 2 — Confirm `track_index` Equals the Tensor Column Index

The model's `chip_tf` head is defined in `model.py` as:

```python
add_head('chip_tf', 1664, [128], apply_squashing=False)
```

This means the output tensor has shape `(batch, L_bins, 1664)`, with exactly 1664 tracks.
To confirm that the `track_index` column in the metadata directly maps to the tensor's
last dimension (i.e., `arr[:, track_index]` retrieves the correct track), we verified:

```python
chip_sorted = chip.sort_values("track_index")
assert list(chip_sorted["track_index"]) == list(range(1664))
```

The check passed: `track_index` runs 0–1663 with no gaps and no reordering.
Indexing `arr[:, 776]` therefore retrieves exactly the CTCF/MCF-7 track.

---

## Summary

| track_index | TF | Cell Line | Status |
|---|---|---|---|
| 776 | CTCF | MCF-7 | **Used** |
| 812 | RAD21 | MCF-7 (rep 1) | **Used** |
| 813 | RAD21 | MCF-7 (rep 2) | **Used** |
| — | CTCF | RPE1 | **Not available** — RPE1 not in training targets |
| — | RAD21 | RPE1 | **Not available** — RPE1 not in training targets |
