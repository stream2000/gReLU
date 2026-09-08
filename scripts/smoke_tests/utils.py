import datetime

import numpy as np
import pandas as pd


def _write_ism_report(
    df: pd.DataFrame,
    model_name: str,
    gene: str,
    devices: str,
    out_path: str,
) -> None:
    """Write an ISM summary text report from a completed ISM result dataframe.

    Pure I/O function — takes the result matrix directly so it can be called
    from ``run_ism()`` without re-reading from disk.

    Args:
        df:         ISM result matrix (rows=bases A/C/G/T, cols=positions,
                    values=log2FC).
        model_name: Model identifier used in the report header.
        gene:       Gene name displayed in the report.
        devices:    Device specification string (for provenance).
        out_path:   Path where the .txt report will be written.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vals = df.values.astype(float)
    col_max = np.max(np.abs(vals), axis=0)
    top_n = min(10, len(df.columns))
    top_idx = np.argsort(col_max)[::-1][:top_n]
    with open(out_path, "w") as f:
        f.write(f"ISM Results Report — {model_name.upper()}\n")
        f.write("=" * 68 + "\n")
        f.write(f"Run Time       : {ts}\n")
        f.write(f"Gene           : {gene}\n")
        f.write(f"GPU Devices    : {devices}\n")
        f.write(f"Matrix Shape   : {df.shape[0]} rows × {df.shape[1]} columns"
                f"  [4 bases × positions]\n\n")
        f.write("── Numerical Summary ──────────────────────────────────────────────\n")
        f.write(f"  Max log2FC   : {vals.max():+.6f}\n")
        f.write(f"  Min log2FC   : {vals.min():+.6f}\n")
        f.write(f"  Mean |log2FC|: {np.mean(np.abs(vals)):.6f}\n")
        f.write(f"  Non-zero Ratio: {np.mean(vals != 0) * 100:.1f}%\n\n")
        f.write("── Top 10 Positions by Signal Strength ──────────────────────────────────────\n")
        f.write(f"  {'Position':>12}  {'max|log2FC|':>12}  Best Mutation Base\n")
        for i in top_idx:
            col = df.columns[i]
            best = df[col].abs().idxmax()
            f.write(f"  {col:>12}  {col_max[i]:>12.6f}  {best}\n")
        f.write("\n── Statistics per Mutation Base ────────────────────────────────────────\n")
        f.write(f"  {'Base':>4}  {'Max':>10}  {'Min':>10}  {'Mean':>10}  Std Dev\n")
        for base in ["A", "C", "G", "T"]:
            r = df.loc[base].values.astype(float)
            f.write(f"  {base:>4}  {r.max():>+10.6f}  {r.min():>+10.6f}"
                    f"  {r.mean():>+10.6f}  {r.std():.6f}\n")
        f.write("\n── Full Matrix (TSV, rows=bases, cols=positions, values=log2FC) ─────\n")
        f.write(df.to_csv(sep="\t"))
    print(f"  Text report → {out_path}")


def _parse_variant(v: str) -> tuple[str, int, str, str]:
    """Parse 'chr1_70355119_G_A' into (chrom, pos, ref, alt)."""
    parts = v.split("_")
    return parts[0], int(parts[1]), parts[2], parts[3]


def _stats(r_arr: np.ndarray) -> dict:
    """Summary statistics over a per-window Pearson R array; ignores NaN/Inf."""
    v = r_arr[np.isfinite(r_arr)]
    return {
        "mean_r":    float(np.mean(v))   if len(v) else float("nan"),
        "median_r":  float(np.median(v)) if len(v) else float("nan"),
        "std_r":     float(np.std(v))    if len(v) else float("nan"),
        "n_windows": int(len(v)),
    }


def _bin_obs(obs_raw: np.ndarray, bin_size: int, n_pred_bins: int) -> np.ndarray:
    """Bin base-resolution observations then centre-crop to match model output length.

    Args:
        obs_raw:     (N, seq_len) base-resolution signal array.
        bin_size:    Number of bases per output bin.
        n_pred_bins: Number of bins in the model prediction (crop target).

    Returns:
        (N, n_pred_bins) binned and cropped array.
    """
    n, total = obs_raw.shape
    obs = obs_raw.reshape(n, total // bin_size, bin_size).mean(axis=-1)
    n_obs_bins = obs.shape[1]
    if n_pred_bins != n_obs_bins:
        crop_start = (n_obs_bins - n_pred_bins) // 2
        obs = obs[:, crop_start : crop_start + n_pred_bins]
    return obs
