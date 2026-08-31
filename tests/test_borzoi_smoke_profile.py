import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "src" / "ft-scripts" / "make_borzoi_smoke_profile.py"


def test_smoke_profile_writes_the_approved_intervals(tmp_path):
    chrom_sizes = tmp_path / "mm10.sizes"
    chrom_sizes.write_text("chr1\t2000000\nchr10\t2000000\nchr11\t2000000\n")
    out_dir = tmp_path / "example_smoke_v1"

    subprocess.run(
        [sys.executable, str(SCRIPT), "--out_dir", str(out_dir), "--chrom_sizes", str(chrom_sizes)],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["assembly"] == "mm10"
    assert manifest["coordinate_convention"] == "BED 0-based, half-open"
    assert manifest["input_window_bp"] == 524_288
    assert manifest["label_window_bp"] == 196_608
    assert manifest["intervals"]["train"] == [
        {
            "chrom": "chr1",
            "input_start": 32_768,
            "input_end": 557_056,
            "label_start": 196_608,
            "label_end": 393_216,
        }
    ]
    assert (out_dir / "val_intervals.bed").read_text() == "chr10\t32768\t557056\n"
