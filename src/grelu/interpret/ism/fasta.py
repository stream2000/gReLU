"""Small FASTA access wrapper for ISM workflows."""

from __future__ import annotations

from pathlib import Path


class FastaReference:
    """Reference genome accessor backed by pyfaidx."""

    def __init__(self, fasta_path: str | Path):
        try:
            from pyfaidx import Fasta
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ImportError("pyfaidx is required for FASTA-backed ISM") from exc

        self.path = Path(fasta_path)
        self._fasta = Fasta(
            str(self.path),
            as_raw=True,
            sequence_always_upper=True,
            rebuild=False,
        )

    def extract(self, chrom: str, start: int, end: int) -> str:
        if start < 0:
            raise ValueError(f"Negative FASTA interval start: {chrom}:{start}-{end}")
        seq = str(self._fasta[str(chrom)][int(start) : int(end)]).upper()
        expected = int(end) - int(start)
        if len(seq) != expected:
            raise ValueError(
                f"Expected {expected} bp for {chrom}:{start}-{end}, got {len(seq)}"
            )
        return seq

    def chrom_length(self, chrom: str) -> int:
        return len(self._fasta[str(chrom)])

    def close(self) -> None:
        self._fasta.close()

    def __enter__(self) -> "FastaReference":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
