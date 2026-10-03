import json

import numpy as np
import pandas as pd

from grelu.interpret.ism.profiles import MutationProfileWriter
from grelu.interpret.ism.shards import finalize_sharded_run


def _mutation(gene, index):
    return {
        "gene": gene,
        "mutation_id": f"{gene}_m{index}",
        "edit_start": 100 + index,
        "edit_end": 101 + index,
        "edit_center_position": 100 + index,
        "variant_offset_from_tss_transcription_bp": index,
        "ref_sequence": "A",
        "alt_sequence": "C",
        "replacement_replicate": 0,
    }


def _write_shard_header(path, checkpoint_path="fake.ckpt"):
    path.mkdir()
    pd.DataFrame(
        [{"track_id": "hsc", "track_group": "hsc_finetuned_10x"}]
    ).to_csv(path / "track_manifest.tsv", sep="\t", index=False)
    metadata = {
        "model_id": "fake",
        "checkpoint_path": checkpoint_path,
        "weights_path": "",
        "checkpoint_sha256": "abc",
        "prepared_manifest_sha256": "def",
        "input_length_bp": 16,
        "output_resolution_bp": 4,
        "output_length_bins": 4,
    }
    (path / "model_metadata.json").write_text(json.dumps(metadata) + "\n")


def _write_gene_shard(path, mutation):
    gene = mutation["gene"]
    (path / "features").mkdir(exist_ok=True)
    features = pd.DataFrame(
        [
            {
                "model_backend": "fake_backend",
                "model_id": "fake",
                "mutation_id": mutation["mutation_id"],
                "track_id": "hsc",
                "readout_id": f"{gene}_readout",
                "output_start": 92,
                "ref_mean": 1.0,
                "alt_mean": 2.0,
                "ref_sum": 4.0,
                "alt_sum": 8.0,
                "signed_delta_mean": 1.0,
                "signed_delta_sum": 4.0,
                "log2fc_mean": 1.0,
                "log2fc_ratio_of_sums": 1.0,
            }
        ]
    )
    features.to_csv(path / f"features/{gene}.tsv", sep="\t", index=False)
    writer = MutationProfileWriter(
        path / "profiles",
        gene=gene,
        mutations=pd.DataFrame([mutation]),
        track_order=["hsc"],
        reference_profiles=np.ones((1, 4), dtype=np.float32),
        native_resolution_bp=4,
        stored_resolution_bp=8,
        dtype="float16",
        pseudocount=1.0,
        output_start=92,
    )
    writer.write_batch(0, np.full((1, 1, 4), 2.0, dtype=np.float32))
    writer.finalize()


def test_finalize_shards_combines_complete_genes(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    genes = pd.DataFrame(
        [
            {"gene": "GeneA", "chrom": "chr1", "analysis_tss": 100},
            {"gene": "GeneB", "chrom": "chr2", "analysis_tss": 200},
        ]
    )
    genes.to_csv(prepared / "genes.tsv", sep="\t", index=False)
    pd.DataFrame(
        [{"locus_id": "locus_a"}, {"locus_id": "locus_b"}]
    ).to_csv(prepared / "loci.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"gene": "GeneA", "readout_id": "GeneA_readout"},
            {"gene": "GeneB", "readout_id": "GeneB_readout"},
        ]
    ).to_csv(prepared / "readouts.tsv", sep="\t", index=False)
    mutations = pd.DataFrame([_mutation("GeneA", 0), _mutation("GeneB", 1)])
    mutations.to_csv(prepared / "mutation_manifest.tsv", sep="\t", index=False)

    sources = [tmp_path / "shard_a", tmp_path / "shard_b"]
    for source in sources:
        _write_shard_header(source)
    _write_gene_shard(sources[0], mutations.iloc[0].to_dict())
    _write_gene_shard(sources[1], mutations.iloc[1].to_dict())

    target = tmp_path / "combined"
    validation = finalize_sharded_run(
        prepared=prepared,
        target=target,
        sources=sources,
        profile_resolution_bp=8,
    )
    assert validation["status"] == "ok"
    assert validation["feature_rows"] == 2
    assert validation["stored_profile_nonfinite_values"] == 0
    assert set(validation["gene_sources"]) == {"GeneA", "GeneB"}
    combined = pd.read_csv(
        target / "features/combined_mutation_features.tsv", sep="\t"
    )
    assert combined.mutation_id.tolist() == ["GeneA_m0", "GeneB_m1"]


def test_finalize_shards_accepts_equivalent_checkpoint_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "runs" / "fake.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    source_a = tmp_path / "shard_a"
    source_b = tmp_path / "shard_b"
    _write_shard_header(source_a, str(checkpoint))
    _write_shard_header(source_b, "runs/fake.ckpt")

    from grelu.interpret.ism.shards import _validate_shard_provenance

    _, metadata = _validate_shard_provenance([source_a, source_b])
    assert metadata["checkpoint_path"] == str(checkpoint)
