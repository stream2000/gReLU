import pandas as pd
import pytest

from grelu.interpret.ism.validation import (
    expected_feature_rows,
    feature_table_diagnostics,
    validate_feature_table,
)


def feature_rows():
    return pd.DataFrame(
        [
            {
                "model_backend": "model",
                "mutation_id": "m0",
                "track_id": "track",
                "readout_id": "readout",
                "ref_mean": 1.0,
                "alt_mean": 2.0,
                "ref_sum": 3.0,
                "alt_sum": 4.0,
                "signed_delta_mean": 1.0,
                "signed_delta_sum": 1.0,
                "log2fc_mean": 1.0,
                "log2fc_ratio_of_sums": 1.0,
            }
        ]
    )


def test_expected_feature_rows_uses_per_gene_manifest_geometry():
    genes = pd.DataFrame({"gene": ["A", "B"]})
    mutations = pd.DataFrame({"gene": ["A", "A", "B"]})
    readouts = pd.DataFrame({"gene": ["A", "B", "B"]})
    assert expected_feature_rows(genes, readouts, mutations, n_tracks=4) == 16


def test_feature_table_validation_reports_duplicate_keys():
    features = pd.concat([feature_rows(), feature_rows()], ignore_index=True)
    diagnostics = feature_table_diagnostics(features, expected_rows=2)
    assert diagnostics.duplicate_feature_keys == 1
    with pytest.raises(RuntimeError, match="duplicate_feature_keys"):
        validate_feature_table(features, expected_rows=2)
