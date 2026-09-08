"""Small synthetic checks for NTv3 training/evaluation orchestration."""

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def entrypoints(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "src" / "ft-scripts"
    monkeypatch.syspath_prepend(str(folder))
    import train_ntv3
    import finish_ntv3
    return train_ntv3, finish_ntv3


def test_track_means(entrypoints):
    train, _ = entrypoints
    labels = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    np.testing.assert_allclose(train.track_means(labels), labels.mean(axis=(0, 2)))
    labels[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        train.track_means(labels)


def test_rotary_cache_is_not_checkpoint_state(monkeypatch):
    from grelu.model.trunks.ntv3 import _omit_rotary_runtime_cache, _clear_rotary_runtime_cache
    module = torch.nn.Module()
    child = torch.nn.Module()
    child.register_buffer("cos_cached", torch.ones(2))
    child.register_buffer("sin_cached", torch.zeros(2))
    child.register_buffer("inv_freq", torch.ones(1))
    module.add_module("rotary", child)
    module.register_state_dict_post_hook(_omit_rotary_runtime_cache)
    module.register_load_state_dict_pre_hook(_clear_rotary_runtime_cache)
    assert list(module.state_dict()) == ["rotary.inv_freq"]
    module.load_state_dict(module.state_dict(), strict=True)
    assert child.cos_cached is None and child.sin_cached is None


def test_evaluation_constant_prediction(entrypoints):
    _, finish = entrypoints

    class ConstantModel(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, x, logits=False):
            return torch.zeros_like(x)

    x = torch.zeros(3, 4, 32)
    target = torch.arange(3 * 4 * 32).reshape(3, 4, 32).float() / 100
    rows = finish.evaluate(ConstantModel(), torch.utils.data.TensorDataset(x, target),
                           np.ones(4), 17, "val")
    assert len(rows) == 8
    for task in finish.TASK_NAMES:
        learned, bias = [r for r in rows if r["task"] == task]
        assert learned["loss"] == pytest.approx(bias["loss"])
        assert learned["mse"] == pytest.approx(bias["mse"])
        assert np.isnan(bias["pearson"])


def test_metric_keys(entrypoints):
    _, finish = entrypoints
    rows = [dict(seed=seed, split=split, predictor=predictor, task=task,
                 loss=1., mse=1., log1p_mse=1., pearson=float("nan"), log1p_pearson=0.)
            for seed in (17,29,43) for split in ("val", "test")
            for predictor in ("ntv3", "bias_only") for task in finish.TASK_NAMES]
    finish.validate_metric_rows(rows)
    with pytest.raises(ValueError, match="keys"):
        finish.validate_metric_rows(rows + rows[:1])
    rows[0]["loss"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        finish.validate_metric_rows(rows)


def test_single_seed_metric_keys(entrypoints):
    _, finish = entrypoints
    rows = [dict(seed=43, split=split, predictor=predictor, task=task,
                 loss=1., mse=1., log1p_mse=1., pearson=0., log1p_pearson=0.)
            for split in ("val", "test") for predictor in ("ntv3", "bias_only")
            for task in finish.TASK_NAMES]
    finish.validate_metric_rows(rows, (43,))


def test_validation_batch_is_independent(entrypoints, monkeypatch):
    train, _ = entrypoints
    monkeypatch.setattr(train.LightningModel, "make_test_loader",
                        lambda self, dataset, batch_size, num_workers: batch_size)
    assert train.make_validation_loader(object(), object(), batch_size=8) == 1
