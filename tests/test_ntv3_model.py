"""Download-free tests; fake MLM checks contracts, not official compatibility."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from grelu.model.trunks import ntv3
from grelu.model.models import NTv3PretrainedProfileModel


REVISION = "a" * 40


class FakeTokenizer:
    all_special_ids = [0, 1, 2, 3, 4, 5]

    def convert_tokens_to_ids(self, base):
        return dict(A=6, C=8, G=9, T=7, N=10)[base]

    def __call__(self, seq, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [self.convert_tokens_to_ids(base) for base in seq]}


class FakeMLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(256, 11)
        self.fail = False
        self.config = SimpleNamespace()

    def get_output_embeddings(self):
        return self.projection

    def core(self, input_ids, **kwargs):
        assert kwargs == dict(output_hidden_states=False, output_attentions=False)
        assert not self.training and not torch.is_grad_enabled()
        if self.fail:
            raise RuntimeError("injected failure")
        # Expanded view avoids materializing 524288 x 256 fake hidden states.
        emb = input_ids.float().unsqueeze(-1).expand(-1, -1, 256)
        return {"embeddings_deconv_7": emb.transpose(1, 2)}


@pytest.fixture
def fake_loader(monkeypatch):
    monkeypatch.setattr(ntv3, "load_ntv3", lambda *args: (FakeTokenizer(), FakeMLM()))


@pytest.fixture
def trunk(fake_loader):
    return ntv3.NTv3FeatureTrunk(revision=REVISION)


def test_token_mapping(trunk):
    x = torch.zeros(1, 4, ntv3.SEQ_LEN)
    for base in range(4):
        x[0, base, base] = 1
    ids = trunk.one_hot_to_input_ids(x)
    assert ids.dtype == torch.long
    assert ids.shape == (1, ntv3.SEQ_LEN)
    assert ids[0, :5].tolist() == [6, 8, 9, 7, 10]


@pytest.mark.parametrize("values", [[.5, .5, 0, 0], [2, -1, 0, 0],
                                    [1, 1, 0, 0], [float("nan"), 0, 0, 0]])
def test_invalid_one_hot(trunk, values):
    x = torch.zeros(1, 4, ntv3.SEQ_LEN)
    x[0, :, 0] = torch.tensor(values)
    with pytest.raises(ValueError, match="Invalid one-hot"):
        trunk.one_hot_to_input_ids(x)


def test_fixed_geometry_and_revision(fake_loader):
    with pytest.raises(ValueError, match="immutable"):
        ntv3.NTv3FeatureTrunk(revision="main")
    with pytest.raises(ValueError, match="geometry"):
        ntv3.NTv3FeatureTrunk(revision=REVISION, seq_len=131072)
    with pytest.raises(ValueError, match="four tasks"):
        NTv3PretrainedProfileModel(n_tasks=3, revision=REVISION)


def test_projection_capture_and_cleanup(trunk):
    ids = torch.tensor([[6, 8, 9, 7, 10]])
    emb = trunk.extract_final_embedding(ids)
    assert emb.shape == (1, 5, 256)
    torch.testing.assert_close(emb[0, :, 0], ids[0].float())
    assert not trunk.ntv3.projection._forward_pre_hooks
    trunk.ntv3.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        trunk.extract_final_embedding(ids)
    assert not trunk.ntv3.projection._forward_pre_hooks


def test_crop_pool_and_backward(fake_loader, monkeypatch):
    model = NTv3PretrainedProfileModel(n_tasks=4, revision=REVISION)
    trunk = model.embedding
    assert (trunk.crop_start, trunk.crop_end) == (163840, 360448)
    positions = torch.arange(ntv3.SEQ_LEN, dtype=torch.float32)[None, :, None]
    monkeypatch.setattr(trunk, "extract_final_embedding", lambda ids: positions.expand(1, -1, 256))
    before = {k: v.clone() for k, v in trunk.state_dict().items()}
    model.train()
    assert model.head.training and not trunk.training and not trunk.ntv3.training
    features = trunk(torch.zeros(1, 4, ntv3.SEQ_LEN))
    assert features.shape == (1, 256, 6144)
    expected = torch.arange(6144) * 32 + 163840 + 15.5
    torch.testing.assert_close(features[0, 0], expected)
    logits = model.head(features)
    assert logits.shape == (1, 4, 6144)
    logits.square().mean().backward()
    assert all(p.grad is None and not p.requires_grad for p in trunk.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
    assert sum(p.numel() for p in model.head.parameters()) == 86148
    for name, value in trunk.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_lightning_checkpoint_reconstruction(fake_loader, tmp_path):
    import pytorch_lightning as pl
    from grelu.lightning import LightningModel

    params = dict(model_type="NTv3PretrainedProfileModel", n_tasks=4, revision=REVISION)
    train = dict(task="regression", loss="poisson_multinomial", total_weight=.2)
    model = LightningModel(params, train)
    checkpoint = dict(state_dict=model.state_dict(), hyper_parameters=dict(model.hparams),
                      **{"pytorch-lightning_version": pl.__version__})
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "fake.ckpt"
    torch.save(checkpoint, path)
    restored = LightningModel.load_from_checkpoint(path)
    assert restored.model_params == model.model_params
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])
    assert all(not p.requires_grad for p in restored.model.embedding.parameters())
