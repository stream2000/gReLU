"""Frozen NTv3-8M final embeddings for the fixed Saijou profile geometry.

Remote model code is loaded only on construction, from an immutable revision.
The projection-input contract must still be qualified on the real checkpoint.
"""

import re

import torch
from torch import nn
from torch.nn import functional as F


CHECKPOINT = "InstaDeepAI/NTv3_8M_pre"
CODE_REVISION = "0ecff3637f0d3ba5b686d1095083218157c2ca34"
SEQ_LEN = 524_288
LABEL_LEN = 196_608
BIN_SIZE = 32


def _omit_rotary_runtime_cache(module, state_dict, prefix, local_metadata):
    # Official remote code persists lazily-created cos/sin caches. They are
    # deterministic runtime state, not pretrained weights, and fresh models
    # have None buffers (otherwise Lightning strict reload rejects these keys).
    for key in list(state_dict):
        if key.startswith(prefix) and key.endswith((".cos_cached", ".sin_cached")):
            del state_dict[key]


def _clear_rotary_runtime_cache(module, state_dict, prefix, local_metadata,
                               strict, missing_keys, unexpected_keys, error_msgs):
    # Trainer resume may load into an already warmed-up model. Reset these
    # derived buffers to the same None state as a freshly constructed model.
    for child in module.modules():
        for name in ("cos_cached", "sin_cached"):
            if name in child._buffers:
                child._buffers[name] = None


def load_ntv3(checkpoint, revision, use_bfloat16_compute):
    """Load the pinned official tokenizer/model; never change the environment."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    kwargs = dict(revision=revision, code_revision=CODE_REVISION,
                  trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
    compute = {}
    if use_bfloat16_compute:
        compute = {name + "_compute_dtype": "bfloat16" for name in (
            "stem", "down_convolution", "transformer_qkvo", "transformer_ffn",
            "up_convolution", "modulation",
        )}
    model = AutoModelForMaskedLM.from_pretrained(checkpoint, **kwargs, **compute)
    return tokenizer, model


class NTv3FeatureTrunk(nn.Module):
    """Map (B,4,524288) DNA to frozen (B,256,6144) pooled features."""

    def __init__(self, checkpoint=CHECKPOINT, revision=None, seq_len=SEQ_LEN,
                 label_len=LABEL_LEN, bin_size=BIN_SIZE, use_bfloat16_compute=True):
        super().__init__()
        if checkpoint != CHECKPOINT:
            raise ValueError("The MVP supports only NTv3_8M_pre")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be an immutable 40-character commit SHA")
        if (seq_len, label_len, bin_size) != (SEQ_LEN, LABEL_LEN, BIN_SIZE):
            raise ValueError("The MVP requires 524288 -> 196608 -> 6144 x 32 geometry")
        self.seq_len = seq_len
        self.crop_start = (seq_len - label_len) // 2
        self.crop_end = self.crop_start + label_len
        self.out_channels = 256
        self.output_len = label_len // bin_size
        self.pool_factor = bin_size
        self.revision = revision
        tokenizer, self.ntv3 = load_ntv3(checkpoint, revision, use_bfloat16_compute)
        ids = [tokenizer.convert_tokens_to_ids(base) for base in "ACGTN"]
        if (any(not isinstance(i, int) or not 0 <= i < 11 for i in ids)
                or len(set(ids)) != 5
                or any(i in tokenizer.all_special_ids for i in ids)):
            raise ValueError("Tokenizer must provide distinct non-special A/C/G/T/N IDs")
        encoded = tokenizer("ACGTN", add_special_tokens=False)["input_ids"]
        if encoded != ids:
            raise ValueError("Tokenizer is not a position-preserving character tokenizer")
        self.register_buffer("base_ids", torch.tensor(ids[:4], dtype=torch.long))
        self.n_id = ids[4]
        projection = self.ntv3.get_output_embeddings()
        if not isinstance(projection, nn.Linear) or (projection.in_features, projection.out_features) != (256, 11):
            raise ValueError("Expected final 256 -> 11 MLM projection; qualify remote adapter")
        # The MLM projection receives GELU(final_embedding), not final_embedding.
        # Request only the last post-skip deconv tensor through the official core API.
        self.ntv3.config.deconv_layers_to_save = (7,)
        self.ntv3.config.embeddings_layers_to_save = ()
        self.ntv3.config.attention_maps_to_save = []
        self.ntv3.requires_grad_(False)
        self.register_state_dict_post_hook(_omit_rotary_runtime_cache)
        self.register_load_state_dict_pre_hook(_clear_rotary_runtime_cache)
        self.train(False)

    def train(self, mode=True):
        # Keep the entire frozen trunk in eval even when BaseModel.train() recurses.
        super().train(False)
        return self

    def one_hot_to_input_ids(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (4, self.seq_len):
            raise ValueError("Expected DNA shape (B,4,524288)")
        if not torch.isfinite(x).all() or not ((x == 0) | (x == 1)).all():
            raise ValueError("Invalid one-hot DNA: require finite binary entries")
        total = x.sum(dim=1)
        if not ((total == 0) | (total == 1)).all():
            raise ValueError("Invalid one-hot DNA: multiple active bases")
        ids = self.base_ids[x.argmax(dim=1)]
        return torch.where(total == 0, self.n_id, ids)

    @torch.no_grad()
    def extract_final_embedding(self, input_ids):
        self.ntv3.eval()
        outputs = self.ntv3.core(input_ids=input_ids, output_hidden_states=False,
                                 output_attentions=False)
        embedding = outputs["embeddings_deconv_7"].transpose(1, 2)
        expected = (*input_ids.shape, self.out_channels)
        if tuple(embedding.shape) != expected:
            raise ValueError("Final deconv embedding does not match (B,L,256)")
        return embedding

    @torch.no_grad()
    def forward(self, x):
        emb = self.extract_final_embedding(self.one_hot_to_input_ids(x))
        emb = emb[:, self.crop_start:self.crop_end, :].transpose(1, 2)
        emb = F.avg_pool1d(emb, self.pool_factor, self.pool_factor)
        if tuple(emb.shape[1:]) != (self.out_channels, self.output_len):
            raise ValueError("Unexpected NTv3 feature shape")
        if not torch.isfinite(emb).all():
            raise FloatingPointError("Nonfinite NTv3 features")
        # Head parameters stay fp32; Lightning autocast owns head mixed precision.
        return emb.detach().float()
