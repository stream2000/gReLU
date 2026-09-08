"""
`grelu.model.heads` contains 'head' layers for sequence-to-function deep
learning models. All heads inherit from the `torch.nn.Module` class, and
define a `forward` function that takes sequence embeddings produced
by earlier layers of the model (tensors of shape (N, embedding_dim, embedding_length))
and returns tensors of shape (N, tasks, output_length).
"""

from typing import List, Optional

import torch
from einops import rearrange
from torch import nn

from grelu.model.blocks import ChannelTransformBlock, LinearBlock
from grelu.model.layers import AdaptivePool


class _NTv3ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _NTv3DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation,
                      dilation=dilation, groups=channels, bias=False),
            _NTv3ChannelLayerNorm(channels), nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.layers(x)


class NTv3LocalProfileHead(nn.Module):
    """Fixed four-track log-rate probe with a 29-bin receptive field."""

    n_tasks = 4

    def __init__(self):
        super().__init__()
        self.input_norm = _NTv3ChannelLayerNorm(256)
        self.input_projection = nn.Conv1d(256, 128, 1)
        self.blocks = nn.Sequential(*[
            _NTv3DepthwiseResidualBlock(128, dilation) for dilation in (1, 2, 4)
        ])
        self.output_projection = nn.Conv1d(128, 4, 1)

    def forward(self, x):
        x = self.input_projection(self.input_norm(x))
        return self.output_projection(self.blocks(x))


class ConvHead(nn.Module):
    """
    A 1x1 Conv layer that transforms the the number of channels in the input and then
    optionally pools along the length axis.

    Args:
        n_tasks: Number of tasks (output channels)
        in_channels: Number of channels in the input
        norm: If True, batch normalization will be included.
        act_func: Activation function for the convolutional layer
        pool_func: Pooling function.
        norm: If True, batch normalization will be included.
        norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layer
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        n_tasks: int,
        in_channels: int,
        act_func: Optional[str] = None,
        pool_func: Optional[str] = None,
        norm: bool = False,
        norm_kwargs: Optional[dict] = None,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        # Save all params
        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.act_func = act_func
        self.pool_func = pool_func
        self.norm = norm

        # Create layers
        self.channel_transform = ChannelTransformBlock(
            self.in_channels,
            self.n_tasks,
            act_func=self.act_func,
            norm=self.norm,
            norm_kwargs=(norm_kwargs or dict()),
            dtype=dtype,
            device=device,
        )
        self.pool = AdaptivePool(self.pool_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input data.
        """
        x = self.channel_transform(x)
        x = self.pool(x)
        return x


class AlphaGenomeFinetuneHead(nn.Module):
    """Track head for AlphaGenome embeddings with crop and bin aggregation.

    AlphaGenome can expose either 128 bp or 1 bp sequence embeddings. This head
    crops the embedding axis to the supervised label window and optionally
    aggregates finer-resolution embeddings to the target bin size.
    """

    def __init__(
        self,
        n_tasks: int,
        in_channels: int,
        resolution: int,
        label_len: int,
        bin_size: int,
        hidden_channels: int = 512,
        hidden_layers: int = 1,
        dropout: float = 0.0,
        norm: bool = True,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        if resolution not in {1, 128}:
            raise ValueError(f"resolution must be 1 or 128, got {resolution}")
        if label_len % resolution != 0:
            raise ValueError("label_len must be divisible by resolution")
        if bin_size % resolution != 0:
            raise ValueError("bin_size must be divisible by resolution")
        if label_len % bin_size != 0:
            raise ValueError("label_len must be divisible by bin_size")

        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.resolution = resolution
        self.label_len = label_len
        self.bin_size = bin_size
        self.crop_bins = label_len // resolution
        self.pool_bins = bin_size // resolution

        layers = []
        channels = in_channels
        for _ in range(hidden_layers):
            layers.append(
                nn.Conv1d(
                    channels,
                    hidden_channels,
                    kernel_size=1,
                    dtype=dtype,
                    device=device,
                )
            )
            if norm:
                layers.append(nn.GroupNorm(1, hidden_channels, dtype=dtype, device=device))
            layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            channels = hidden_channels

        layers.append(
            nn.Conv1d(
                channels,
                n_tasks,
                kernel_size=self.pool_bins,
                stride=self.pool_bins,
                dtype=dtype,
                device=device,
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.crop_bins:
            raise ValueError(
                f"Embedding length {x.shape[-1]} is shorter than requested "
                f"label crop {self.crop_bins}"
            )
        crop_start = (x.shape[-1] - self.crop_bins) // 2
        x = x[..., crop_start : crop_start + self.crop_bins]
        return self.net(x)


class MLPHead(nn.Module):
    """
    This block implements the multi-layer perceptron (MLP) module.

    Args:
        n_tasks: Number of tasks (output channels)
        in_channels: Number of channels in the input
        in_len: Length of the input
        norm: If True, batch normalization will be included.
        act_func: Activation function for the linear layers
        hidden_size: A list of dimensions for each hidden layer of the MLP.
        dropout: Dropout probability for the linear layers.
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        n_tasks: int,
        in_channels: int,
        in_len: int,
        act_func: Optional[str] = None,
        hidden_size: List[int] = [],
        norm: bool = False,
        dropout: float = 0.0,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        # Save params
        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.in_len = in_len
        self.act_func = act_func
        self.hidden_size = hidden_size
        self.norm = norm
        self.dropout = dropout

        # Create layers
        self.blocks = nn.ModuleList()
        in_len = self.in_len * self.in_channels

        # hidden layers
        for h in self.hidden_size:
            self.blocks.append(
                LinearBlock(
                    in_len,
                    h,
                    norm=self.norm,
                    act_func=self.act_func,
                    dropout=self.dropout,
                    dtype=dtype,
                    device=device,
                )
            )
            in_len = h  # Output len of this block is the input len of next block

        # Final layer
        self.blocks.append(
            LinearBlock(
                in_len,
                self.n_tasks,
                norm=self.norm,
                act_func=None,
                dropout=self.dropout,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input data.
        """
        # Concatenate channels into the length axis
        x = rearrange(x, "b t l -> b 1 (t l)")
        # Apply linear blocks on the length axis
        for block in self.blocks:
            x = block(x)
        # Swap output tasks back to the channels axis
        x = rearrange(x, "b 1 l -> b l 1")
        return x
