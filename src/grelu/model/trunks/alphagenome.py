from typing import Optional, Union, Dict, Any, Sequence
import torch
import torch.nn as nn
from alphagenome_pytorch.model import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy


'''
# TODO

1. Cropping
2. Trunk or Model; Seems like we contains all the heads here
3. Parameter limits: We can output multiple trunks/resolution/organism at the same time, while
   current design limits it.
'''


# Standard 1-D track heads available at resolution 128 (all assays).
# chip_tf and chip_histone are only available at 128bp; the rest support 1bp + 128bp.
_ALL_TRACK_HEADS = ('atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone')


class AlphaGenomeTrunk(nn.Module):
    """
    A trunk that wraps the AlphaGenome model from alphagenome-pytorch.

    Args:
        num_organisms: Number of organisms (default 2: human, mouse).
        organism_index: Default organism index to use for inference.
        output_key: The output modality to extract. Either:
            - A single head name (e.g. 'atac', 'rna_seq', 'contact_maps').
            - The literal 'all' to concatenate all standard 1-D track heads
              (atac, dnase, procap, cage, rna_seq, chip_tf, chip_histone) along
              the channel dim. Only valid at resolution=128 (since chip_* heads
              are 128-only).
            - A list/tuple of head names — also concatenated along channels.
        resolution: The resolution to extract (1 or 128). Must be 128 when
            output_key includes chip_tf or chip_histone.
        dtype_policy: DtypePolicy for precision control.
        weights_path: Optional path to a pretrained weights file (.pth).
        gradient_checkpointing: If True, enable gradient checkpointing.
        **kwargs: Additional arguments passed to AlphaGenome constructor.
    """
    def __init__(
        self,
        num_organisms: int = 2,
        organism_index: int = 0,
        output_key: Union[str, Sequence[str]] = "atac",
        resolution: int = 128,
        dtype_policy: Optional[DtypePolicy] = None,
        weights_path: Optional[str] = None,
        gradient_checkpointing: bool = False,
        **kwargs
    ):
        super().__init__()

        self.organism_index = organism_index
        self.resolution = resolution

        # Normalize output_key into either a single string or a tuple of head names.
        if isinstance(output_key, str):
            if output_key == "all":
                self.output_keys = _ALL_TRACK_HEADS
                self._multi_head = True
                self.output_key = output_key
            else:
                self.output_keys = (output_key,)
                self._multi_head = False
                self.output_key = output_key
        else:
            self.output_keys = tuple(output_key)
            self._multi_head = True
            self.output_key = tuple(output_key)

        if self._multi_head and resolution != 128:
            # chip_tf / chip_histone only exist at 128bp; mixing resolutions
            # would yield mismatched length dims that cannot be concatenated.
            raise ValueError(
                f"Multi-head output requires resolution=128 (got {resolution}). "
                f"Heads chip_tf and chip_histone are only available at 128bp."
            )

        if weights_path:
            self.model = AlphaGenome.from_pretrained(
                weights_path,
                dtype_policy=dtype_policy,
                num_organisms=num_organisms,
                gradient_checkpointing=gradient_checkpointing,
                **kwargs
            )
        else:
            self.model = AlphaGenome(
                num_organisms=num_organisms,
                dtype_policy=dtype_policy,
                gradient_checkpointing=gradient_checkpointing,
                **kwargs
            )

        # Determine out_channels for gReLU head
        if self._multi_head:
            total = 0
            for k in self.output_keys:
                if k not in self.model.heads:
                    raise ValueError(
                        f"Multi-head mode only supports standard track heads "
                        f"({list(self.model.heads.keys())}); got '{k}'."
                    )
                total += self.model.heads[k].num_tracks
            self.out_channels = total
        elif output_key in self.model.heads:
            self.out_channels = self.model.heads[output_key].num_tracks
        elif output_key == "contact_maps":
            self.out_channels = 28 # CONTACT_MAPS_OUTPUT_TRACKS
        elif output_key == "splice_sites":
            self.out_channels = 5
        elif output_key == "splice_site_usage":
            self.out_channels = self.model.splice_sites_usage_head.num_output_tracks
        elif output_key == "splice_junctions":
            self.out_channels = self.model.splice_sites_junction_head._num_tissues * 2
        else:
            self.out_channels = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, 4, L)

        Returns:
            Output tensor of shape (N, C, L_out)
        """
        # gReLU uses (N, 4, L), AlphaGenome expects (N, L, 4)
        x = x.transpose(1, 2)

        # Determine organism index per batch
        batch_size = x.shape[0]
        organism_index = torch.full(
            (batch_size,),
            self.organism_index,
            dtype=torch.long,
            device=x.device
        )

        # Restrict computed heads / resolutions for efficiency when possible.
        # Only the standard track heads accept the `heads=` filter; for non-standard
        # outputs (contact_maps, splicing) we let the model compute everything.
        forward_kwargs = dict(channels_last=False)
        if self._multi_head or self.output_key in self.model.heads:
            forward_kwargs["heads"] = tuple(self.output_keys)
            forward_kwargs["resolutions"] = (self.resolution,)

        if self.training:
            # Use forward directly during training to keep gradients
            # We set channels_last=False to get (N, C, L) back
            outputs = self.model(x, organism_index, **forward_kwargs)
        else:
            # Use predict for inference (handles no_grad and autocast)
            outputs = self.model.predict(x, organism_index, **forward_kwargs)

        if self._multi_head:
            # Concatenate every selected head along the channel dimension.
            # All standard track heads at the same resolution share length L,
            # so we get (B, sum(T_k), L).
            tensors = []
            for k in self.output_keys:
                t = outputs[k]
                if isinstance(t, dict):
                    t = t[self.resolution]
                tensors.append(t)
            return torch.cat(tensors, dim=1)

        # Extract desired output (single head)
        out = outputs[self.output_key]

        if isinstance(out, dict):
            out = out[self.resolution]

        # For contact_maps, AlphaGenome returns (B, T, S1, S2) if channels_last=False
        # gReLU expects (B, C, L). For contact maps, this might need special handling if used in 1D tasks.
        # But for standard tracks, it's (B, T, S).

        return out


class AlphaGenomeFeatureTrunk(nn.Module):
    """AlphaGenome embedding trunk for downstream fine-tuning heads.

    This wrapper exposes AlphaGenome sequence embeddings instead of pretrained
    assay predictions so gReLU can attach a normal trainable head. The output is
    always NCL format: ``(batch, channels, bins)``.
    """

    def __init__(
        self,
        num_organisms: int = 2,
        organism_index: int = 1,
        resolution: int = 128,
        dtype_policy: Optional[DtypePolicy] = None,
        weights_path: Optional[str] = None,
        gradient_checkpointing: bool = False,
        **kwargs
    ):
        super().__init__()
        if resolution not in {1, 128}:
            raise ValueError(f"AlphaGenome feature resolution must be 1 or 128, got {resolution}")

        self.organism_index = organism_index
        self.resolution = resolution
        self.out_channels = 1536 if resolution == 1 else 3072

        if weights_path:
            self.model = AlphaGenome.from_pretrained(
                weights_path,
                dtype_policy=dtype_policy,
                num_organisms=num_organisms,
                gradient_checkpointing=gradient_checkpointing,
                **kwargs
            )
        else:
            self.model = AlphaGenome(
                num_organisms=num_organisms,
                dtype_policy=dtype_policy,
                gradient_checkpointing=gradient_checkpointing,
                **kwargs
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gReLU uses (N, 4, L), AlphaGenome expects (N, L, 4).
        x = x.transpose(1, 2)
        organism_index = torch.full(
            (x.shape[0],),
            self.organism_index,
            dtype=torch.long,
            device=x.device,
        )

        outputs = self.model.encode(
            x,
            organism_index,
            resolutions=(self.resolution,),
            channels_last=False,
        )
        return outputs[f"embeddings_{self.resolution}bp"]
