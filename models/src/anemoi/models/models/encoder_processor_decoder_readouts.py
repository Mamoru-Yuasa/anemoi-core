# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Encoder-processor-decoder that reads datasets out after chosen processor layers."""

import logging
from typing import Optional

import einops
import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec

LOGGER = logging.getLogger(__name__)

READOUT_RESIDUALS = ("anchored", "accumulated")


def scale_gradient(x: Tensor, scale: float) -> Tensor:
    """Return ``x`` unchanged in the forward pass, with its gradient multiplied by ``scale``."""
    if scale == 1.0:
        return x
    if scale == 0.0:
        return x.detach()
    return scale * x + (1.0 - scale) * x.detach()


class AnemoiModelEncProcDecReadouts(AnemoiModelEncProcDec):
    """Read each dataset out of the latent after chosen processor layers.

    ``model.model.readout_layers`` maps every output dataset to the processor layers (counted
    from one) after which it is decoded, one output time per layer and in time order, e.g.
    ``{gsm: [16], msm: [10, 16]}`` for a global member every 6 h and a regional member every
    3 h in a 6 h block. The number of read-outs of a dataset must equal its number of output
    steps. The decoder of such a dataset returns one time per call, so its weights do not
    depend on the number of output steps.

    Options in ``model.model``:

    readout_gradient_scale : float, default 1.0
        Gradient gate on read-outs taken before the last processor layer. The forward pass is
        unchanged; the gradient that these read-outs send into the encoders and the processor
        is multiplied by this factor (the decoders receive the full gradient). 0 trains the
        encoders and the processor as if the intermediate read-outs did not exist.
    readout_residual : "anchored" or "accumulated", default "anchored"
        "anchored": every output is the latest input state plus its decoded increment.
        "accumulated": every output is the previous output plus its decoded increment.
    readout_lead_time : bool, default False
        Give the decoder the lead time of the read-out (its position in the block, k / K) as one
        more feature of the destination nodes.
    """

    def __init__(self, *, model_config, **kwargs) -> None:
        model_options = model_config.model.model
        readout_layers = model_options.get("readout_layers", None)
        if not readout_layers:
            raise ValueError(f"{type(self).__name__} needs model.model.readout_layers")
        self.readout_layers = {str(name): [int(layer) for layer in layers] for name, layers in readout_layers.items()}
        self.readout_gradient_scale = float(model_options.get("readout_gradient_scale", 1.0))
        self.readout_residual = str(model_options.get("readout_residual", "anchored"))
        self.readout_lead_time = bool(model_options.get("readout_lead_time", False))
        if not 0.0 <= self.readout_gradient_scale <= 1.0:
            raise ValueError(f"readout_gradient_scale must lie in [0, 1], got {self.readout_gradient_scale}")
        if self.readout_residual not in READOUT_RESIDUALS:
            raise ValueError(f"readout_residual must be one of {READOUT_RESIDUALS}, got {self.readout_residual!r}")
        super().__init__(model_config=model_config, **kwargs)
        self._check_readouts()

    def _supports_dataset_timestep_counts(self) -> bool:
        return True

    def _check_readouts(self) -> None:
        if not hasattr(self.processor, "run_layers_with_readouts"):
            raise NotImplementedError(f"{type(self.processor).__name__} cannot return intermediate states")
        missing = set(self.target_datasets) - set(self.readout_layers)
        extra = set(self.readout_layers) - set(self.target_datasets)
        if missing or extra:
            raise ValueError(
                f"readout_layers must name every output dataset: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for dataset_name, layers in self.readout_layers.items():
            if layers != sorted(set(layers)):
                raise ValueError(f"readout_layers[{dataset_name!r}] must increase strictly, got {layers}")
            self.processor.check_readout_layers(layers)
            steps = self.n_step_output_for_dataset(dataset_name)
            if len(layers) != steps:
                raise ValueError(
                    f"Dataset {dataset_name!r} has {steps} output steps but {len(layers)} read-out layers {layers}"
                )
            decoder_name = self.dataset2decoder[dataset_name]
            shared = [d for d in self.decoder2datasets[decoder_name] if d != dataset_name]
            if shared:
                raise NotImplementedError(f"Decoder {decoder_name!r} is shared with {shared}; give each its own")
        LOGGER.info(
            "Read-out layers %s of %d, gradient scale %.3g on read-outs before the last layer, %s residual%s",
            self.readout_layers,
            self.processor.num_layers,
            self.readout_gradient_scale,
            self.readout_residual,
            ", lead time given to the decoders" if self.readout_lead_time else "",
        )

    # The decoder of a read-out dataset returns one time per call.
    def _calculate_output_dim(self, dataset_name: str) -> int:
        if dataset_name in self.readout_layers:
            return self.num_output_channels[dataset_name]
        return super()._calculate_output_dim(dataset_name)

    def _calculate_target_dim(self, dataset_name: str) -> int:
        extra = 1 if (self.readout_lead_time and dataset_name in self.readout_layers) else 0
        return super()._calculate_target_dim(dataset_name) + extra

    def forward(
        self,
        x: dict[str, Tensor],
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        state = self._encode(x, model_comm_group=model_comm_group, grid_shard_sizes=grid_shard_sizes)

        layers = sorted({layer for dataset_layers in self.readout_layers.values() for layer in dataset_layers})
        latents = self._process(state, model_comm_group=model_comm_group, readout_layers=layers)
        last_layer = self.processor.num_layers

        x_out_dict = {}
        for dataset_name in self.target_datasets:
            x_target_latent, shard_sizes_target = self._assemble_targets(
                x[dataset_name],
                state["x_data_latent"].get(dataset_name, None),
                state["batch_size"],
                grid_shard_sizes,
                model_comm_group,
                dataset_name,
            )
            dataset_layers = self.readout_layers[dataset_name]
            x_skip = state["x_skip"].get(dataset_name, None)
            previous = None
            outputs = []
            for k, layer in enumerate(dataset_layers):
                latent = latents[layer]
                if self.latent_skip:
                    latent = latent + state["x_latent"]
                if layer < last_layer:
                    latent = scale_gradient(latent, self.readout_gradient_scale)

                target = x_target_latent
                if self.readout_lead_time:
                    lead = torch.full_like(target[..., :1], (k + 1) / len(dataset_layers))
                    target = torch.cat([target, lead], dim=-1)

                x_out = self._decode(
                    latent,
                    target,
                    shard_sizes_target,
                    state,
                    dataset_name,
                    model_comm_group=model_comm_group,
                )
                state_k = self._assemble_readout(
                    x_out,
                    x_skip,
                    previous,
                    batch_size=state["batch_size"],
                    ensemble_size=state["ensemble_size"],
                    dtype=x[dataset_name].dtype,
                    dataset_name=dataset_name,
                )
                outputs.append(state_k)
                previous = state_k
            x_out_dict[dataset_name] = torch.cat(outputs, dim=1)

        return x_out_dict

    def _assemble_readout(
        self,
        x_out: Tensor,
        x_skip: Tensor | None,
        previous: Tensor | None,
        *,
        batch_size: int,
        ensemble_size: int,
        dtype: torch.dtype,
        dataset_name: str,
    ) -> Tensor:
        """One output time: reshape, add the residual and apply the boundings.

        The residual is the latest input state ("anchored", or the first read-out of
        "accumulated") or the previous output ("accumulated"), on the prognostic variables.
        """
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch 1 ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=dtype)
            .clone()
        )
        output_idx = self._internal_output_idx[dataset_name]
        if self.readout_residual == "accumulated" and previous is not None:
            x_out[..., output_idx] += previous[..., output_idx]
        elif x_skip is not None:
            assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
            x_out[..., output_idx] += x_skip[:, :1][..., self._internal_input_idx[dataset_name]]

        for bounding in self.boundings[dataset_name]:
            x_out = bounding(x_out)
        return x_out
