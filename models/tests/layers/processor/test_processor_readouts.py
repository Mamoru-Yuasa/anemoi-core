# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Intermediate states of the graph-transformer processor (read-outs)."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.layers.processor import GraphTransformerProcessor
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.models.encoder_processor_decoder_readouts import scale_gradient

NUM_NODES = 60
NUM_EDGES = 240
CHANNELS = 32


def _processor(num_layers: int, num_chunks: int, checkpointing: bool):
    torch.manual_seed(0)
    graph = HeteroData()
    graph[("nodes", "to", "nodes")].edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))
    graph[("nodes", "to", "nodes")].edge_attr1 = torch.rand((NUM_EDGES, 3))
    provider = create_graph_provider(
        graph=graph[("nodes", "to", "nodes")],
        edge_attributes=["edge_attr1"],
        src_size=NUM_NODES,
        dst_size=NUM_NODES,
        trainable_size=2,
    )
    processor = GraphTransformerProcessor(
        num_layers=num_layers,
        num_channels=CHANNELS,
        num_chunks=num_chunks,
        num_heads=4,
        mlp_hidden_ratio=2,
        qk_norm=False,
        cpu_offload=False,
        layer_kernels=load_layer_kernels(instance=False),
        graph_attention_backend="pyg",
        edge_dim=provider.edge_dim,
        gradient_checkpointing=checkpointing,
    )
    edge_attr, edge_index, _ = provider.get_edges(batch_size=1)
    shard_info = GraphShardInfo(nodes=[NUM_NODES], edges=[NUM_EDGES])
    return processor, (1, shard_info, edge_attr, edge_index)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("readouts", [[2, 4], [3, 4], [1, 2, 3], [4]])
def test_readouts_match_truncated_processors(readouts, checkpointing):
    """The state after layer l equals the output of the first l layers."""
    processor, args = _processor(num_layers=4, num_chunks=2, checkpointing=checkpointing)
    x = torch.rand(NUM_NODES, CHANNELS)
    states = processor(x, *args, readout_layers=readouts)
    assert sorted(states) == sorted(set(readouts) | {4})
    torch.testing.assert_close(states[4], processor(x, *args), rtol=0, atol=0)
    for layer in readouts:
        truncated, _ = _processor(num_layers=layer, num_chunks=1, checkpointing=False)
        truncated.load_state_dict({k: v for k, v in processor.state_dict().items() if k in truncated.state_dict()})
        torch.testing.assert_close(states[layer], truncated(x, *args), rtol=1e-6, atol=1e-6)


def test_readout_layers_are_checked():
    processor, args = _processor(num_layers=4, num_chunks=2, checkpointing=False)
    x = torch.rand(NUM_NODES, CHANNELS)
    with pytest.raises(ValueError):
        processor(x, *args, readout_layers=[5])
    with pytest.raises(ValueError):
        processor(x, *args, readout_layers=[0])


def test_gradient_reaches_only_layers_before_the_readout():
    processor, args = _processor(num_layers=4, num_chunks=2, checkpointing=True)
    x = torch.rand(NUM_NODES, CHANNELS)
    states = processor(x, *args, readout_layers=[2])
    states[2].pow(2).sum().backward()
    for layer_id, block in enumerate(processor.proc):
        grads = [p.grad for p in block.parameters() if p.requires_grad]
        reached = any(g is not None and g.abs().sum() > 0 for g in grads)
        assert reached == (layer_id < 2)


@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0])
def test_scale_gradient(scale):
    a = torch.rand(5, requires_grad=True)
    y = scale_gradient(a * 2.0, scale)
    torch.testing.assert_close(y, a.detach() * 2.0)
    (y.sum() + 0.0 * a.sum()).backward()  # keeps a graph when scale is 0
    torch.testing.assert_close(a.grad, torch.full_like(a, 2.0 * scale))
