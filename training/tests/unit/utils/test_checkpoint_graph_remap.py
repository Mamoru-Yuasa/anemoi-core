# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from torch_geometric.data import HeteroData

from anemoi.training.utils.checkpoint import remap_graph_trainables

DECODER = "model.model.decoder_graph_provider.data.trainable.trainable"
DECODER_LAYOUT = "model.model.decoder_graph_provider.data.trainable_layout_version"
HIDDEN_NODES = "model.model.node_attributes.trainable_tensors.hidden.trainable"
DATA_NODES = "model.model.node_attributes.trainable_tensors.data.trainable"


def _graph(hidden: list[list[float]], decoder_src: list[int]) -> HeteroData:
    """Three data nodes, each decoded from one hidden node."""
    graph = HeteroData()
    graph["data"].x = torch.tensor([[0.0, 0.0], [0.0, 0.1], [0.0, 0.2]])
    graph["hidden"].x = torch.tensor(hidden)
    graph[("hidden", "to", "data")].edge_index = torch.tensor([decoder_src, [0, 1, 2]])
    return graph


def _state(n_hidden: int, value: float) -> dict[str, torch.Tensor]:
    return {
        DECODER: torch.full((3, 2), value) + torch.arange(3.0)[:, None],
        DECODER_LAYOUT: torch.tensor(1),
        HIDDEN_NODES: torch.full((n_hidden, 2), value) + torch.arange(float(n_hidden))[:, None],
        DATA_NODES: torch.full((3, 2), value) + torch.arange(3.0)[:, None],
    }


def test_same_shape_edges_are_matched_by_position_not_index() -> None:
    # Two hidden nodes in the checkpoint; the model inserts a refined node before
    # them, so data node 2 is now decoded from the new node.
    ckpt_graph = _graph([[0.0, 0.0], [0.0, 0.2]], decoder_src=[0, 0, 1])
    model_graph = _graph([[0.0, 0.15], [0.0, 0.0], [0.0, 0.2]], decoder_src=[1, 1, 0])
    state = _state(2, 100.0)
    model = _state(3, 0.0)

    remap_graph_trainables(state, model, ckpt_graph, model_graph, hidden_name="hidden")

    # Decoder edges 0 and 1 link the same points and keep their parameters;
    # edge 2 now starts at the new node and keeps the model's initial values.
    torch.testing.assert_close(state[DECODER][:2], torch.tensor([[100.0, 100.0], [101.0, 101.0]]))
    torch.testing.assert_close(state[DECODER][2], model[DECODER][2])
    # The hidden-node parameters of the two unchanged nodes move to their new rows.
    torch.testing.assert_close(state[HIDDEN_NODES][1], torch.tensor([100.0, 100.0]))
    torch.testing.assert_close(state[HIDDEN_NODES][2], torch.tensor([101.0, 101.0]))
    torch.testing.assert_close(state[HIDDEN_NODES][0], model[HIDDEN_NODES][0])
    # The data nodes did not change.
    torch.testing.assert_close(state[DATA_NODES], _state(2, 100.0)[DATA_NODES])


def test_identical_graphs_are_left_untouched() -> None:
    graph = _graph([[0.0, 0.0], [0.0, 0.2]], decoder_src=[0, 0, 1])
    state = _state(2, 100.0)
    expected = {key: value.clone() for key, value in state.items()}

    remap_graph_trainables(state, _state(2, 0.0), graph, graph, hidden_name="hidden")

    for key, value in expected.items():
        torch.testing.assert_close(state[key], value)


def test_older_edge_layout_is_not_remapped() -> None:
    ckpt_graph = _graph([[0.0, 0.0], [0.0, 0.2]], decoder_src=[0, 0, 1])
    model_graph = _graph([[0.0, 0.15], [0.0, 0.0], [0.0, 0.2]], decoder_src=[1, 1, 0])
    state = _state(2, 100.0)
    state[DECODER_LAYOUT] = torch.tensor(0)
    original = state[DECODER].clone()

    remap_graph_trainables(state, _state(3, 0.0), ckpt_graph, model_graph, hidden_name="hidden")

    torch.testing.assert_close(state[DECODER], original)
