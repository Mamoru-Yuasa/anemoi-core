.. _checkpoint-dataset-map:

#######################################################
 Transfer learning across datasets and graphs (fork)
#######################################################

.. note::

   This page documents additions of the fork
   `Mamoru-Yuasa/anemoi-core <https://github.com/Mamoru-Yuasa/anemoi-core>`_
   (branch ``research/mixed-cadence-anemoi-0.20``, tag
   ``mixed-cadence-2``) to
   :class:`~anemoi.training.checkpoint.loading.strategies.TransferLearningLoader`.
   They are not part of upstream ``ecmwf/anemoi-core``.

Upstream ``TransferLearningLoader`` copies every checkpoint tensor whose
**name and shape** match a tensor of the new model. Two situations of
staged training are not covered by that rule:

-  the datasets change name, e.g. a model pretrained on ``era5`` is
   fine-tuned on ``gsm`` and ``msm``: no encoder or decoder tensor has
   the same name;
-  the graph changes, e.g. a uniform mesh is refined or replaced by a
   stretched mesh: trainable node and edge tensors belong to particular
   nodes and edges, so a tensor of the same shape can describe different
   edges, and one of a different shape can still share most of them.

The fork adds a ``dataset_map`` option and a transfer of graph-specific
trainables by graph position.

*************
 Dataset map
*************

.. code:: yaml

   training:
     checkpoint:
       source:
         _target_: anemoi.training.checkpoint.sources.local.LocalSource
         path: /path/to/era5-pretrained.ckpt
       loading:
         _target_: anemoi.training.checkpoint.loading.strategies.TransferLearningLoader
         skip_mismatched: true
         dataset_map: {gsm: era5, msm: era5}   # model dataset -> checkpoint dataset

``dataset_map`` maps each **model** dataset to the **checkpoint** dataset
whose network weights it starts from. One checkpoint dataset may feed
several model datasets. Before anything else, the loader applies
:func:`~anemoi.training.utils.checkpoint.map_checkpoint_datasets` to the
loaded checkpoint (in memory; the file is not changed):

-  state-dict keys under ``encoder``, ``decoder``,
   ``encoder_graph_provider``, ``decoder_graph_provider`` and
   ``trainable_tensors`` whose dataset component is the checkpoint name
   are copied under the model name (the originals stay and are reported
   as "not present in the model");
-  ``hyper_parameters.data_indices`` and
   ``hyper_parameters.metadata.dataset`` get entries for the model names
   (copies of the checkpoint entries), so the usual variable-order and
   unit checks compare each model dataset with the checkpoint dataset it
   starts from.

Normalisers are **not** mapped: every model dataset keeps the statistics
of its own data. A checkpoint dataset without network weights raises
``ValueError``.

Messages to expect::

   Mapped 63 checkpoint tensors of dataset 'era5' to model dataset 'gsm'
   Mapped 63 checkpoint tensors of dataset 'era5' to model dataset 'msm'
   The order of the variables in the model matches ...
   Dataset 'era5' does not contain variables_metadata. Skipping unit compatibility check for this dataset.

The last warning concerns only the original checkpoint name (``era5``),
which has no counterpart among the model datasets; the mapped datasets
are checked. A unit mismatch raises ``ValueError`` unless
``training.check_variables_compatibility.ignore_units`` is set.

*****************************************
 Graph-specific trainables by position
*****************************************

When the checkpoint stores its graph (``hyper_parameters.graph_data``),
the model has one, and ``model.model.hidden_nodes_name`` is set, the
loader matches the rows of these tensors by **graph position** before
the shape filter:

.. list-table::
   :header-rows: 1

   -  -  state-dict key
      -  row key
   -  -  ``...node_attributes.trainable_tensors.<nodes>.trainable``
      -  node coordinates
   -  -  ``...encoder_graph_provider.<dataset>.trainable.trainable``
      -  source and destination coordinates of the edge
         ``(<dataset>, to, hidden)``
   -  -  ``...decoder_graph_provider.<dataset>.trainable.trainable``
      -  same for ``(hidden, to, <dataset>)``
   -  -  ``...processor_graph_provider.trainable.trainable``
      -  same for ``(hidden, to, hidden)``

Coordinates are compared exactly after rounding the radians to
:math:`10^{-8}`. Edge rows follow the destination-sorted layout of the
graph providers; edge tensors are remapped only when the checkpoint has
the same ``trainable_layout_version`` as the model. Rows found in the
checkpoint graph are copied; the others keep the model's initial
values. Tensors whose graphs are identical are left to the normal
transfer. Example (level-5 mesh to level-6 mesh)::

   Remapped model.model.node_attributes.trainable_tensors.hidden.trainable by graph position:
       10242 of 40962 rows from the checkpoint, the rest freshly initialised
   Remapped model.model.processor_graph_provider.trainable.trainable by graph position:
       81840 of 327600 rows from the checkpoint, the rest freshly initialised

A tensor whose nodes or edges are missing from either graph (e.g. the
``gsm`` data nodes in the graph of an ERA5 checkpoint) is reported with
``Not remapping ... by graph position`` and then follows the
name-and-shape rule; data-node tensors of a different grid are skipped
as shape-mismatched.

**********
 Reference
**********

.. autoclass:: anemoi.training.checkpoint.loading.strategies.TransferLearningLoader
   :no-members:
   :no-index:

.. autofunction:: anemoi.training.utils.checkpoint.map_checkpoint_datasets
   :no-index:

.. autofunction:: anemoi.training.utils.checkpoint.remap_graph_trainables
   :no-index:

.. autofunction:: anemoi.training.utils.checkpoint.remap_rows_by_keys
   :no-index:

.. autofunction:: anemoi.training.utils.checkpoint.node_row_keys
   :no-index:

.. autofunction:: anemoi.training.utils.checkpoint.edge_row_keys
   :no-index:
