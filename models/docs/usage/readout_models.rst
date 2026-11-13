.. _usage-readout-models:

#################################################
 Read-out models and per-dataset step counts
#################################################

.. note::

   This page documents additions of the fork
   `Mamoru-Yuasa/anemoi-core <https://github.com/Mamoru-Yuasa/anemoi-core>`_
   (branch ``research/mixed-cadence-anemoi-0.20``, tag
   ``mixed-cadence-2``). They are not part of upstream
   ``ecmwf/anemoi-core``. They are used with the training task
   ``MixedCadenceForecaster`` (anemoi-training, *Mixed-cadence
   training*).

***********************************
 Per-dataset input and output steps
***********************************

``n_step_input`` and ``n_step_output`` of
:class:`~anemoi.models.interface.AnemoiModelInterface` and
:class:`~anemoi.models.models.BaseGraphModel` accept either one integer
(upstream behaviour) or a mapping ``{dataset: count}``. The training
task supplies them (``task.num_input_timesteps`` and
``task.num_output_timesteps``); with ``MixedCadenceForecaster`` and a
6h block, global 6h and regional 3h inputs ``{gsm: 2, msm: 3}`` give
outputs ``{gsm: 1, msm: 2}``.

With a mapping:

-  the keys must equal the model datasets and the values must be
   positive integers (``ValueError`` otherwise);
-  each encoder's input width is ``n_step_input[d] * channels[d] +
   node attributes``, each decoder's output width ``n_step_output[d] *
   channels[d]``;
-  ``predict_step`` keeps the latest ``n_step_input[d]`` dates of each
   dataset, and the inference metadata stores ``input_timesteps`` per
   dataset;
-  the tendency pre-/post-processors are single-step or step-wise per
   dataset, from that dataset's ``n_step_output``.

``BaseGraphModel.n_step_input_for_dataset(name)`` and
``n_step_output_for_dataset(name)`` return the count of one dataset for
both forms.

A model class opts in through ``_supports_dataset_timestep_counts()``.
In the fork only ``AnemoiModelEncProcDec`` (the exact class) and
``AnemoiModelEncProcDecReadouts`` do; the ensemble, transport and
hierarchical models raise ``NotImplementedError`` when given a mapping.

******************
 Read-out models
******************

:class:`~anemoi.models.models.AnemoiModelEncProcDecReadouts` decodes a
dataset after chosen **processor layers**, one output time per layer,
instead of decoding all output times from the last layer. For a global
dataset every 6h and a regional dataset every 3h in a 6h block, the
regional +3h state can be read after layer 10 and the +6h states of
both after the last layer (16):

.. code:: text

   inputs ─► encoders ─► processor layers 1 … 10 ─┬─► … 16 ─┬─► decoder gsm ─► G[t+6]
                                                  │         └─► decoder msm ─► R[t+6]
                                                  └─► decoder msm ─► R[t+3]

Configuration
=============

.. code:: yaml

   model:
     processor:
       num_layers: 16
     model:
       _target_: anemoi.models.models.AnemoiModelEncProcDecReadouts
       readout_layers:
         gsm: [16]
         msm: [10, 16]
       readout_gradient_scale: 0.1
       readout_residual: anchored
       readout_lead_time: true

``readout_layers`` (required)
   For every output dataset, the processor layers (counted from 1) after
   which it is decoded, strictly increasing, one per output step and in
   time order. The number of layers must equal the dataset's number of
   output steps; every output dataset must be listed and needs a decoder
   of its own (a decoder shared by several datasets raises
   ``NotImplementedError``).

``readout_gradient_scale`` (default ``1.0``, range ``[0, 1]``)
   Gradient gate on read-outs taken before the last layer. The forward
   pass is unchanged; the gradient these read-outs send into the
   processor and the encoders is multiplied by this factor, while the
   decoders receive the full gradient. ``0`` trains the shared layers as
   if the intermediate read-outs did not exist.

``readout_residual`` (default ``"anchored"``)
   ``"anchored"``: every output is the latest input state plus its
   decoded increment. ``"accumulated"``: every output is the previous
   output plus its decoded increment (the first one is anchored).

``readout_lead_time`` (default ``false``)
   Add the position of the read-out in the block, :math:`k / K`, as one
   more feature of the decoder's destination nodes.

The options are validated by ``ReadoutModelSchema`` (``model.model``
schema ``ReadoutModel``).

How it works
============

``AnemoiModelEncProcDec.forward`` is split into ``_encode``,
``_process`` and ``_decode`` (behaviour unchanged). The read-out model
calls the processor with ``readout_layers``;
``GraphTransformerProcessor.forward`` then returns a dictionary from
each read-out layer (and the last layer) to the node features after it.
``BaseProcessor.run_layers_with_readouts`` runs the layers in segments
split at the chunk boundaries **and** at the read-out layers, so a
read-out does not need to lie on a chunk boundary, and gradient
checkpointing applies per segment. The latent skip connection, if
enabled, is added to every read-out latent. A read-out dataset's decoder
returns one time per call, so its weights do not depend on the number
of output steps; the outputs are concatenated along the time dimension.

Limitations
===========

-  Only ``GraphTransformerProcessor`` accepts ``readout_layers`` in its
   ``forward``; the other processors are not supported (the
   construction check only tests for ``run_layers_with_readouts``, which
   every ``BaseProcessor`` has, so an unsupported processor fails at the
   first forward pass).
-  Every output dataset needs its own decoder.

***********
 Reference
***********

.. autoclass:: anemoi.models.models.encoder_processor_decoder_readouts.AnemoiModelEncProcDecReadouts
   :members: forward
   :no-undoc-members:
   :show-inheritance:
   :no-index:

.. autofunction:: anemoi.models.models.encoder_processor_decoder_readouts.scale_gradient
   :no-index:

.. automethod:: anemoi.models.layers.processor.BaseProcessor.run_layers_with_readouts
   :no-index:

.. automethod:: anemoi.models.layers.processor.BaseProcessor.check_readout_layers
   :no-index:
