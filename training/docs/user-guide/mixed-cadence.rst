.. _mixed-cadence target:

########################
 Mixed-cadence training
########################

.. note::

   This page documents an addition of the fork
   `Mamoru-Yuasa/anemoi-core <https://github.com/Mamoru-Yuasa/anemoi-core>`_
   (branch ``research/mixed-cadence-anemoi-0.20``, tag
   ``mixed-cadence-2``). It is not part of upstream ``ecmwf/anemoi-core``.
   The complete list of differences is in ``FORK.md`` at the root of the
   repository.

Mixed-cadence training trains one model on several datasets that exist
at **different native time steps**, e.g. global analyses every 6 h and
regional analyses every 3 h or 1 h, and predicts every dataset at its
own time step. No value of a dataset is loaded, interpolated or
predicted at a date where that dataset does not exist.

*************
 The block
*************

Every model call advances all datasets by one common **block**. A
dataset with time step :math:`\Delta_d` has :math:`K_d = B / \Delta_d`
outputs in a block of length :math:`B`. With a 6 h block, global data
every 6 h and regional data every 3 h, the default configuration is

.. code:: text

   F(G[t-6], G[t], R[t-6], R[t-3], R[t]) -> (G[t+6], R[t+3], R[t+6])

where ``G`` is the global and ``R`` the regional dataset. The model
receives each dataset's input window, predicts all outputs of the block
in one forward pass, and the rollout moves on by one block. The outputs
of one block are fed back only at the next block.

*********************
 Configuring the task
*********************

The task is :class:`~anemoi.training.tasks.MixedCadenceForecaster`
(config group ``task=mixed_cadence_forecaster``):

.. code:: yaml

   task:
     _target_: anemoi.training.tasks.MixedCadenceForecaster
     block_timestep: 6h
     dataset_timesteps: {global: 6h, regional: 3h}   # regional: 1h gives six outputs
     multistep_input: {global: 2, regional: 3}       # latest N dates at each time step
     # input_offsets: {regional: [-6h, 0h]}          # explicit window, see below
     rollout: {start: 1, epoch_increment: 0, maximum: 1}
     validation_rollout: null

``block_timestep``
   Time advanced by one model call, common to all datasets.

``dataset_timesteps``
   Native time step of every dataset (keys are the dataset names of
   ``data.datasets``). Each time step must divide ``block_timestep``.

``multistep_input``
   Number of input dates per dataset, either one integer for all
   datasets or a mapping with exactly the keys of ``dataset_timesteps``.
   The inputs are the latest ``N`` dates at the dataset's own time step:
   ``N = 3`` at 3 h gives the offsets ``[-6h, -3h, 0h]``.

``input_offsets``
   Optional explicit input offsets per dataset, overriding
   ``multistep_input`` for those datasets. They must end at ``0h``, be
   multiples of the dataset's time step, and each one must be either an
   input or an output of the previous block (otherwise the task raises
   ``ValueError`` at construction). Example: ``regional: [-6h, 0h]``
   gives ``F(G[t-6], G[t], R[t-6], R[t])``.

``rollout`` / ``validation_rollout``
   As for :class:`~anemoi.training.tasks.forecaster.Forecaster`, counted
   in blocks.

Combinations of time steps
==========================

Any number of datasets and any combination of time steps can be used if

-  every time step divides the block (e.g. 12h/4h with a 12h block,
   6h/2h/1h with a 6h block, 24h/6h with a 24h block; 6h/4h needs a 12h
   block and is rejected with a 6h block);
-  every input offset can be filled from an input or an output of the
   previous block;
-  every dataset has data at the block boundaries, since samples are
   built from the dates common to all datasets.

The model must accept a different number of input and output steps per
dataset. In this fork that is
:class:`~anemoi.models.models.AnemoiModelEncProcDec` and
:class:`~anemoi.models.models.AnemoiModelEncProcDecReadouts` (see the
anemoi-models page *Read-out models and per-dataset step counts*). The
ensemble, transport and hierarchical models raise
``NotImplementedError`` when given per-dataset counts.

*****************************
 What happens in the pipeline
*****************************

Offsets per dataset
===================

``BaseTask.get_offsets`` and ``get_output_offsets`` of this task take
a ``dataset_name``. The datamodule calls them through
:func:`anemoi.training.data.relative_time_indices.get_dataset_offsets`,
which passes ``dataset_name`` only to tasks whose method accepts it, so
the upstream tasks are called as before. The relative date indices of
each dataset are therefore computed at its own frequency, and the
tendency statistics are requested at each dataset's own lead times
(e.g. ``6h`` global, ``3h`` and ``6h`` regional).

Samples aligned by date
=======================

Upstream intersects the valid anchor **positions** of all readers. For
readers with different frequencies (or the same frequency but different
start dates) the same position refers to different dates. The fork
replaces this, for single time-series readers, with
:func:`~anemoi.training.data.usable_indices.compute_aligned_anchors`:

#. each reader computes its valid anchors for its own relative date
   indices;
#. anchors whose window does not match the requested dates exactly are
   dropped (this rejects windows across gaps in the date array that are
   not marked as missing);
#. the valid anchor **dates** of all readers are intersected;
#. every reader keeps its own position for each shared date.

``MultiDataset.reader_anchors`` holds one anchor array per reader; row
``i`` of every array is the same sample. ``MultiDataset.anchors`` is the
array of the first reader, and ``MultiDataset.frequency`` is the first
reader's frequency. Trajectory readers (several sequences) keep the
upstream positional intersection. The log reports, for example::

   Data reader 'gsm' has 1458 valid anchors
   Data reader 'msm' has 2916 valid anchors
   MultiDataset has 1458 valid anchors after intersection.

Rollout and the advance map
===========================

For each dataset the task builds an **advance map** that fills every
input slot of the next block either from an output of the current block
(``outin``, prognostic variables; forcings are taken from the batch) or
from a current input slot (``inin``). The same map is used in training
(``advance_input``) and stored in the checkpoint for inference.

Loss scaling
============

``UniformTimeStepScaler`` sizes its weights with each dataset's number
of output steps (``create_scalers`` passes ``dataset_name``).

**********************
 Checkpoint metadata
**********************

``fill_metadata`` writes ``metadata_inference.task =
"mixed-cadence-forecaster"`` and, for every dataset, under
``metadata_inference.<dataset>.timesteps``:

.. list-table::
   :header-rows: 1

   -  -  key
      -  content (global 6h, regional 3h, 6h block, default inputs)
   -  -  ``timestep``
      -  ``6h`` / ``3h``
   -  -  ``input_offsets``, ``output_offsets``
      -  ``[-6h, 0h]``, ``[6h]`` / ``[-6h, -3h, 0h]``, ``[3h, 6h]``
   -  -  ``input_relative_date_indices``, ``output_relative_date_indices``
      -  the same in units of the dataset's time step
   -  -  ``relative_date_indices_training``
      -  union of the two
   -  -  ``rollout_shift``
      -  the block, ``6h``
   -  -  ``advance_map``
      -  ``{"inin": [...], "outin": [...]}``

The offset keys are those of ``OffsetForecaster`` checkpoints. The fork
of anemoi-inference (tag ``mixed-cadence-2``) detects the task name and
rolls the model out block by block, emitting each dataset at its own
dates.

*************
 Limitations
*************

-  The plot callbacks (``plot_adapter``) assume one number of input and
   output steps for all datasets; disable them
   (``diagnostics.plot.datasets_to_plot: []``) for mixed-cadence runs.
-  Only ``AnemoiModelEncProcDec`` and ``AnemoiModelEncProcDecReadouts``
   accept per-dataset step counts.
-  Inference needs the anemoi-inference fork; upstream anemoi-inference
   treats the checkpoint as a single-cadence model.
-  Verified with global 6 h / regional 3 h on real data (JMA GSM and
   MSM) and global 6 h / regional 1 h in unit tests.

***********
 Reference
***********

.. autoclass:: anemoi.training.tasks.MixedCadenceForecaster
   :members: get_offsets, get_input_offsets, get_output_offsets, advance_input, fill_metadata
   :no-undoc-members:
   :show-inheritance:
   :no-index:

.. autofunction:: anemoi.training.data.usable_indices.compute_aligned_anchors
   :no-index:

.. autofunction:: anemoi.training.data.relative_time_indices.get_dataset_offsets
   :no-index:
