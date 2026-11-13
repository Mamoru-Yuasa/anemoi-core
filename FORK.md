# Fork: mixed-cadence training on anemoi-core

This fork of [ecmwf/anemoi-core](https://github.com/ecmwf/anemoi-core) adds
training of one model on datasets that exist at different native time steps
(e.g. global analyses every 6 h, regional analyses every 3 h or 1 h), a
read-out model that decodes datasets after chosen processor layers, and
transfer learning across renamed datasets and changed graphs. It is research
code for [Mamoru-Yuasa/regional_AI](https://github.com/Mamoru-Yuasa/regional_AI)
(branch `research/mixed-cadence-global6h-regional3h1h`), which uses it as a
submodule.

| | |
|---|---|
| Branch | `research/mixed-cadence-anemoi-0.20` |
| Tag | `mixed-cadence-2` (release notes on GitHub) |
| Base | upstream release commit `6a2377d37` (2026-11-03, "chore: Release main (#1376)"): anemoi-models 0.20.0, anemoi-training 0.17.2, anemoi-graphs 0.9.9 |
| Packages changed | `anemoi-training`, `anemoi-models` (`anemoi-graphs` is unchanged) |
| Companion fork | [Mamoru-Yuasa/anemoi-inference](https://github.com/Mamoru-Yuasa/anemoi-inference), tag `mixed-cadence-2` (inference of mixed-cadence checkpoints) |
| Earlier tag | `mixed-cadence-1` (branch `research/mixed-cadence-global6h-regional3h1h`, on upstream main `02ca484` of 2026-09-25); its checkpoints are not loaded by this version |

## Installing

PyTorch must already be installed for your accelerator (upstream supports
`torch>=2.6,<2.12`; this fork was tested with 2.10.0+rocm7.2.4).

```bash
TAG=mixed-cadence-2
pip install \
  "anemoi-graphs   @ git+https://github.com/Mamoru-Yuasa/anemoi-core@${TAG}#subdirectory=graphs" \
  "anemoi-models   @ git+https://github.com/Mamoru-Yuasa/anemoi-core@${TAG}#subdirectory=models" \
  "anemoi-training @ git+https://github.com/Mamoru-Yuasa/anemoi-core@${TAG}#subdirectory=training" \
  "anemoi-inference @ git+https://github.com/Mamoru-Yuasa/anemoi-inference@${TAG}" \
  "anemoi-datasets==0.5.50"
```

or, for development, editable from a clone:

```bash
git clone --branch mixed-cadence-2 https://github.com/Mamoru-Yuasa/anemoi-core
pip install -e anemoi-core/graphs -e anemoi-core/models -e anemoi-core/training
```

The packages report versions derived from the upstream release tags, e.g.
`anemoi-training 0.17.2.post18`, `anemoi-models 0.20.0.post18`,
`anemoi-graphs 0.9.9.post18` (the fork tag does not match the
setuptools-scm tag patterns of the packages).

## What the fork adds

| Feature | Documentation |
|---|---|
| `MixedCadenceForecaster`: one common block per model call, every dataset predicted at its own time step, per-dataset input windows, advance map, offset-based checkpoint metadata | `training/docs/user-guide/mixed-cadence.rst`, `training/docs/modules/tasks.rst` |
| Multi-dataset samples aligned by **date** instead of position (`compute_aligned_anchors`, `MultiDataset.reader_anchors`) | `training/docs/user-guide/mixed-cadence.rst` |
| Per-dataset task offsets, tendency statistics at each dataset's lead times, per-dataset `UniformTimeStepScaler` | `training/docs/user-guide/mixed-cadence.rst` |
| Per-dataset `n_step_input` / `n_step_output` in the models | `models/docs/usage/readout_models.rst` |
| `AnemoiModelEncProcDecReadouts`: decode each dataset after chosen processor layers, with gradient gate, residual mode and lead-time feature | `models/docs/usage/readout_models.rst`, `models/docs/modules/models.rst` |
| `TransferLearningLoader(dataset_map=...)`: start renamed model datasets from a checkpoint dataset | `training/docs/checkpoint_dataset_map.rst` |
| Transfer of trainable node and edge tensors by graph position when the graph changes | `training/docs/checkpoint_dataset_map.rst` |

## Commits on top of upstream

| Commit | Change |
|---|---|
| `1c2357a9b` feat(training): allow tasks to define time offsets per dataset | `compute_relative_date_indices` and the tendency statistics call the task's offset methods with `dataset_name` when the method accepts it (`get_dataset_offsets`); upstream tasks are called as before. |
| `bd3d937f3` feat(training): align multi-dataset samples by date, not by position | For single time-series readers, valid anchors are mapped to dates, the dates are intersected and each reader keeps its own positions (`MultiDataset.reader_anchors`); windows across unmarked gaps in the date array are rejected. Trajectory readers keep the positional intersection. `MultiDataset.frequency` is the first reader's. |
| `d1305ca85` feat(training): compute tendency statistics at each dataset's lead time | `statistics_tendencies` uses each dataset's output offsets. |
| `e01240ae3` feat(training): size the uniform time-step scaler per dataset | `create_scalers` receives `dataset_name`; `UniformTimeStepScaler` uses that dataset's number of output steps. |
| `c84d839d4` feat(models): accept per-dataset input and output step counts | `n_step_input` / `n_step_output` may be `{dataset: count}`; used for encoder/decoder widths, `predict_step` history and inference metadata. Models opt in through `_supports_dataset_timestep_counts`; only `AnemoiModelEncProcDec` does here. |
| `84a4dd6ae` feat(models): build tendency processors with each dataset's output steps | Single-step or step-wise tendency processors per dataset. |
| `6f2cbe7d5` feat(training): add MixedCadenceForecaster task | Task, schema (`MixedCadenceForecasterSchema`) and config group `task=mixed_cadence_forecaster`. |
| `43b55543f` feat(training): store offset-based timestep metadata for mixed-cadence checkpoints | `input_offsets`, `output_offsets`, `rollout_shift` (= block) and `advance_map` per dataset in the inference metadata; `metadata_inference.task = "mixed-cadence-forecaster"`. |
| `682559ab1` feat(training): allow explicit per-dataset input offsets in MixedCadenceForecaster | `input_offsets` for windows that skip native dates (e.g. regional `[-6h, 0h]`); offsets the previous block cannot supply are rejected. |
| `38172526f` fix(training): transfer graph-specific trainables by graph position | `remap_graph_trainables`: rows of trainable node/edge tensors matched by coordinates against the checkpoint graph; unmatched rows keep their initial values. |
| `e155230c3` refactor(models): split the encoder-processor-decoder forward into steps | `AnemoiModelEncProcDec.forward` = `_encode` → `_process` → `_decode` → `_assemble_output`; no change in behaviour. |
| `9f03339a6` feat(models): return the graph-transformer state after chosen layers | `BaseProcessor.run_layers_with_readouts`, `check_readout_layers`, `run_layer_range`; `GraphTransformerProcessor.forward(readout_layers=...)` returns `{layer: features}`. |
| `f662737a7` feat(models): add AnemoiModelEncProcDecReadouts | Read-out model, `ReadoutModelSchema`, `scale_gradient`. |
| `c5c4e7e05` feat(training): start renamed datasets from checkpoint datasets | `map_checkpoint_datasets` and `TransferLearningLoader(dataset_map=...)`. |
| `6a48ea067` fix(training): copy mapped datasets from the checkpoint as loaded | Every mapped copy reads the checkpoint as loaded, so names may be swapped in one map. |
| `0730af7d0`, `d680b8c05`, and the commit of this file: docs | The documentation pages listed above, `FORK.md` and a pointer in `README.md`. |

Tests added (805 lines): `training/tests/unit/data/test_multidataset.py`,
`test_relative_time_indices.py`, `test_datamodule_mixed_frequency.py`,
`training/tests/unit/tasks/test_mixed_cadence_forecaster.py`,
`training/tests/unit/utils/test_checkpoint_dataset_map.py`,
`test_checkpoint_graph_remap.py`, `models/tests/models/test_base_graph_model.py`,
`models/tests/layers/processor/test_processor_readouts.py`.

## Compatibility

- Upstream tasks and single-cadence configurations behave as before. Samples
  are now matched by date, so datasets with the same time step but different
  start dates are aligned correctly instead of by position.
- Configurations follow upstream 0.17.2 / 0.20.0. Coming from
  `mixed-cadence-1` (upstream of 2026-09-25):
  - checkpoint loading is configured only under `training.checkpoint`
    (`source`, `loading`, `modifiers`); `system.input.warm_start`,
    `training.transfer_learning`, `training.load_weights_only`,
    `training.run_id`, `training.fork_run_id` and
    `training.submodules_to_freeze` are rejected (upstream #1196);
  - the training dataset is map-style and needs `torchdata`; the base seed
    is `training.seed` (upstream #1433);
  - models receive the `model` sub-config, so model options such as
    `readout_layers` are read from `model.model` (upstream #1441);
  - DDP needs `static_graph: false` with `accum_grad_batches > 1`.
- Checkpoints of `mixed-cadence-1` are not loaded; retrain, or migrate with
  `anemoi-models migration sync` at your own risk (not tested here).

## Limitations

- Per-dataset step counts: only `AnemoiModelEncProcDec` and
  `AnemoiModelEncProcDecReadouts`; ensemble, transport and hierarchical models
  raise `NotImplementedError`.
- Read-outs: only `GraphTransformerProcessor`; every read-out dataset needs its
  own decoder.
- The plot callbacks assume one step count for all datasets; disable them
  (`diagnostics.plot.datasets_to_plot: []`) for mixed-cadence runs.
- Inference of mixed-cadence checkpoints needs the anemoi-inference fork.

## Verification

- Unit tests (ROCm 7.2.4 / PyTorch 2.10 image, CPU): anemoi-training 1,698
  passed (2 failures from the optional packages `wandb` and `torch_dct`, not
  related to the fork), anemoi-models `tests/models` and
  `tests/layers/processor` 77 passed.
- GPU (AMD RX 9070 XT, 2026-11-12/13): ERA5 O320 pretraining at two mesh
  levels with graph-position transfer between them, then JMA GSM (6 h) + MSM
  (3 h) fine-tuning with `dataset_map: {gsm: era5, msm: era5}` and the
  read-out model, all completed; losses and transferred row counts are
  recorded in `docs/staged_training.md` of regional_AI.
