# Phase 43 testing report

Branch `testing/phase-43`, baseline commit `bd3dd72`. All 99 original tests passed before the first test edit. The first attempt had three test fixture setup errors because Windows denied access to pytest's default temporary directory. A clean rerun using a fresh temporary directory established the baseline below.

| Run | Passed | Failed | Skipped | Expected failures | Statement coverage | Branch coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original baseline | 99 | 0 | 0 | 0 | 13.49% (354/2,624) | 4.17% (33/792) |
| Initial tiling and GeoTIFF milestone | 122 | 0 | 0 | 0 | 24.89% (653/2,624) | 13.13% (104/792) |
| Accepted intermediate milestone | 157 | 0 | 0 | 1 | 76.26% (2,001/2,624) | 56.57% (448/792) |
| Final | 235 | 0 | 0 | 0 | 88.80% (2,330/2,624) | 70.08% (555/792) |

The final Windows CPU run took 56.27 seconds and had zero errors or unexpected passes. Phase 43 added **136 collected cases** over the original 99, including **77** during final completion. The existing expected-failure case now passes and is counted once. The statement target was at least 80%.

Coverage outputs include [terminal output with missing lines](phase43-terminal.txt), [HTML](phase43-htmlcov/index.html), [XML](phase43-coverage.xml), [final JSON](phase43-final.json), and [baseline JSON](phase43-baseline.json).

## Master plan completion checklist

- [x] Losses and metrics. Factory options and unknown names, extreme logits, single valid pixels, finite gradients, invalid pixel exclusion, and accumulated segmentation metrics.
- [x] Dataset and preprocessing. Synthetic GeoTIFF image and mask loading, split initialization, normalization, sentinels and nonfinite values, seeded spatial alignment, and rejection of missing or malformed inputs.
- [x] Training. Real CPU optimizer updates and evaluation with a one-weight model, valid-pixel-weighted loss, all-invalid batches, nonfinite loss, early stopping, checkpoint restoration, best and last checkpoint selection, CSV logging, trainer orchestration, and Python, NumPy and PyTorch seeding.
- [x] Inference core. Manifest hashes and model compatibility, device fallback, input validation, production preprocessor, predictor, postprocessor, tile geometry and blending, memory and disk parity, automatic storage selection, and resource limits.
- [x] GIS and pipeline. Raster and GeoJSON exports, geometry simplification and fidelity, area accounting, artifact integrity, rollback after staged writes, config validation, result serialization, pipeline integration, and CLI subprocess contracts.
- [x] Reference regression. The existing Mekong test retains the CPU production bundle, source raster and notebook-derived expected results.
- [x] Coverage and protection. Statement and branch auditing, warning investigation, production manifest verification, and protected-path Git review.

## Approved CLI fix

```diff
         "--memory-budget-mib",
-        type=float,
+        type=int,
         default=128
```

The explicit value `128` now completes an inference-only subprocess run. A fractional value returns argparse exit code 2. Zero and negative values reach the existing `PipelineConfig` positivity check and return structured errors with exit code 1. The default remains 128. The targeted CLI suite passed all seven tests.

## Warning investigation

- **Affine.** The accepted milestone logged 2,255 instances of ``PendingDeprecationWarning: Use `@` matmul instead of `*` mul operator for matrix multiplication``. The installed affine 3.0.1 dependency, required by Rasterio 1.5.2, emits it in `Affine.__mul__` with `stacklevel=2`. The warning therefore points to the project call sites in `area_analyzer.py` (168, 170, 213, 215, 217 and 221) and `geospatial.py` (236–239 and 280 in the milestone, with the same operator also at 174). Area calculations apply four corner transforms per raster row, accounting for most repetitions. The method delegates to `__matmul__` in the installed version, and no functional difference was observed in the verified run. A separately approved remediation should replace the project call sites with `@` and rerun the area, georeferencing and Mekong checks.
- **Albumentations.** Importing Albumentations 2.0.8 for the dataset tests attempted a version lookup. Its `check_version.py:147` emitted `UserWarning: Error fetching version info <urlopen error [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions>`. The lookup is separate from augmentation, and the seeded alignment test ran successfully. The final invocation used pytest's standard warning reporting and recorded 2,262 affine warnings and one Albumentations warning.

## Remaining coverage gaps and limitations

Detailed uncovered lines are in the terminal and HTML reports. The remaining paths include CUDA execution, several filesystem and model load failure modes, malformed metadata permutations, and rare geometry and tile I/O errors. CPU tests cover device selection without executing CUDA kernels. The CLI lives in `scripts/` and its subprocess tests sit outside the `floodvision` coverage denominator. Training exercises only a synthetic one-weight model. The earlier explicit-memory-budget defect is resolved. No other production defect was reproduced in the final suite.

## Workspace and manifest verification

The production bundle verified the size and SHA-256 for all four required manifest entries, including the production weights. The model loaded on CPU in evaluation mode and the Mekong reference used it successfully. The reference GeoTIFF and production bundle were accessed read-only. Tests wrote training checkpoints, predictions, and staged artifacts only in pytest temporary directories.

Git diff and path-scoped status identified `ml/scripts/predict.py` as the only production source change. They showed no changes to protected tracked research, dataset, notebook, configuration, checkpoint, artifact, backend, frontend or package source paths. The production bundle and reference raster were also inspected separately because Git does not validate ignored files. Preexisting ignored output directories with Windows access restrictions remain outside the inspection. The Git branch remained `testing/phase-43` at `bd3dd72`. No commits, pushes or merges were performed.

During the first Phase 43 milestone, pytest-cov cleanup removed the preexisting ignored `ml/.coverage` data file. The current run used a fresh temporary coverage data path.

## Modified file inventory

- `ml/tests/inference/test_accumulator.py`
- `ml/tests/inference/test_area_analyzer.py`
- `ml/tests/inference/test_artifact_manager.py`
- `ml/tests/inference/test_geometry_fidelity.py`
- `ml/tests/inference/test_model_loader.py`
- `ml/tests/inference/test_postprocessor.py`
- `ml/tests/inference/test_tiling.py`
- `ml/tests/inference/test_validator.py`
- `ml/tests/inference/test_vectorizer.py`
- `ml/tests/integration/test_cli.py`
- `ml/tests/integration/test_pipeline.py`
- `ml/tests/regression/test_reference_prediction.py`
- `ml/tests/unit/test_metrics.py`
- `ml/tests/unit/test_preprocessing.py`
- `ml/tests/inference/test_geometry_exports.py`
- `ml/tests/inference/test_pipeline_contracts.py`
- `ml/tests/inference/test_prediction_engine.py`
- `ml/tests/inference/test_raster_exporter.py`
- `ml/tests/inference/test_tile_reader.py`
- `ml/tests/unit/test_dataset.py`
- `ml/tests/unit/test_loss_factory.py`
- `ml/tests/unit/test_training.py`
- `ml/scripts/predict.py`
- `ml/PHASE_43_TESTING_REPORT.md`
- `ml/phase43-baseline.json`
- `ml/phase43-final.json`
- `ml/phase43-coverage.xml`
- `ml/phase43-terminal.txt`
- `ml/phase43-htmlcov/` contains the generated HTML pages, coverage status, and static assets.

## Module-wise coverage

The paths are relative to `ml/src/floodvision/`. A dash indicates a zero denominator.

| Module | Statements | Branches |
| --- | ---: | ---: |
| `__init__.py` | — | — |
| `datasets/__init__.py` | — | — |
| `datasets/flood_dataset.py` | 100.00% (43/43) | 100.00% (4/4) |
| `evaluation/__init__.py` | — | — |
| `inference/__init__.py` | — | — |
| `inference/accumulator.py` | 85.23% (127/149) | 68.18% (45/66) |
| `inference/area_analyzer.py` | 84.76% (89/105) | 58.33% (21/36) |
| `inference/artifact_manager.py` | 87.50% (70/80) | 66.67% (20/30) |
| `inference/exceptions.py` | 100.00% (11/11) | — |
| `inference/geometry_fidelity.py` | 89.80% (44/49) | 64.29% (9/14) |
| `inference/geometry_processor.py` | 88.03% (125/142) | 58.82% (20/34) |
| `inference/geometry_utils.py` | 79.31% (23/29) | 66.67% (12/18) |
| `inference/geospatial.py` | 76.54% (62/81) | 43.75% (7/16) |
| `inference/model_loader.py` | 78.29% (137/175) | 64.29% (45/70) |
| `inference/pipeline.py` | 89.32% (92/103) | 69.23% (18/26) |
| `inference/pipeline_config.py` | 100.00% (68/68) | 97.22% (35/36) |
| `inference/pipeline_result.py` | 98.25% (56/57) | 83.33% (5/6) |
| `inference/postprocessor.py` | 80.00% (60/75) | 63.33% (19/30) |
| `inference/predictor.py` | 80.56% (58/72) | 68.75% (22/32) |
| `inference/preprocessor.py` | 78.13% (75/96) | 62.50% (25/40) |
| `inference/raster_exporter.py` | 90.24% (74/82) | 77.50% (31/40) |
| `inference/schemas.py` | 95.06% (77/81) | 0.00% (0/2) |
| `inference/tile_reader.py` | 90.41% (66/73) | 75.00% (9/12) |
| `inference/tiled_predictor.py` | 93.33% (112/120) | 79.41% (27/34) |
| `inference/tiling.py` | 90.10% (91/101) | 75.00% (21/28) |
| `inference/validator.py` | 77.17% (71/92) | 63.79% (37/58) |
| `inference/vector_exporter.py` | 83.70% (77/92) | 62.50% (25/40) |
| `inference/vectorizer.py` | 86.11% (93/108) | 65.79% (25/38) |
| `losses/__init__.py` | — | — |
| `losses/combined.py` | 100.00% (15/15) | — |
| `losses/dice.py` | 94.44% (17/18) | 75.00% (3/4) |
| `losses/factory.py` | 100.00% (13/13) | 100.00% (6/6) |
| `losses/masked_bce.py` | 94.44% (17/18) | 75.00% (3/4) |
| `metrics/__init__.py` | — | — |
| `metrics/segmentation.py` | 100.00% (33/33) | 100.00% (4/4) |
| `models/__init__.py` | — | — |
| `models/baseline_cnn.py` | 100.00% (34/34) | — |
| `models/blocks.py` | 100.00% (35/35) | — |
| `models/factory.py` | 100.00% (12/12) | 100.00% (2/2) |
| `models/mobilenet_unet.py` | 91.49% (43/47) | 50.00% (2/4) |
| `models/resnet_unet.py` | 90.70% (39/43) | 50.00% (2/4) |
| `models/unet.py` | 100.00% (34/34) | — |
| `preprocessing/__init__.py` | — | — |
| `preprocessing/normalization.py` | 100.00% (25/25) | 100.00% (6/6) |
| `preprocessing/sar.py` | 100.00% (17/17) | 100.00% (4/4) |
| `preprocessing/transforms.py` | 100.00% (5/5) | — |
| `training/__init__.py` | — | — |
| `training/checkpoint.py` | 100.00% (9/9) | — |
| `training/early_stopping.py` | 100.00% (18/18) | 100.00% (6/6) |
| `training/engine.py` | 100.00% (59/59) | 100.00% (16/16) |
| `training/logger.py` | 100.00% (13/13) | 100.00% (2/2) |
| `training/trainer.py` | 100.00% (70/70) | 85.71% (12/14) |
| `utils/__init__.py` | — | — |
| `utils/config.py` | 100.00% (11/11) | 100.00% (4/4) |
| `utils/seed.py` | 90.91% (10/11) | 50.00% (1/2) |

## Reproduce the final run

From `ml/` in PowerShell, using the existing virtual environment.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$phaseTemp = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
$env:COVERAGE_FILE = "$phaseTemp.cov"
.venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp $phaseTemp --tb=short --cov=floodvision --cov-branch --cov-report=term-missing --cov-report=html:phase43-htmlcov --cov-report=xml:phase43-coverage.xml --cov-report=json:phase43-final.json | Tee-Object -FilePath phase43-terminal.txt
```
