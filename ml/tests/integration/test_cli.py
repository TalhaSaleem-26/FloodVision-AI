import json
import subprocess
import sys

import numpy as np
import pytest


def run_cli(ml_dir, *arguments):
    return subprocess.run(
        [sys.executable, "-B", str(ml_dir / "scripts" / "predict.py"), *map(str, arguments)],
        cwd=ml_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        check=False,
    )


@pytest.mark.integration
def test_cli_help_lists_input_and_output_options(ml_dir):
    completed = run_cli(ml_dir, "--help")

    assert completed.returncode == 0
    assert "--input" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--inference-only" in completed.stdout
    assert not completed.stderr


@pytest.mark.integration
def test_cli_reports_missing_bundle_as_json_on_stderr(ml_dir, tmp_path):
    output = tmp_path / "analysis"
    completed = run_cli(
        ml_dir, "--input", tmp_path / "input.tif", "--bundle-dir", tmp_path / "missing",
        "--output-dir", output, "--json",
    )

    assert completed.returncode == 1
    assert not completed.stdout
    error = json.loads(completed.stderr)
    assert error["status"] == "failed"
    assert error["error"]["type"] == "ArtifactNotFoundError"
    assert error["exit_code"] == 1
    assert not output.exists()


@pytest.mark.integration
@pytest.mark.requires_model
def test_cli_inference_only_emits_json_without_gis_exports(
    ml_dir, tmp_path, raster_factory, production_bundle_dir
):
    source = raster_factory(np.full((2, 8, 12), -18, dtype=np.float32))
    output = tmp_path / "analysis"
    completed = run_cli(
        ml_dir, "--input", source, "--bundle-dir", production_bundle_dir, "--device",
        "cpu", "--output-dir", output, "--inference-only", "--json",
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "completed"
    assert result["statistics"]["valid_pixels"] == 96
    assert result["area_analytics"] is None
    assert result["vectorization"] is None
    assert {entry.name for entry in output.iterdir()} == {"analysis_report.json", "manifest.json"}


@pytest.mark.integration
@pytest.mark.requires_model
def test_cli_accepts_explicit_integer_memory_budget(
    ml_dir, tmp_path, raster_factory, production_bundle_dir
):
    source = raster_factory(np.full((2, 8, 12), -18, dtype=np.float32))
    output = tmp_path / "analysis"
    completed = run_cli(
        ml_dir, "--input", source, "--bundle-dir", production_bundle_dir,
        "--device", "cpu", "--output-dir", output, "--inference-only", "--json",
        "--memory-budget-mib", "128",
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "completed"
    assert (output / "manifest.json").is_file()


@pytest.mark.integration
@pytest.mark.parametrize("value", ["1.5", "0", "-1"])
def test_cli_rejects_invalid_memory_budgets(ml_dir, tmp_path, value):
    output = tmp_path / "analysis"
    completed = run_cli(
        ml_dir, "--input", tmp_path / "input.tif", "--output-dir", output,
        "--memory-budget-mib", value, "--json",
    )

    assert not completed.stdout
    assert not output.exists()
    if value == "1.5":
        assert completed.returncode == 2
        assert "--memory-budget-mib: invalid int value" in completed.stderr
    else:
        assert completed.returncode == 1
        assert json.loads(completed.stderr)["error"] == {
            "type": "InputValidationError",
            "message": "memory_budget_mib must be positive.",
        }
