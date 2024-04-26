"""Integration test for running and fetching an experiment from cache."""

import json
from importlib.resources import files
from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest
from jsonschema import validate

from autotuning_methodology.curves import StochasticOptimizationAlgorithm
from autotuning_methodology.experiments import (
    ResultsDescription,
    execute_experiment,
    get_args_from_cli,
    get_experiment_schema_filepath,
)

# get the path to the package
package_path = Path(files("autotuning_methodology")).parent.parent
# package_path = ""

# setup file paths
mockfiles_path_root = package_path / Path("tests/autotuning_methodology/integration/mockfiles/")
mockfiles_path_source = mockfiles_path_root / "mock_gpu.json"
mockfiles_path = mockfiles_path_root
experiment_filepath_test = mockfiles_path / "test.json"
assert experiment_filepath_test.exists()
kernel_id = "mocktest_kernel_convolution"
cached_visualization_path = package_path / Path(f"cached_data_used/visualizations/test_run_experiment/{kernel_id}")
cached_visualization_file = cached_visualization_path / "mock_GPU_random_sample_10_iter.npz"
cached_visualization_imported_path = package_path / Path(
    f"cached_data_used/visualizations/test_output_file_writer/{kernel_id}"
)
cached_visualization_imported_file = cached_visualization_imported_path / "mock_GPU_ktt_profile_searcher.npz"
normal_cachefiles_path = package_path / Path(f"cached_data_used/cachefiles/{kernel_id}")
normal_cachefile_destination = normal_cachefiles_path / "mock_gpu.json"
experiment_import_filepath_test = mockfiles_path / "test_import_runs.json"
assert experiment_import_filepath_test.exists()
import_runs_source_path = mockfiles_path / "import_runs"
import_runs_path = package_path / Path("cached_data_used/import_runs")
import_runs_filepaths: list[Path] = list()


def _remove_dir(path: Path):
    """Utility function for removing a directory and the contained files."""
    assert path.exists()
    for sub in path.iterdir():
        sub.unlink()
    path.rmdir()


def setup_module():
    """Setup of the test, creates / copies files where necessary."""
    assert mockfiles_path_source.exists()
    normal_cachefiles_path.mkdir(parents=True, exist_ok=True)
    assert normal_cachefiles_path.exists()
    normal_cachefile_destination.write_text(mockfiles_path_source.read_text())
    assert normal_cachefile_destination.exists()
    # cached_visualization_path.mkdir(parents=True, exist_ok=True)
    # assert cached_visualization_path.exists()
    # copy the import run test files to the import run folder
    assert import_runs_source_path.exists()
    import_runs_path.mkdir(parents=True, exist_ok=True)
    assert import_runs_path.exists()
    for import_run_file in import_runs_source_path.iterdir():
        if not import_run_file.is_file():
            continue
        assert import_run_file.exists()
        destination = Path(import_runs_path / import_run_file.name).resolve()
        import_runs_filepaths.append(Path(copyfile(str(import_run_file), str(destination))))
        assert destination == import_runs_filepaths[-1].resolve()
        assert destination.exists() and destination.is_file()


def teardown_module():
    """Teardown of the tests, removes files where necessary."""
    if normal_cachefile_destination.exists():
        normal_cachefile_destination.unlink()
    _remove_dir(normal_cachefiles_path)
    if cached_visualization_file.exists():
        cached_visualization_file.unlink()
    _remove_dir(cached_visualization_path)
    if cached_visualization_imported_file.exists():
        cached_visualization_imported_file.unlink()
    _remove_dir(cached_visualization_imported_path)
    # delete the import run test files from the import run folder
    for import_run_file in import_runs_filepaths:
        import_run_file.unlink()


def test_CLI_input():
    """Test bad and good CLI inputs."""
    # improper input 1
    with pytest.raises(SystemExit) as e:
        dummy_args = ["-dummy_arg=option"]
        get_args_from_cli(dummy_args)
    assert e.type == SystemExit
    assert e.value.code == 2

    # improper input 2
    with pytest.raises(ValueError, match="Invalid '-experiment' option"):
        get_args_from_cli([""])

    # proper input
    args = get_args_from_cli(["bogus_filename"])
    assert args == "bogus_filename"


def test_bad_experiment():
    """Attempting to run a non-existing experiment file should raise a clear error."""
    experiment_filepath = "bogus_filename"
    with pytest.raises(AssertionError, match=" does not exist, attempted path: "):
        execute_experiment(experiment_filepath, profiling=False)

    experiment_filepath = "experiment_files/bogus_filename"
    with pytest.raises(AssertionError, match=" does not exist, attempted path: "):
        execute_experiment(experiment_filepath, profiling=False)


def test_run_experiment_bad_kernel_path():
    """Run an experiment with a bad kernel path."""
    experiment_filepath = str(mockfiles_path / "test_bad_kernel_path.json")
    with pytest.raises(FileNotFoundError, match="No such path"):
        execute_experiment(experiment_filepath, profiling=False)


@pytest.fixture(scope="session")
def test_run_experiment():
    """Run a dummy experiment."""
    assert normal_cachefile_destination.exists()
    if cached_visualization_file.exists():
        cached_visualization_file.unlink()
    assert not cached_visualization_file.exists()
    (experiment, strategies, results_descriptions) = execute_experiment(str(experiment_filepath_test), profiling=False)
    validate_experiment_results(experiment, strategies, results_descriptions)


@pytest.mark.usefixtures("test_run_experiment")
def test_cached_experiment():
    """Retrieve a cached experiment."""
    assert normal_cachefiles_path.exists()
    assert normal_cachefile_destination.exists()
    assert cached_visualization_path.exists()
    assert cached_visualization_file.exists()
    (experiment, strategies, results_descriptions) = execute_experiment(str(experiment_filepath_test), profiling=False)
    validate_experiment_results(experiment, strategies, results_descriptions)


def test_import_run_experiment():
    """Import runs from an experiment."""
    assert import_runs_path.exists()
    (experiment, strategies, results_descriptions) = execute_experiment(
        str(experiment_import_filepath_test), profiling=False
    )
    assert cached_visualization_imported_path.exists()
    assert cached_visualization_imported_file.exists()
    validate_experiment_results(experiment, strategies, results_descriptions)


@pytest.mark.usefixtures("test_run_experiment")
def test_curve_instance():
    """Test a Curve instance."""
    # setup the test
    (experiment, strategies, results_descriptions) = execute_experiment(str(experiment_filepath_test), profiling=False)
    kernel_name = experiment["kernels"][0]
    gpu_name = experiment["GPUs"][0]
    strategy_name = strategies[0]["name"]
    results_description = results_descriptions[gpu_name][kernel_name][strategy_name]
    curve = StochasticOptimizationAlgorithm(results_description)

    # set the input data
    x_1d = np.array([0.1, 1.1, 2.1, 3.1])
    y_1d = np.log(x_1d)
    x_test_1d = np.array([1.5, 2.5, 3.5, 4.5])
    x = x_1d.reshape((-1, 1))
    y = y_1d.reshape((-1, 1))
    confidence_level = 0.95

    # test the prediction intervals
    pred_interval = curve._get_prediction_interval_separated(x, y, x_test_1d, confidence_level)
    assert pred_interval.shape == (4, 3)
    assert not np.any(np.isnan(pred_interval))
    pred_interval = curve._get_prediction_interval_bagging(x_1d, y_1d, x_test_1d, confidence_level, num_repeats=3)
    assert pred_interval.shape == (4, 3)
    assert not np.any(np.isnan(pred_interval))
    methods = ["inductive_conformal"]  # extend with: ["conformal", "normalized_conformal", "mondrian_conformal"
    for method in methods:
        pred_interval = curve._get_prediction_interval_conformal(x_1d, y_1d, x_test_1d, confidence_level, method=method)
        assert pred_interval.shape == (4, 2)
        assert not np.any(np.isnan(pred_interval))


def validate_experiment_results(
    experiment,
    strategies,
    results_descriptions,
):
    """Validate the types and contents returned from an experiment."""
    assert isinstance(experiment, dict)
    assert isinstance(strategies, list)
    assert isinstance(results_descriptions, dict)

    # validate the contents
    schemafilepath = get_experiment_schema_filepath()
    with open(schemafilepath) as schemafile:
        schema = json.load(schemafile)
        validate(instance=experiment, schema=schema)
    kernel_name = experiment["kernels"][0]
    gpu_name = experiment["GPUs"][0]
    assert len(strategies) == 1
    strategy_name = strategies[0]["name"]
    assert isinstance(results_descriptions[gpu_name][kernel_name][strategy_name], ResultsDescription)
