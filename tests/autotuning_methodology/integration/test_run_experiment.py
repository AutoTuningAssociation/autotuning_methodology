import pytest
from pathlib import Path
from autotuning_methodology.experiments import execute_experiment, get_args_from_cli

mockfiles_path = Path("../tests/autotuning_methodology/integration/mockfiles/")
cached_visualization_path = Path("cached_data_used/visualizations/test_run_experiment/mocktest_kernel_convolution/mock_GPU_random_sample_100_iter.npz")


def test_CLI_input():
    """Test bad and good CLI inputs"""

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
    """Attempting to run a non-existing experiment file should raise a clear error"""
    experiment_filepath = "bogus_filename"
    with pytest.raises(AssertionError, match=" does not exist, attempted path: "):
        results = execute_experiment(experiment_filepath, profiling=False)

    experiment_filepath = "experiment_files/bogus_filename"
    with pytest.raises(AssertionError, match=" does not exist, attempted path: "):
        results = execute_experiment(experiment_filepath, profiling=False)


def test_run_experiment_bad_kernel_path():
    """Run an experiment with a bad kernel path"""
    experiment_filepath = str(mockfiles_path / "test_bad_kernel_path.json")
    with pytest.raises(FileNotFoundError, match="No such path"):
        execute_experiment(experiment_filepath, profiling=False)


def test_run_experiment():
    """Run a dummy experiment"""
    if cached_visualization_path.exists():
        # os.remove  # TODO
    assert not cached_visualization_path.exists()
    experiment_filepath = str(mockfiles_path / "test.json")
    results = execute_experiment(experiment_filepath, profiling=False)
    print(results)
    assert False
    # with pytest.raises()


def test_cached_experiment():
    """Retrieve a cached experiment"""
    assert cached_visualization_path.exists()
