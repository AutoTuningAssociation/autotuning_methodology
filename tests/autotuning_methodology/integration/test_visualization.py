"""Integration test for visualization and quantification."""

from pathlib import Path

from autotuning_methodology.visualize_experiments import Visualize

# setup file paths
mockfiles_path_root = Path("tests/autotuning_methodology/integration/mockfiles/")
mockfiles_path_source = mockfiles_path_root / "mock_gpu.json"
mockfiles_path = Path("..") / mockfiles_path_root
experiment_id = "mocktest_kernel_convolution"
experiment_title = f"{experiment_id}_on_mock_GPU"
cached_visualization_path = Path(f"cached_data_used/visualizations/test_run_experiment/{experiment_id}")
cached_visualization_file = cached_visualization_path / "mock_GPU_random_sample_10_iter.npz"
normal_cachefiles_path = Path(f"cached_data_used/cachefiles/{experiment_id}")
normal_cachefile_destination = normal_cachefiles_path / "mock_gpu.json"
plot_path = Path("generated_plots")
plot_path_fevals = plot_path / f"{experiment_title}_fevals.png"
plot_path_time = plot_path / f"{experiment_title}_time.png"
plot_path_aggregated = plot_path / "aggregated.png"
plot_path_split_times_fevals = plot_path / f"{experiment_title}_split_times_fevals.png"
plot_path_split_times_time = plot_path / f"{experiment_title}_split_times_time.png"
plot_path_split_times_bar = plot_path / f"{experiment_title}_split_times_bar.png"
plot_path_baselines_comparison = plot_path / f"{experiment_title}_baselines.png"
plot_filepaths = [
    plot_path_fevals,
    plot_path_time,
    plot_path_aggregated,
    plot_path_split_times_fevals,
    plot_path_split_times_time,
    plot_path_split_times_bar,
    plot_path_baselines_comparison,
]


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
    # plot_path.mkdir(parents=True, exist_ok=True)
    # assert plot_path.exists()


def teardown_module():
    """Teardown of the tests, removes files where necessary."""
    if normal_cachefile_destination.exists():
        normal_cachefile_destination.unlink()
    _remove_dir(normal_cachefiles_path)
    if cached_visualization_file.exists():
        cached_visualization_file.unlink()
    _remove_dir(cached_visualization_path)
    if plot_path.exists():
        for plot_filepath in plot_filepaths:
            plot_filepath.unlink(missing_ok=True)


def test_visualize_experiment():
    """Visualize a dummy experiment."""
    assert normal_cachefile_destination.exists()
    if cached_visualization_file.exists():
        cached_visualization_file.unlink()
    assert not cached_visualization_file.exists()
    experiment_filepath = str(mockfiles_path / "test.json")
    Visualize(
        experiment_filepath,
        save_figs=True,
        save_extra_figs=True,
        continue_after_comparison=True,
        compare_extra_baselines=True,
    )
    for plot_filepath in plot_filepaths:
        assert plot_filepath.exists(), f"{plot_filepath} does not exist"
