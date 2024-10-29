"""Integration test for visualization."""

from pathlib import Path

import pytest
from test_run_experiment import (
    _remove_dir,
    cached_visualization_file,
    experiment_filepath_test,
    experiment_path,
    kernel_id,
    mockfiles_path_source,
    normal_cachefile_destination,
    normal_cachefiles_path,
    plot_path,
)

from autotuning_methodology.visualize_experiments import Visualize

# setup file paths
experiment_title = f"{kernel_id}_on_mock_GPU"
plot_path_fevals = plot_path / f"{experiment_title}_fevals.png"
plot_path_time = plot_path / f"{experiment_title}_time.png"
plot_path_heatmap = plot_path / "random_sample_10_iter_heatmap_applications_gpus.png"
plot_path_heatmap_time = plot_path / "random_sample_10_iter_heatmap_time_searchspaces.png"
plot_path_aggregated = plot_path / "aggregated.png"
plot_path_split_times_fevals = plot_path / f"{experiment_title}_split_times_fevals.png"
plot_path_split_times_time = plot_path / f"{experiment_title}_split_times_time.png"
plot_path_split_times_bar = plot_path / f"{experiment_title}_split_times_bar.png"
plot_path_baselines_comparison = plot_path / f"{experiment_title}_baselines.png"
plot_filepaths: list[Path] = [
    plot_path_fevals,
    plot_path_time,
    plot_path_heatmap,
    plot_path_heatmap_time,
    plot_path_aggregated,
    plot_path_split_times_fevals,
    plot_path_split_times_time,
    plot_path_split_times_bar,
    plot_path_baselines_comparison,
]


def setup_module():
    """Setup of the test, creates / copies files where necessary."""
    assert mockfiles_path_source.exists()
    normal_cachefiles_path.mkdir(parents=True, exist_ok=True)
    assert normal_cachefiles_path.exists()
    normal_cachefile_destination.write_text(mockfiles_path_source.read_text())
    assert normal_cachefile_destination.exists()


def teardown_module():
    """Teardown of the tests, removes files where necessary."""
    if normal_cachefile_destination.exists():
        normal_cachefile_destination.unlink()
    _remove_dir(normal_cachefiles_path)
    if plot_path.exists():
        for plot_filepath in plot_filepaths:
            plot_filepath.unlink(missing_ok=True)
        plot_path.rmdir()
    _remove_dir(experiment_path)


@pytest.mark.dependency()
def test_visualize_experiment():
    """Visualize a dummy experiment."""
    assert normal_cachefile_destination.exists()
    if cached_visualization_file.exists():
        cached_visualization_file.unlink()
    assert not cached_visualization_file.exists()
    experiment_filepath = str(experiment_filepath_test)
    Visualize(
        experiment_filepath,
        save_figs=True,
        save_extra_figs=True,
        continue_after_comparison=True,
        compare_extra_baselines=True,
    )


@pytest.mark.dependency(depends=["test_visualize_experiment"])
@pytest.mark.parametrize("plot_filepath", plot_filepaths)
def test_visualized_plot(plot_filepath: Path):
    """Test whether valid plots have been produced."""
    for plot_filepath in plot_filepaths:
        assert (
            plot_filepath.exists()
        ), f"{plot_filepath} does not exist, files in folder: {[f.name for f in plot_filepath.parent.iterdir() if f.is_file()]}"
