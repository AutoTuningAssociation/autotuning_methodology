"""Integration test for results reporting."""

from test_run_experiment import (
    _remove_dir,
    cached_visualization_file,
    experiment_filepath_test,
    kernel_id,
    mockfiles_path_source,
    normal_cachefile_destination,
    normal_cachefiles_path,
)

from autotuning_methodology.experiments import get_experiment, get_experimental_groups
from autotuning_methodology.report_experiments import get_strategy_scores

# setup file paths
experiment_title = f"{kernel_id}_on_mock_GPU"


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


def test_visualize_experiment():
    """Report on a dummy experiment."""
    # make sure the paths exists
    assert normal_cachefile_destination.exists()
    if cached_visualization_file.exists():
        cached_visualization_file.unlink()
    assert not cached_visualization_file.exists()

    # get the experiment details
    experiment_filepath = str(experiment_filepath_test)
    experiment = get_experiment(experiment_filepath)
    strategies = get_experimental_groups(experiment)  # TODO fix this test that used to use get_strategies

    # get the scores
    strategies_scores = get_strategy_scores(experiment_filepath)

    # validate the results
    assert len(strategies_scores) == len(strategies)
    for strategy in strategies:
        assert strategy["name"] in strategies_scores
    for strategy, score in strategies_scores.items():
        assert isinstance(score, dict)
        assert isinstance(strategy, str)
        assert "score" in score
        assert "error" in score
        for value in score.values():
            assert isinstance(value, float)
