"""Main experiments code."""

from __future__ import annotations  # for correct nested type hints e.g. list[str], tuple[dict, str]

import json
from argparse import ArgumentParser
from importlib.resources import files
from math import ceil
from os import getcwd, makedirs
from pathlib import Path

from jsonschema import ValidationError, validate

from autotuning_methodology.caching import ResultsDescription
from autotuning_methodology.runner import collect_results, convert_KTT_output_to_standard
from autotuning_methodology.searchspace_statistics import SearchspaceStatistics

PACKAGE_ROOT = Path(__file__).parent.parent.parent


def get_args_from_cli(args=None) -> str:
    """Set the Command Line Interface arguments definitions, get and return the argument values.

    Args:
        args: optional list of arguments for testing without CLI interaction. Defaults to None.

    Raises:
        ValueError: on invalid argument.

    Returns:
        The filepath to the experiments file.
    """
    cli = ArgumentParser()
    cli.add_argument(
        "experiment", type=str, help="The experiment setup json file to execute, see experiments/template.json"
    )
    args = cli.parse_args(args)
    filepath: str = args.experiment
    if filepath is None or filepath == "":
        raise ValueError("Invalid '--experiment' option. Run 'visualize_experiments.py -h' to read more.")
    return filepath


def get_experiment_schema_filepath():
    """Obtains and checks the filepath to the JSON schema.

    Returns:
        the filepath to the schema in Traversable format.
    """
    schemafile = files("autotuning_methodology").joinpath("schema.json")
    assert schemafile.is_file(), f"Path to schema.json does not exist, attempted path: {schemafile}"
    return schemafile


def make_and_check_path(filename: str, parent=None, extension=None) -> Path:
    filename_path = Path(filename)
    if filename_path.is_absolute() is False and parent is not None:
        filename_path = PACKAGE_ROOT / Path(parent).joinpath(filename).resolve()
    if filename_path.exists():
        return filename_path
    # try and add extension
    if extension is None:
        raise FileNotFoundError(f"{filename_path} does not exist.")
    filename_path = Path(str(filename_path) + extension)
    if filename_path.exists():
        return filename_path
    raise FileNotFoundError(f"{filename_path} does not exist.")


def get_experiment(filename: str) -> dict:
    """Validates and gets the experiment from the experiments .json file.

    Args:
        filename: path to the experiments .json file.

    Returns:
        Experiment dictionary object.
    """
    # get the path to the experiment file
    # folder_name = "experiment_files"
    extension = ".json"
    if not filename.endswith(extension):
        filename = filename + extension
    path = Path(filename)
    # if not filename.startswith(folder_name + "/"):
    #     path = folder / filename
    # else:
    #     path = Path(filename)
    assert path.exists(), f"Path to experiment file does not exist, attempted path: {path}, CWD: {getcwd()}"

    # get the path to the schema
    schemafile = get_experiment_schema_filepath()

    # open the experiment file and validate using the schema file
    with open(path, "r", encoding="utf-8") as file, open(schemafile, "r", encoding="utf-8") as schemafile:
        schema = json.load(schemafile)
        experiment: dict = json.load(file)
        try:
            validate(instance=experiment, schema=schema)
            return experiment
        except ValidationError as e:
            print(e)
            raise ValidationError("Experiment file does not comply with schema")


def get_experimental_groups(experiment: dict) -> list[dict]:
    """Prepares all the experimental groups as all combinations of application and gpus (from experimental_groups_defaults) and big experimental groups from setup file (experimental_groups, usually search methods). Check additional settings for each experimental group. Prepares the directory structure for the whole experiment.

    Args:
        experiment: the experiment dictionary object.

    Returns:
        The experimental groups in the experiment dictionary object.
    """
    experimental_groups_defaults = experiment["experimental_groups_defaults"]
    search_strategies = experiment["search_strategies"]

    # set up the directory structure
    experiment["parent_folder_absolute_path"] = Path(experiment["parent_folder"]).resolve()
    # if folder "run" does not exist, create
    makedirs(experiment["parent_folder_absolute_path"].joinpath("run"), exist_ok=True)
    makedirs(experiment["parent_folder_absolute_path"].joinpath("setup"), exist_ok=True)

    # create folders for each experimental group from file
    for strategy in search_strategies:
        makedirs(experiment["parent_folder_absolute_path"].joinpath("run").joinpath(strategy["name"]), exist_ok=True)

    # generate all experimental groups
    # with applications and gpus provided in experimental_groups_defaults
    # and search strategies provided in search_strategies
    all_experimental_groups = generate_all_experimental_groups(
        search_strategies, experimental_groups_defaults, experiment["parent_folder_absolute_path"]
    )

    # additional check beyond validation
    # if every experimental group has autotuner set
    # set autotuner_path to default installation if not set by the user
    for group in all_experimental_groups:
        if group.get("autotuner") is None:
            raise KeyError(
                "Property 'autotuner' must be set for all groups, either in experimental_groups_defaults or in experimental_groups. It is not set for",
                group["full_name"],
            )
        if group["autotuner"] == "KTT":
            if group["samples"] != 1:
                raise NotImplementedError(
                    "KTT currently supports only one sample per run and output. Please set samples=1 for group['full_name']."
                )
            if group.get("autotuner_path") is None:
                raise NotImplementedError(
                    "Default autotuner_path is not supported yet for KTT, please set autotuner_path for ",
                    group["full_name"]
                    + " to directory with KttTuningLauncher and pyktt.so, e.g. /home/user/KTT/Build/x86_64_Release.",
                )
            elif Path(group["autotuner_path"]).exists() is False:
                raise FileNotFoundError(
                    f"Directory {group['autotuner_path']} does not exists. Try setting the absolute path."
                )
            elif Path(group["autotuner_path"]).joinpath("KttTuningLauncher").exists() is False:
                raise FileNotFoundError(
                    f"Directory {group['autotuner_path']} does not contain KttTuningLauncher. Have you used --tuning-loader when premaking KTT?"
                )
            elif Path(group["autotuner_path"]).joinpath("pyktt.so").exists() is False:
                raise FileNotFoundError(
                    f"Directory {group['autotuner_path']} does not contain pyktt.so. Have you used --python when premaking KTT?"
                )
            # TODO make and set default autotuner path

    return all_experimental_groups


def generate_all_experimental_groups(
    search_strategies: list[dict], experimental_groups_defaults: dict, parent_folder_path: Path
) -> list[dict]:
    """Generates all experimental groups for the experiment as a combination of given applications, gpus and search strategies from experiments setup file.

    Args:
        search_strategies: list of dictionaries with settings for various search strategies from experiments setup file, section search_strategies.
        experimental_groups_defaults: a dictionary with default settings for experimental groups from experiments setup file, section experimental_groups_defaults.
        parent_folder_path: path to experiment parent folder that stores all files generated in the experiment.

    Returns:
        A list of dictionaries, one for each experimental group.
    """
    experimental_groups = []

    for gpu in experimental_groups_defaults["gpus"]:
        for application in experimental_groups_defaults["applications"]:
            for strategy in search_strategies:
                group = strategy

                for default in experimental_groups_defaults:
                    if default not in group and default not in [
                        "applications",
                        "gpus",
                        "pattern_for_full_search_space_filenames",
                    ]:
                        group[default] = experimental_groups_defaults[default]

                group["full_name"] = "_".join([gpu, application["name"], group["name"]])

                group["gpu"] = gpu
                group["application_name"] = application["name"]

                group["application_folder"] = Path(application["folder"])
                group["application_input_file"] = make_and_check_path(
                    application["input_file"], application["folder"], None
                )
                group["input_file"] = parent_folder_path.joinpath("setup").joinpath(
                    "_".join([group["full_name"], "input.json"])
                )

                if experimental_groups_defaults.get("pattern_for_full_search_space_filenames") is None:
                    group["full_search_space_file"] = get_full_search_space_filename_from_input_file(
                        group["application_input_file"]
                    )
                else:
                    group["full_search_space_file"] = get_full_search_space_filename_from_pattern(
                        experimental_groups_defaults["pattern_for_full_search_space_filenames"],
                        gpu,
                        application["name"],
                    )

                if group["autotuner"] == "KTT":
                    # convert full search space file from KTT output format to standard format
                    # note that full search space file in KTT output format still gets injected to input json, that is because KTT needs to have that file in its own format
                    # the converted file is loaded with this package when calculating search space statistics
                    group["converted_full_search_space_file"] = convert_KTT_to_standard_full_search_space_file(
                        group["full_search_space_file"], parent_folder_path.joinpath("setup")
                    )

                group["output_file"]: Path
                group["output_file"] = (
                    parent_folder_path.joinpath("run")
                    .joinpath(group["name"])
                    .joinpath(group["full_name"] + ".json")
                    .resolve()
                )

                generate_input_file(group)
                experimental_groups.append(group)

    return experimental_groups


def get_full_search_space_filename_from_input_file(input_filename: Path) -> Path:
    """Returns a path to full search space file that is provided in the input json file in KernelSpecification.SimulationInput.

    Args:
        input_filename: path to input json file.

    Raises:
        KeyError: if the path is not provided, but is expected.

    Returns:
        A path to full search space file that was written in the input json file.
    """
    with open(input_filename, "r", encoding="utf-8") as input_file:
        input_json = json.load(input_file)
        if input_json["KernelSpecification"].get("SimulationInput") is None:
            raise KeyError(
                "SimulationInput, i.e. full search space file is expected and not defined in",
                input_filename,
                ". Please set the path to that file in KernelSpecification.SimulationInput in input json file or set pattern_for_full_search_space_filename in experiments setup json file.",
            )
        full_search_space_filename = make_and_check_path(
            input_json["KernelSpecification"]["SimulationInput"], str(input_filename.parent), ".json"
        )
    # need to return filename WITHOUT .json, KTT (and probably also others) needs that in SimulationInput in input json as other autotuner can take other formats
    return full_search_space_filename.parent.joinpath(full_search_space_filename.stem)


def get_full_search_space_filename_from_pattern(pattern: dict, gpu: str, application_name: str) -> Path:
    """Returns a path to full search space file that is generated from the pattern provided in experiments setup file.

    Args:
        pattern: pattern regex string
        gpu: name of the gpu, needs to be plugged into the pattern
        application_name: name of the application, needs to be plugged into the pattern

    Raises:
        NotImplementedError: if the regex expects other variables than just application name and gpu.

    Returns:
        A path to full search file generated from the pattern.
    """
    filename = pattern["regex"].replace("${applications}", application_name).replace("${gpus}", gpu)
    if "${" in filename:
        raise NotImplementedError(
            f"Variables other than applications and gpus are not yet supported for pattern matching. Unresolved: {filename}."
        )
    full_search_space_filename = make_and_check_path(filename)
    return full_search_space_filename


def convert_KTT_to_standard_full_search_space_file(full_search_space_file: Path, setup_folder: Path) -> Path:
    """Converts KTT-formatted full search space file to the standard format recognized by this package.

    Args:
        full_search_space_file: the path to KTT-formatted full search space file
        setup_folder: path to setup directory for this experiment

    Returns:
        A path to newly created full search space file in standard format, in the setup directory of the experiment
    """
    converted_output = convert_KTT_output_to_standard(full_search_space_file.with_suffix(".json"))
    converted_filename = setup_folder.joinpath(full_search_space_file.stem + "_converted.json")

    with open(converted_filename, "w", encoding="utf-8") as converted_file:
        json.dump(converted_output, converted_file, indent=4)

    return converted_filename


def calculate_budget(group: dict, statistics_settings: dict, searchspace_stats: SearchspaceStatistics) -> dict:
    """Calculates the budget for the experimental group, given cutoff point provided in experiments setup file.

    Args:
        group: a dictionary with settings for experimental group
        statistics_settings: a dictionary with settings related to statistics
        searchspace_stats: a SearchspaceStatistics instance with cutoff points determined from related full search space files

    Returns:
        A modified group dictionary.
    """
    group["budget"] = {}
    # set cutoff point
    _, cutoff_point_fevals, cutoff_point_time = searchspace_stats.cutoff_point_fevals_time(
        statistics_settings["cutoff_percentile"]
    )

    # +10% margin, to make sure cutoff_point is reached by compensating for potential non-valid evaluations  # noqa: E501
    cutoff_margin = group.get("cutoff_margin", 1.1)

    # set when to stop
    if statistics_settings["cutoff_type"] == "time":
        group["budget"]["time_limit"] = cutoff_point_time * cutoff_margin
    else:
        group["budget"]["max_fevals"] = min(int(ceil(cutoff_point_fevals * cutoff_margin)), searchspace_stats.size)

    # write to group's input file as Budget
    with open(group["input_file"], "r", encoding="utf-8") as fp:
        input_json = json.load(fp)
        if input_json.get("Budget") is None:
            input_json["Budget"] = []
            input_json["Budget"].append({})
        if group["budget"].get("time_limit") is not None:
            input_json["Budget"][0]["Type"] = "TuningDuration"
            input_json["Budget"][0]["BudgetValue"] = group["budget"]["time_limit"]
        else:  # it's max_fevals
            input_json["Budget"][0]["Type"] = "ConfigurationCount"
            input_json["Budget"][0]["BudgetValue"] = group["budget"]["max_fevals"]

    with open(group["input_file"], "w", encoding="utf-8") as fp:
        json.dump(input_json, fp, indent=4)

    return group


def generate_input_file(group: dict):
    """Creates a input json file specific for a given application, gpu and search method.

    Args:
        group: dictionary with settings for a given experimental group.
    """
    with open(group["application_input_file"], "r", encoding="utf-8") as fp:
        input_json = json.load(fp)
        input_json["KernelSpecification"]["SimulationInput"] = str(group["full_search_space_file"])
        input_json["General"]["OutputFile"] = str(group["output_file"].parent.joinpath(group["output_file"].stem))
        if input_json["General"]["OutputFormat"] != "JSON":
            raise RuntimeError(
                f"Only JSON output format is supported. Please set General.OutputFormat to JSON in {group['application_input_file']}."
            )
        if input_json["KernelSpecification"].get("Device") is None:
            input_json["KernelSpecification"]["Device"] = {}
            input_json["KernelSpecification"]["Device"]["Name"] = group["gpu"]
        else:
            input_json["KernelSpecification"]["Device"]["Name"] = group["gpu"]

        input_json["Search"] = {}
        input_json["Search"]["Name"] = group["search_method"]
        if group.get("search_method_hyperparameters") is not None:
            input_json["Search"]["Attributes"] = []
            for param in group["search_method_hyperparameters"]:
                attribute = {}
                attribute["Name"] = param["name"]
                attribute["Value"] = param["value"]
                input_json["Search"]["Attributes"].append(attribute)
    # note that this is written to a different file, specific for gpu, application and search method
    with open(group["input_file"], "w", encoding="utf-8") as fp:
        json.dump(input_json, fp, indent=4)


def execute_experiment(filepath: str, profiling: bool = False) -> tuple[dict, dict, dict]:
    """Executes the experiment by retrieving it from the cache or running it.

    Args:
        filepath: path to the experiments .json file.
        profiling: whether profiling is enabled. Defaults to False.

    Raises:
        FileNotFoundError: if the path to the kernel specified in the experiments file is not found.

    Returns:
        A tuple of the experiment dictionary, the experimental groups executed, and the resulting list of ``ResultsDescription``.
    """
    experiment = get_experiment(filepath)
    experiment_folderpath = Path(experiment["parent_folder"])
    print(f"Starting experiment '{experiment['name']}'")

    all_experimental_groups = get_experimental_groups(experiment)

    # prepare objective_time_keys, in case it was defined as all, explicitly list all keys
    objective_time_keys: list[str] = experiment["statistics_settings"]["objective_time_keys"]
    if "all" in objective_time_keys:
        objective_time_keys = []
        # get the path to the schema
        schemafile = get_experiment_schema_filepath()
        # open the experiment file and validate using the schema file
        with open(schemafile, "r", encoding="utf-8") as schemafile:
            schema = json.load(schemafile)
            objective_time_keys = schema["properties"]["statistics_settings"]["properties"]["objective_time_keys"][
                "items"
            ]["enum"]
        objective_time_keys.remove("all")
        experiment["statistics_settings"]["objective_time_keys"] = objective_time_keys

    experiment["experimental_groups_defaults"]["applications_names"] = []
    for application in experiment["experimental_groups_defaults"]["applications"]:
        experiment["experimental_groups_defaults"]["applications_names"].append(application["name"])

    # initialize the matrix of results_descriptions based on provided gpus and applications
    # initialize searchspace statistics, one for each full search file
    results_descriptions: dict[str, dict[str, dict[str, ResultsDescription]]] = {}
    searchspace_statistics: dict[str, dict[str, SearchspaceStatistics]] = {}

    for gpu in experiment["experimental_groups_defaults"]["gpus"]:
        results_descriptions[gpu] = {}
        searchspace_statistics[gpu] = {}
        for application in experiment["experimental_groups_defaults"]["applications_names"]:
            results_descriptions[gpu][application] = {}

    # just iterate over experimental_groups, collect results and write to proper place
    for group in all_experimental_groups:

        print(f" | - running on GPU '{group['gpu']}'")
        print(f" | - | tuning application '{group['application_name']}'")
        print(f" | - | - | with settings of experimental group '{group['display_name']}'")

        # create SearchspaceStatistics for full search space file associated with this group, if it does not exist
        if searchspace_statistics.get(group["gpu"]).get(group["application_name"]) is None:
            full_search_space_file_path = None
            if group.get("converted_full_search_space_file") is None:
                full_search_space_file_path = group["full_search_space_file"]
            else:
                full_search_space_file_path = group["converted_full_search_space_file"]

            searchspace_statistics[group["gpu"]][group["application_name"]] = SearchspaceStatistics(
                application_name=group["application_name"],
                device_name=group["gpu"],
                minimization=experiment["statistics_settings"]["minimization"],
                objective_time_keys=objective_time_keys,
                objective_performance_keys=experiment["statistics_settings"]["objective_performance_keys"],
                full_search_space_file_path=full_search_space_file_path,
            )

        # calculation of budget can be done only now, after searchspace statistics have been initialized
        group = calculate_budget(
            group, experiment["statistics_settings"], searchspace_statistics[group["gpu"]][group["application_name"]]
        )

        results_description = ResultsDescription(
            run_folder=experiment_folderpath / "run" / group["name"],
            application_name=group["application_name"],
            device_name=group["gpu"],
            group_name=group["name"],
            group_display_name=group["display_name"],
            stochastic=group["stochastic"],
            objective_time_keys=objective_time_keys,
            objective_performance_keys=experiment["statistics_settings"]["objective_performance_keys"],
            minimization=experiment["statistics_settings"]["minimization"],
        )

        # if the strategy is in the cache, use cached data
        if ("ignore_cache" not in group or group["ignore_cache"] is False) and results_description.has_results():
            print(" | - | - | -> retrieved from cache")
        else:  # execute each strategy that is not in the cache
            results_description = collect_results(
                group["input_file"],
                group,
                results_description,
                searchspace_statistics[group["gpu"]][group["application_name"]],
                profiling=profiling,
            )

        # set the results
        results_descriptions[group["gpu"]][group["application_name"]][group["name"]] = results_description

    return experiment, all_experimental_groups, searchspace_statistics, results_descriptions


def entry_point():  #  pragma: no cover
    """Entry point function for Experiments."""
    experiment_filepath = get_args_from_cli()
    execute_experiment(experiment_filepath, profiling=False)


if __name__ == "__main__":
    entry_point()
