"""Script to overwrite Kernel Tuner brute forced cache files with the objective values of a KTT brute force search.

Notes: this requires a fully bruteforced KTT and fully bruteforced KernelTuner (KT) cachefile on the same search space.
Objective value is assumed to be time by default. Time is assumed to be in microseconds for KTT and miliseconds for KT. 
"""

import json
from pathlib import Path

from autotuning_methodology.runner import ktt_param_mapping

kerneltuner_cachefiles_path = Path(__file__).parent.resolve()
assert kerneltuner_cachefiles_path.exists()
ktt_data_path = kerneltuner_cachefiles_path / "KTT data"
assert ktt_data_path.exists()

files_to_import = [f for f in ktt_data_path.iterdir() if f.is_file() and f.suffix == ".json"]
ktt_objective_name = "Duration"
kt_objective_name = "time"

error_status_mapping = {
    "ok": None,
    "devicelimitsexceeded": '"CompilationFailedConfig"',
    "computationfailed": '"RuntimeFailedConfig"',
}

for file in files_to_import:
    # find the associated KernelTuner cachefile to write to
    ktt_data = dict(json.loads(file.read_bytes()))
    metadata = ktt_data["Metadata"]
    device = str(metadata["Device"])
    device_filename = device.replace("NVIDIA GeForce ", "").replace(" ", "_")
    kernel = str(ktt_data["Results"][0]["KernelName"])
    kernel_filename = kernel.lower()
    kerneltuner_cachefile = kerneltuner_cachefiles_path / kernel_filename / f"{device_filename}.json"
    assert kerneltuner_cachefile.exists()
    ktt_param_mapping_kernel = ktt_param_mapping[kernel_filename]
    print(f"Importing objective values from KTT to KernelTuner file for '{kernel}' on {device}")

    # for each configuration in the KTT file, use the value in the KernelTuner file
    config_to_change = dict()
    kerneltuner_data = dict(json.loads(kerneltuner_cachefile.read_bytes()))
    ktt_results = ktt_data["Results"]
    cache = kerneltuner_data["cache"]
    assert len(cache) == len(ktt_results)
    for ktt_config in ktt_results:
        # convert the configuration to T4 style dictionary for fast lookups in the mapping
        configuration_ktt = dict()
        for param in ktt_config["Configuration"]:
            configuration_ktt[param["Name"]] = param["Value"]

        # convert the configuration data with the mapping in the correct order
        configuration = dict()
        param_map = ktt_param_mapping_kernel
        assert len(param_map) == len(
            configuration_ktt
        ), f"Mapping provided for {len(param_map)} params, but configuration has {len(configuration_ktt)}"
        for param_name, mapping in param_map.items():
            param_value = configuration_ktt[param_name]
            # if the mapping is None, do not include the parameter
            if mapping is None:
                pass
            # if the mapping is a tuple, the first argument is the new parameter name and the second the value
            elif isinstance(mapping, tuple):
                param_mapped_name, param_mapped_value = mapping
                if callable(param_mapped_value):
                    param_mapped_value = param_mapped_value(param_value)
                configuration[param_mapped_name] = param_mapped_value
            # if it's a list of tuples, map to multiple parameters
            elif isinstance(mapping, list):
                for param_mapped_name, param_mapped_value in mapping:
                    if callable(param_mapped_value):
                        param_mapped_value = param_mapped_value(param_value)
                    configuration[param_mapped_name] = param_mapped_value
            else:
                raise ValueError(f"Can not apply parameter mapping of {type(mapping)} ({mapping})")

        # get and validate the Kernel Tuner configuration
        lookup_string = ",".join(str(v) for v in configuration.values())  # the key to lookup the configuration
        assert lookup_string in cache
        kt_config = cache[lookup_string]
        for param, value in configuration.items():
            assert kt_config[param] == value

        # replace the objective in the KT configuration with the objective in the KTT configuration
        kt_old_objective_value = kt_config[kt_objective_name]
        kt_new_objective_value = ""
        status = error_status_mapping[str(ktt_config["Status"]).lower()]
        if status is None:
            kt_new_objective_value = ktt_config["ComputationResults"][0][ktt_objective_name] / 1000
        else:
            kt_new_objective_value = status
        kerneltuner_data["cache"][lookup_string][kt_objective_name] = kt_new_objective_value
        config_to_change[lookup_string] = (kt_old_objective_value, kt_new_objective_value)
        # print(f"Replacing {kt_old_objective_value} with {kt_new_objective_value}")

    # load the individual lines of the file
    with kerneltuner_cachefile.open(mode="r") as fp:
        lines = fp.readlines()
        cache_start = False
    # write the new data to file
    with kerneltuner_cachefile.open(mode="w") as fp:
        # for each line in the cache part of the file, lookup the config string in the changes dictionary and replace
        for line in lines:
            if '"cache":' in line:
                cache_start = True
                fp.write(line)
            elif not cache_start or line[:1] == "}" or len(line) < 3:
                fp.write(line)
            else:
                lookup_string = line.split(":")[0].replace('"', "").strip()
                old_value, new_value = config_to_change[lookup_string]
                line = line.replace(f'"time": {old_value},', f'"time": {new_value},', 1)
                fp.write(line)

    # kerneltuner_cachefile.write_text(json.dumps(kerneltuner_data, indent=3))
