"""Utility script to extend a Kernel Tuner cachefile by using another cachefile that has those parameters. Use with caution."""

import json
import numbers
from copy import deepcopy
from itertools import product
from pathlib import Path

# set the files to use
basepath = Path(__file__).parent
target_infile = basepath / "convolution_milo" / "MI50_original.json"
target_outfile = basepath / "convolution_milo" / "MI50_extended.json"
extra_sourcefile = basepath / "convolution_milo" / "MI250X.json"

# load the JSON files
with target_infile.open() as fp:
    target: dict = json.load(fp)
    new_target = deepcopy(target)
with extra_sourcefile.open() as fp:
    extra_source: dict = json.load(fp)["cache"]

# define the parameters to add, their default value, and their list of values
# caution: order must be the same as in `extra_sourcefile`, `extra_sourcefile` use the superset of parameter values
parameters_to_add = {
    "use_shmem": (1, [0, 1]),
    "use_cmem": (1, [1]),
    "filter_height": (15, [15]),
    "filter_width": (15, [15]),
}
default_config_string = ",".join([str(p[0]) for p in parameters_to_add.values()])

# add parameters to header
for param, (_, values) in parameters_to_add.items():
    new_target["tune_params_keys"].append(param)
    new_target["tune_params"][param] = values

# add parameters to cache
# caution: does not take restrictions into account
extra_configurations = list(product(*[p[1] for p in parameters_to_add.values()]))
for config_string, base_config in target["cache"].items():
    # lookup the base config in the other cachefile using the defaults
    source_base_config: dict = extra_source[f"{config_string},{default_config_string}"]

    # delete the old config from the new data
    del new_target["cache"][config_string]

    # for each existing config, add as many new configurations as needed by inferring from source
    for extra_config in extra_configurations:
        extra_config_string = ",".join([str(p) for p in extra_config])
        new_config_string = f"{config_string},{extra_config_string}"

        # lookup the extra config in the other cachefile to use as a basis
        try:
            source_extra_config: dict = extra_source[new_config_string]
        except KeyError:
            # as we assume that the extra source is a superset, this config is most likely skipped due to restrictions
            continue
        new_target_config = deepcopy(source_extra_config)

        # change the values for target based on the relative difference between target, source base and source extra
        def change_relatively(target_base, source_base, source_extra):
            # check if there are any error values
            if isinstance(target_base, str):
                return target_base
            elif isinstance(source_extra, str):
                return source_extra
            elif isinstance(source_base, str):
                return source_base
            # make sure all are the same type
            assert type(target_base) == type(source_base) == type(source_extra)
            if isinstance(target_base, (list, tuple)):
                # if we're dealing with lists, go recursive
                assert len(target_base) == len(source_base) == len(source_extra)
                return [
                    change_relatively(target_base[i], source_base[i], source_extra[i]) for i in range(len(target_base))
                ]
            # final check for the type
            if not isinstance(target_base, numbers.Real):
                raise ValueError(
                    f"Relative value change is not possible for non-numeric values of type {type(target_base)} ({target_base})"
                )
            # since we're dealing with numbers, we can do the relative value change
            try:
                fraction = source_extra / source_base
                return target_base * fraction
            except ZeroDivisionError:
                return target_base

        # apply the relative value change
        for key in [
            "time",
            "times",
            "compile_time",
            "verification_time",
            "benchmark_time",
            "strategy_time",
            "framework_time",
            "GFLOP/s",
        ]:
            new_target_config[key] = change_relatively(
                base_config[key], source_base_config[key], source_extra_config[key]
            )

        # add the new config to the new target data
        new_target["cache"][new_config_string] = new_target_config

# check that the extension is succesful
assert len(new_target["cache"]) == len(
    extra_source
), f"Lengths don't match; target: {len(new_target['cache'])}, source: {len(extra_source)}"

# write to the target file
with target_outfile.open("w+") as fp:
    json.dump(new_target, fp)
