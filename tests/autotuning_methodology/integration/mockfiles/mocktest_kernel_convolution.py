#!/usr/bin/env python
import sys

import numpy
import kernel_tuner
from kernel_tuner.file_utils import store_output_file, store_metadata_file
from collections import OrderedDict


def tune(device_name: str, strategy="mls", strategy_options=None, verbose=True, quiet=False, simulation_mode=True):
    # input dimensions and data
    image_width = 4096
    image_height = 4096
    filter_width = 15
    filter_height = 15
    problem_size = (image_width, image_height)
    size = numpy.prod(problem_size)

    input_size = (problem_size[0] + filter_width - 1) * (problem_size[1] + filter_height - 1)
    output_image = numpy.zeros(size).astype(numpy.float32)
    input_image = numpy.random.randn(input_size).astype(numpy.float32)
    filter_weights = numpy.random.randn(filter_width * filter_height).astype(numpy.float32)

    cmem_args = {"d_filter": filter_weights}
    args = [output_image, input_image, filter_weights]

    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda p: (image_width * image_height * filter_width * filter_height * 2 / 1e9) / (
        p["time"] / 1e3
    )

    # setup tunable parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [1, 2, 4, 8, 16, 32, 48, 64, 96, 112, 128]
    tune_params["block_size_y"] = [1, 2, 4, 8, 16, 32]
    tune_params["filter_height"] = [filter_height]
    tune_params["filter_width"] = [filter_width]
    tune_params["read_only"] = [0, 1]
    tune_params["tile_size_x"] = [1, 2, 3, 4, 5, 6, 7, 8]
    tune_params["tile_size_y"] = [1, 2, 3, 4, 5, 6, 7, 8]
    tune_params["use_padding"] = [0, 1]

    restrict = ["block_size_x*block_size_y>=64", "block_size_x*block_size_y<=1024", "tile_size_x*tile_size_y<30"]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    # #compute the answer using a naive kernel
    # params = {
    #     "filter_width": filter_width,
    #     "filter_height": filter_height,
    #     "block_size_x": 16,
    #     "block_size_y": 16
    # }
    # results = kernel_tuner.run_kernel("convolution_naive", "convolution.cu", problem_size, args, params, grid_div_y=["block_size_y"],
    #                                   grid_div_x=["block_size_x"])
    # gc.collect()
    # #set non-output fields to None
    # answer = [results[0], None, None]

    # start tuning
    results, env = kernel_tuner.tune_kernel(
        "convolution_kernel",
        "convolution.cu",
        problem_size,
        args,
        tune_params,
        grid_div_y=grid_div_y,
        grid_div_x=grid_div_x,
        cmem_args=cmem_args,
        restrictions=restrict,
        cache="cached_data_used/cachefiles/mocktest_kernel_convolution/" + device_name.lower(),
        metrics=metrics,
        iterations=32,
        device=0,
        verbose=verbose,
        quiet=quiet,
        strategy=strategy,
        strategy_options=strategy_options,
        simulation_mode=simulation_mode,
    )

    store_output_file("cached_data_used/last_run/_tune_configuration-results.json", results, tune_params)
    store_metadata_file("cached_data_used/last_run/_tune_configuration-metadata.json")
    return results, env


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./mocktest_kernel_convolution.py [device name]")
        exit(1)

    device_name = sys.argv[1]

    tune(device_name)
