# Autotuning Methodology Software Package

<p align="center">
  <img width="25%" src="https://fjwillemsen.github.io/autotuning_methodology/_static/logo_autotuning_methodology.svg" />
</p>

[![Build Status](https://github.com/fjwillemsen/autotuning_methodology/actions/workflows/build-test-python-package.yml/badge.svg)](https://github.com/fjwillemsen/autotuning_methodology/actions/workflows/build-test-python-package.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/fjwillemsen/autotuning_methodology/publish-documentation.yml?label=docs)](https://fjwillemsen.github.io/autotuning_methodology/)
[![Python Versions](https://img.shields.io/pypi/pyversions/autotuning_methodology)](https://pypi.org/project/autotuning_methodology/)
![PyPI - License](https://img.shields.io/pypi/l/autotuning_methodology)
[![PyPI Downloads](https://img.shields.io/pypi/dm/autotuning_methodology)](https://pypi.org/project/autotuning_methodology/)


This repository contains the software package accompanying the paper "A Methodology for Comparing Auto-Tuning Optimization Algorithms". 
It makes the guidelines in the methodology easy to apply: simply specify the  `.json` file, run `autotuning_visualize [path_to_json]` and wait for the results!

## Installation
The package can be installed by cloning this repository and running `pip install .`. Python >= 3.9 is supported.  

## Notable features
- Official software by the authors of the methodology-defining paper. 
- Supports [BAT benchmark suite](https://github.com/NTNU-HPC-Lab/BAT) and [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner).
- Split executer and visualizer to allow running the algorithms on a cluster and visualize locally. 
- Caching built-in to avoid duplicate executions.  
- Planned support for T1 input and T4 output files.
- Notebook / interactive window mode; in this case, plots are shown in the notebook / window instead of written to a folder. 

<img width="674" alt="example run in interactive window" src="https://user-images.githubusercontent.com/6725103/232880006-70a05b0e-a4e4-4cc7-bea9-473959c474c2.png">
<img width="483" alt="example run in interactive window 2" src="https://user-images.githubusercontent.com/6725103/232881244-d432ea8e-801a-44b1-9acb-b98cc1b740ac.png">

## Usage

### Entry points
There are two entry points defined: `autotuning_experiment` and `autotuning_visualize`. Both take one argument: the path to an experiment file (see below). 

### Input files
To get started, all you need is an experiments file. This is a `json` file that describes the details of your comparison: which algorithms to use, which programs to tune on which devices, the graphs to output and so on. 
You can find the API and an example `experiments.json` in the [documentation](). 

### File references
As we are dealing with input and output files, file references matter. 
When calling the entrypoints, we are already providing the path to an experiments file. 
File references in experiments files are relative to the location of the experiment file itself. 
File references in tuning scripts are relative to the location of the tuning script itself. Tuning scripts need to have the global literals `file_path_results` and `file_path_metadata` for this package to know where to get the results. 
Plots outputted by this package are placed in a folder called `generated_plots` relative to the current working directory. 


## Contributing

### Setup
If you're looking to contribute to this package: welcome!
Start out by installing with `pip install -e .[dev]` (this installs the package in editable mode alongside the development dependencies). 
During development, unit and integration tests can be ran with `pytest`. 
[Black](https://pypi.org/project/black/) is used as a formatter, and [Ruff](https://pypi.org/project/ruff/) is used as a linter to check the formatting, import sorting et cetera. 
When using Visual Studio Code, use the `settings.json` found in `.vscode` to automatically have the correct linting, formatting and sorting during development. 
In addition, install the extensions recommended by us by searching for `@recommended:workspace` in the extensions tab for a better development experience. 

### Documentation
The documentation can be found [here](https://fjwillemsen.github.io/autotuning_methodology/). 
Locally, the documentation can be build with `make clean html` from the `docs` folder, but the package must have been installed in editable mode with `pip install -e .`. 
Upon pushing to main or publishing a version, this documentation will be built and published to the GitHub Pages. 
The Docstring format used is Google. Type hints are to be included in the function signature and therefor omitted from the docstring. In Visual Studio Code, the `autoDocstring` extension can be used to automatically infer docstrings. When referrring to functions and parameters in the docstring outside of their definition, use double backquotes to be compatible with both MarkDown and ReStructuredText, e.g.: *"skip_draws_check: skips checking that each value in ``draws`` is in the ``dist``."*.

### Tests
Before contributing a pull request, please run `nox` and ensure it has no errors. This will test against all Python versions explicitely supported by this package, and will check whether the correct formatting has been applied.
Upon submitting a pull request or pushing to main, these same checks will be ran remotely via GitHub Actions. 

### Publishing
For publising the package to PyPI (the Python Package Index), we use [Flit](https://flit.pypa.io) and the [to-pypi-using-flit](https://github.com/AsifArmanRahman/to-pypi-using-flit/tree/v1/) GitHub Action to automate this. 

[Semantic version numbering](https://semver.org) is used as follows: `MAJOR.Minor.patch`. 
`MAJOR` version for incompatible API changes.
`Minor` version for functionality in a backward compatible manner.
`patch` version for backward compatible bug fixes. 
In addition, [PEP 440](https://peps.python.org/pep-0440/) is adhered to, specifically for [pre-release versioning](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#id62). 
