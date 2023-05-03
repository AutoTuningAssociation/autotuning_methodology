# Autotuning Methodology Software Package
![build status](https://github.com/fjwillemsen/autotuning_methodology/actions/workflows/build-test-python-package.yml/badge.svg)

This repository contains the software package accompanying the paper "A Methodology for Comparing Auto-Tuning Optimization Algorithms". 
It makes the guidelines in the methodology easy to apply: simply specify the  `.json` file, run it with `python visualize_experiments.py` and wait for the results!

## Installation
The package can be installed by cloning this repository and running `pip install .`. Python >= 3.8 is supported. 

## Notable features
- Official software by the authors of the methodology-defining paper. 
- Supports [BAT benchmark suite](https://github.com/NTNU-HPC-Lab/BAT) and [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner).
- Split executer and visualizer to allow running the algorithms on a cluster and visualize locally. 
- Caching built-in to avoid duplicate executions.  
- Planned support for T1 input and T4 output files.
- Notebook / interactive window mode; in this case, plots are shown in the notebook / window instead of written to a folder. 

<img width="674" alt="example run in interactive window" src="https://user-images.githubusercontent.com/6725103/232880006-70a05b0e-a4e4-4cc7-bea9-473959c474c2.png">
<img width="483" alt="example run in interactive window 2" src="https://user-images.githubusercontent.com/6725103/232881244-d432ea8e-801a-44b1-9acb-b98cc1b740ac.png">

## Contributing

### Setup
If you're looking to contribute to this package: welcome!
Start out by installing with `pip install -e .[dev]` (this installs the package in editable mode alongside the development dependencies). 
During development, unit and integration tests can be ran with `pytest`. 
[Black](https://pypi.org/project/black/) is used as a formatter, and [Ruff](https://pypi.org/project/ruff/) is used as a linter to check the formatting, import sorting et cetera. 
When using Visual Studio Code, use the `settings.json` found in `.vscode` to automatically have the correct linting, formatting and sorting during developments. 

### Documentation
Locally, the documentation can be build with `make html` from the `docs` folder. 
Upon pushing to main or publishing a version, this documentation will be built and published to the GitHub Pages. 

### Tests
Before contributing a pull request, please run `nox` and ensure it has no errors. This will test against all Python versions explicitely supported by this package, and will check whether the correct formatting has been applied.
Upon submitting a pull request or pushing to main, these same checks will be ran remotely via GitHub Actions. 
