# Autotuning Methodology Software Package
This repository contains the software package accompanying the paper "A Methodology for Comparing Auto-Tuning Optimization Algorithms". 
It makes the guidelines in the methodology easy to apply: simply specify the  `.json` file, run it with `python visualize_experiments.py` and wait for the results!

## Notable features: 
- Official software by the authors of the methodology-defining paper. 
- Supports [BAT benchmark suite](https://github.com/NTNU-HPC-Lab/BAT) and [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner).
- Split executer and visualizer to allow running the algorithms on a cluster and visualize locally. 
- Caching built-in to avoid duplicate executions.  
- Planned support for T1 input and T4 output files.
- Notebook / interactive window mode; in this case, plots are shown in the notebook / window instead of written to a folder. 

<img width="674" alt="example run in interactive window" src="https://user-images.githubusercontent.com/6725103/232880006-70a05b0e-a4e4-4cc7-bea9-473959c474c2.png">
<img width="483" alt="example run in interactive window 2" src="https://user-images.githubusercontent.com/6725103/232881244-d432ea8e-801a-44b1-9acb-b98cc1b740ac.png">
