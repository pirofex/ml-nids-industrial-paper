# Towards a Better Understanding of Machine Learning based Network Intrusion Detection Systems in Industrial Networks

## Installation

### Ubuntu 18.04.

* `sudo apt install python-rpy2`
* removed all dependencies from the file which are listed under "pip"
* `pip install matplotlib==3.2.2`
* `pip install pdpbox==0.2.0`
* `pip install xlrd==1.2.0`
* in pdp_plot_utils.py (something line `anaconda3/envs/lukas-thesis/lib/python3.7/site-packages/pdpbox/pdp_plot_utils.py`), replace the following:

```
    # inter_ax.clabel(c2, contour_label_fontsize=fontsize, inline=1) # old
    inter_ax.clabel(c2, fontsize=fontsize, inline=1) # new
```

### Windows 10

- The code was tested on Windows 10 Version 1909 with the VS Code IDE
- The dependencies which need to be installed are listed in 'dependencies_windows.yml'
  If using Anaconda, the following command creates a new environment with the
  needed dependencies: `conda env create -f dependencies_windows.yml`

## Run

* run `conda env create -f dependencies.yml`
* to start conda env: `conda activate lukas-thesis`
* to run the starter code: `python src/_playground.py`
* to stop the conda env: `conda deactivate`


