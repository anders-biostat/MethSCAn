[![Tests](https://img.shields.io/github/actions/workflow/status/anders-biostat/MethSCAn/install_lint_test.yml?branch=master)](https://github.com/anders-biostat/MethSCAn/actions/workflows/install_lint_test.yml)
[![PyPI](https://img.shields.io/pypi/v/methscan?logo=PyPI)](https://pypi.org/project/MethSCAn)
[![PyPIDownloads](https://pepy.tech/badge/scbs)](https://pepy.tech/project/methscan)
[![GitHub](https://img.shields.io/github/v/tag/anders-biostat/MethSCAn?logo=github)](https://github.com/anders-biostat/MethSCAn)

a command line tool for **S**ingle-**C**ell **An**alysis of **Meth**ylation data.

## Installation

This software requires a working installation of [Python 3](https://www.python.org/downloads/) (≥3.8) and requires the use of a shell terminal.
It was extensively tested on Linux (Ubuntu 18, 20 and 22) and MacOS, and briefly tested on Windows 10.

You can install *MethSCAn* from the Python package index as follows:
```
python3 -m pip install --upgrade pip  # you need a recent pip version
python3 -m pip install methscan
```
Installation of *MethSCAn* should take no longer than a few seconds. All required [dependencies](https://github.com/anders-biostat/MethSCAn/blob/master/pyproject.toml) are automatically installed, this may take a few minutes.
Afterwards, restart your terminal. The installation is now finished and the command line interface should now be available when typing the command `methscan` in your terminal.
If this is not the case, check the "troubleshooting" section below.  


## Updating to the latest version
Just use `--upgrade` when installing the package, otherwise it's the same process as installing:
```
python3 -m pip install --upgrade methscan
```
Afterwards, make sure that the latest version is correctly installed:
```
methscan --version
```

## [Tutorial](tutorial.html) of a typical `methscan` run
A tutorial / demo can be found [here](tutorial.html).
This gives instructions on how to use *MethSCAn* on a small example data set which we provide.

Also make sure to read the help by typing `methscan --help` or by checking [this page](commands.html).


## What can this package do?

*MethSCAn* takes as input a number of single-cell methylation files and allows you to quickly and easily obtain a cell × region matrix for downstream analysis (e.g. PCA, UMAP or clustering).
It also facilitates quality control, allows you to discover variably methylated regions (VMRs), accurately quantifies methylation in genomic intervals, and stores your sc-methylomes in an efficient manner.
Lastly, you can also select two cell populations and identify differentially methylated regions (DMRs) between them.

![schematic showing the capabilities of MethSCAn.](Fig_workflow.png)

You can find a list of the available `methscan` commands [here](commands.html).


## Publication / Citation

For a detailed explanation of the methods implemented in *MethSCAn*, please check out our open access article in Nature Methods:

*Analyzing single-cell bisulfite sequencing data with MethSCAn*  
Lukas PM Kremer, Martina Braun, Svetlana Ovchinnikova, Leonie Kuechenhoff, Santiago Cerrizuela, Ana Martin-Villalba, Simon Anders.  
Nature Methods, 2024
doi: [https://doi.org/10.1038/s41592-024-02347-x](https://doi.org/10.1038/s41592-024-02347-x)

Please cite this article if you used MethSCAn in your research.  
Note that this package was formerly known as ['scbs'](https://github.com/LKremer/scbs) and later renamed to *MethSCAn*.

## Hardware requirements

For intermediate data sets consisting of 1000 to 5000 cells, we recommend to use a computer with at least 16 gigabytes of RAM.
Very large data sets (~100k cells) require at least 128 GB.
Multiple CPU cores are not strictly required but will greatly speed up some commands such as `methscan scan` or `methscan diff` when using the `--threads` argument.


## Troubleshooting

#### Installation issues

Carefully check the output log of PIP. Look for a message like `WARNING: The script methscan is installed in '/home/ubuntu/.local/bin' which is not on PATH.`, which would indicate that you need to add `/home/ubuntu/.local/bin` to your path. Alternatively, you can copy `/home/ubuntu/.local/bin/methscan` to e.g. `/usr/local/bin`.

If you encounter other problems during installation, make sure you have Python3.8 or higher, and make sure you have the latest PIP version. If the problem persists, consider installing `methscan` in a clean Python environment (for example using [venv](https://docs.python.org/3/library/venv.html)).

#### Too many open files
If you encounter a "too many open files" error during `methscan prepare` (`OSError: [Errno 24] Too many open files`), you need to increase the maximum number of files that can be opened. On Unix systems, try `ulimit -n 99999`.


## Source Code

The source code is hosted at the GitHub repository [anders-biostat/MethSCAn](https://github.com/anders-biostat/MethSCAn).


## Authors
- [Lukas PM Kremer](https://github.com/LKremer)
- [Martina Braun](https://github.com/martinabraun)
- [Svetlana Ovchinnikova](https://github.com/kloivenn)
- [Leonie Küchenhoff](https://github.com/LeonieKuechenhoff)
- Santiago Cerrizuela
- Ana Martin-Villalba
- [Simon Anders](https://github.com/simon-anders)

Developed in the [Anders lab](https://www.bioquant.uni-heidelberg.de/groups/anders/team) and the [Martin-Villalba lab](https://martin-villalba-lab.github.io/).
