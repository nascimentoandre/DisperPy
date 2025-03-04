# About

DisperPy is a free and open-source Python software, based on machine learning
algorithms, which allows to automatically extract group velocity dispersion curves
from earthquake data. The code aims to solve two main tasks: (1) to assess
dispersion quality and (2) to pick the respective group velocity dispersion
from seismograms (in the SAC format).

A few sample waveforms are available in the `sample_data` folder to test the
code, and more details on how to use the code can be found in the manual within
the `doc` directory.

# Quick start

## Requirements

- obspy;
- numpy;
- scipy;
- matplotlib;
- scikit-learn;
- pytorch;
- fastai.

We highly recommend the usage of
[Anaconda](https://www.anaconda.com/docs/main). We also recommend to create a
separate Anaconda environment to avoid possible version conflicts. A
straightforward way to create such an environment with the dependencies
installed is:

```sh
conda create -n dp python=3.11 fastai obspy scikit-learn scipy
```

## Running the code

The script `main.py` wrapps all the functions necessary to extract group
dispersion, and to run it you simply need to provide the path to the directory
containing the SAC files:

```sh
python main.py -f sample_data
```

# Citation

xxxxxxxx
