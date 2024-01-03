# zms2
Image analysis pipeline for MS2 reporters in large, dense tissues like zebrafish embryos. See the paper by Eck, Moretti, and Schlomann, bioRxiv (2024).

## Installation

zms2 relies heavily on GPU-accelerated image analysis and requires a CUDA-compatible GPU already set up. This code has also only been tested on Linux machines running. Currently only local installation via pip is supported. First clone this repository.

We recommend creating a conda environment using our provided zms2_gpu_env.yml file:

```bash
conda env create -f zms2_gpu_env.yml 
conda activate zms2
```

Then navigate to the zms2 directory and install via pip

```bash
pip install .
```

