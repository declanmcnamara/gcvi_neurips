# Globally Convergent Variational Inference


This repository contains code to reproduce the experiments from "Globally Convergent Variational Inference", appearing in the Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS), 2024.


To get started, create a fresh virtual environment and install required packages as follows:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To reproduce the results for any of the experiments complete the following steps.
 - Navigate to the subdirectory of interest (e.g., `cd rotated_mnist_full`).
 - Modify the `dir` field of the `.yaml` file in `config` to be the path to this repository on your system.
 - Run the scripts (e.g., `./runner.sh`). Note you may need to run `chmod +x runner.sh` to make this executable.

The experiments were run on GPUs, allowing for significantly faster computation. By default, we made CPU the default device for all experiments. To run on your GPU, modify the `training.device` field in the `runner.sh` scripts or in the `.yaml` config file.