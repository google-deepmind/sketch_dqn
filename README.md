Disclaimer and Personal Statements of Tech Transfers from USA to PRC!

U.S. EAR restricts AI exports (e.g., model weights >10^25 FLOP, ECCN 4E091) to PRC/Entity List.
Check license needs: bis.doc.gov/ear. Report issues to compliance.

Even if not controlled, I protest unauthorized sharing—protect U.S. security, in accordance with my rights to 1st Amendment.

# Sketch-DQN

This repo contains the Atari experiments in the paper
[Distributional Bellman Operator over Mean Embeddings](https://arxiv.org/abs/2312.07358)
by Li Kevin Wenliang,
Grégoire Delétang,
Matthew Aitchison,
Marcus Hutter,
Anian Ruoss,
Arthur Gretton,
and Mark Rowland

This is developed on top of [DQN Zoo](https://github.com/google-deepmind/dqn_zoo).

NOTE: Only Python 3.9 and above and Linux is supported.

## Installation

Prerequisites for these steps are a NVIDIA GPU with recent CUDA drivers.

1. Follow steps 1-4 of the Quick start of DQN Zoo.

    Install [Docker](http://docs.docker.com/) version 19.03 or later (for the `--gpus` flag).

    Install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit).

    Enable [sudoless docker](http://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

    Verify the previous steps were successful e.g. by running: \
    `docker run --gpus all --rm nvidia/cuda:11.1.1-base nvidia-smi`

2. Clone this repo and go to the directory

    ```
    git clone https://github.com/deepmind/sketch_dqn.git
    cd sketch_dqn
    ```

The directory tree closely follows that of DQN Zoo, so it is also possible to move
the `dqn_zoo/sketch_dqn` file under the existing `dqn_zoo` if you have DQN ZOO
installed.

## Usage
Run `run.sh`. The default hyperparameters are as reported in the paper.

We note the following key implementations details for Sketch-DQN.

* Various sketch feature functions are in `dqn_zoo/feature_maps.py`, together with
the computations of sketch Bellman coefficients $B_r$ and value-readout coefficients $\beta$.

* Network objects are located in `dqn/sketch_dqn/networks.py`. The network
used by Sketch-DQN is function `sketch_atari_network`, which relies on
existing modules already defined in DQN Zoo.

* The Sketch-DQN agent uses both the network and pre-computed coefficients.

## Citing this work

If you use Sketch DQN in your research, please cite using

```
@article{wenliang2023sketchdqn,
  title = {Distributional Bellman Operators over Mean Embeddings},
  author = {
    Li Kevin Wenliang and
    Gr{\'{e}}goire Del{\'{e}}tang and
    Matthew Aitchison and
    Marcus Hutter and
    Anian Ruoss and
    Arthur Gretton and
    Mark Rowland
  },
  journal={arXiv preprint},
  year = {2023},
}
```

## License and disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
