#!/bin/bash
# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# This script does the following:
# 1. Clones the DQN Zoo repository.
# 2. Builds a Docker image with all necessary dependencies and runs unit tests.
# 3. Starts a short run of DQN on Pong in a GPU-accelerated container.

# Before running:
# * Install Docker version 19.03 or later for the --gpus options.
# * Install NVIDIA Container Toolkit.
# * Enable sudoless docker.
# * Verify installation, e.g. with:
#   `docker run --gpus all --rm nvidia/cuda:11.1.1-base nvidia-smi`.

# To remove all containers run:
# `docker rm -vf $(docker ps -a -q)`

# To remove all images run:
# `docker rmi -f $(docker images -a -q)`

set -u -e  # Check for uninitialized variables and exit if any command fails.

echo "Remove container if it exists"
docker rm dqn_zoo_sketch || true

echo "Remove image if it exists"
docker rmi dqn_zoo_sketch:latest || true

echo "Build image with tag 'dqn_zoo:latest' and run tests"
docker build -t dqn_zoo_sketch:latest .

echo "Run DQN on GPU in a container named dqn_zoo_sketch"
docker run --gpus all --name dqn_zoo_sketch dqn_zoo_sketch:latest \
    -m dqn_zoo.sketch_dqn.run_atari \
    --jax_platform_name=gpu \
    --environment_name=pong \

echo "Finished"
