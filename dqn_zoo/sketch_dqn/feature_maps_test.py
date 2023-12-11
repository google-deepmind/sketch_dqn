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

"""Tests for basis functions."""

from absl.testing import absltest
import chex
import numpy as np

from dqn_zoo.sketch_dqn import feature_maps


class TestBaisFunction(absltest.TestCase):

  def setUp(self):
    super().setUp()
    grid_lim = 10
    bias_lim = 12
    self.num_features = 100
    self.num_grid = 10_000
    self.biases = np.linspace(-bias_lim, bias_lim, self.num_features)
    self.grid = np.linspace(-grid_lim, grid_lim, self.num_grid)

  def test_regression(self):
    feature_map = feature_maps.build_feature_map(
        name='sigmoid',
        biases=self.biases,
        slope=3.0,
    )

    feature_values = feature_map(self.grid)
    chex.assert_shape(feature_values, (self.num_grid, self.num_features))
    target_fun = lambda x: feature_map(0.9 * x + 0.5)
    target_values = target_fun(self.grid)

    bellman_coeff = feature_maps.coeff_by_regression(
        grid=self.grid,
        feature_fun=feature_map,
        target_fun=target_fun,
        regularization_weight=1e-9,
    )[0]
    assert np.max(np.abs(feature_values @ bellman_coeff - target_values)) < 0.01

  def test_affine_feature_map(self):
    affine_feature_map = feature_maps.build_feature_map(
        name='sigmoid',
        biases=self.biases,
        slope=3.0,
        append_one=True,
    )
    feature_values = affine_feature_map(self.grid)
    chex.assert_shape(feature_values, (self.num_grid, self.num_features + 1))


if __name__ == '__main__':
  absltest.main()
