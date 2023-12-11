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

"""Tests for networks."""

from absl.testing import absltest
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tree

from dqn_zoo.sketch_dqn import networks


def _sample_input(input_shape):
  return jnp.zeros((1,) + input_shape, dtype=jnp.float32)


class SimpleLayersTest(absltest.TestCase):

  def test_linear(self):
    layer = hk.transform(networks.linear(4))
    params = layer.init(jax.random.PRNGKey(1), _sample_input((3,)))
    self.assertCountEqual(['linear'], params)
    lin_params = params['linear']
    self.assertCountEqual(['w', 'b'], lin_params)
    chex.assert_shape(lin_params['w'], (3, 4))
    chex.assert_shape(lin_params['b'], (4,))

  def test_conv(self):
    layer = hk.transform(networks.conv(4, (3, 3), 2))
    params = layer.init(jax.random.PRNGKey(1), _sample_input((7, 7, 3)))
    self.assertCountEqual(['conv2_d'], params)
    conv_params = params['conv2_d']
    self.assertCountEqual(['w', 'b'], conv_params)
    chex.assert_shape(conv_params['w'], (3, 3, 3, 4))
    chex.assert_shape(conv_params['b'], (4,))


class LinearWithSharedBiasTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rng_key = jax.random.PRNGKey(1)
    self.init_rng_key, self.apply_rng_key = jax.random.split(rng_key)
    self.input_shape = (4,)
    self.output_shape = (3,)
    self.weights_shape = (self.input_shape[0], self.output_shape[0])
    network_fn = networks.linear_with_shared_bias(self.output_shape[0])
    self.network = hk.transform(network_fn)

  def test_bias_parameter_shape(self):
    params = self.network.init(
        self.init_rng_key, _sample_input(self.input_shape)
    )
    self.assertLen(tree.flatten(params), 2)

    def check_params(path, param):
      if path[-1] == 'b':
        self.assertNotEqual(self.output_shape, param.shape)
        chex.assert_shape(param, (1,))
      elif path[-1] == 'w':
        chex.assert_shape(param, self.weights_shape)
      else:
        self.fail('Unexpected parameter %s.' % path)

    tree.map_structure_with_path(check_params, params)

  def test_output_shares_bias(self):
    bias = 1.23
    params = self.network.init(
        self.init_rng_key, _sample_input(self.input_shape)
    )

    def replace_params(path, param):
      if path[-1] == 'b':
        return jnp.ones_like(param) * bias
      else:
        return jnp.zeros_like(param)

    params = tree.map_structure_with_path(replace_params, params)
    output = self.network.apply(
        params, self.apply_rng_key, jnp.zeros((1,) + self.input_shape)
    )
    chex.assert_shape(output, (1,) + self.output_shape)
    np.testing.assert_allclose([bias] * self.output_shape[0], list(output[0]))


class SketchDQNTest(absltest.TestCase):

  def test_shape(self):
    rng_key = jax.random.PRNGKey(1)
    self.init_rng_key, self.apply_rng_key = jax.random.split(rng_key)
    self.num_feature = 100
    dummpy_identity_coeffs = jnp.zeros((self.num_feature, 1))
    num_actions = 4
    network_fn = networks.sketch_atari_network(
        num_actions=num_actions,
        identity_coeffs=dummpy_identity_coeffs,
        feature_type='sigmoid',
    )
    self.network = hk.transform(network_fn)
    params = self.network.init(self.init_rng_key, _sample_input((96, 96, 4)))
    output = self.network.apply(
        params, self.apply_rng_key, _sample_input((96, 96, 4))
    )
    chex.assert_shape(output.q_sketch, (1, self.num_feature, num_actions))
    chex.assert_shape(output.q_values, (1, num_actions))


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
