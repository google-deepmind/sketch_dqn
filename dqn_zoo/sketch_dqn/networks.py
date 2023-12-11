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

"""DQN agent network components and implementation."""

from typing import Any, Callable, NamedTuple, Tuple, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


NetworkFn = Callable[..., Any]


class SketchNetworkOutputs(NamedTuple):
  q_values: jnp.ndarray
  q_sketch: jnp.ndarray


def _dqn_default_initializer(
    num_input_units: int) -> hk.initializers.Initializer:
  """Default initialization scheme inherited from past implementations of DQN.

  This scheme was historically used to initialize all weights and biases
  in convolutional and linear layers of DQN-type agents' networks.
  It initializes each weight as an independent uniform sample from [`-c`, `c`],
  where `c = 1 / np.sqrt(num_input_units)`, and `num_input_units` is the number
  of input units affecting a single output unit in the given layer, i.e. the
  total number of inputs in the case of linear (dense) layers, and
  `num_input_channels * kernel_width * kernel_height` in the case of
  convolutional layers.

  Args:
    num_input_units: number of input units to a single output unit of the layer.

  Returns:
    Haiku weight initializer.
  """
  max_val = np.sqrt(1 / num_input_units)
  return hk.initializers.RandomUniform(-max_val, max_val)


@jax.custom_gradient
def _scale_gradient(x, scale: float):
  """Identity function that scales the gradient flowing backwards."""
  return x, lambda g: (scale * g, 0.)


def conv(
    num_features: int,
    kernel_shape: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
) -> NetworkFn:
  """Convolutional layer with DQN's legacy weight initialization scheme."""

  def net_fn(inputs):
    """Function representing conv layer with DQN's legacy initialization."""
    num_input_units = inputs.shape[-1] * kernel_shape[0] * kernel_shape[1]
    initializer = _dqn_default_initializer(num_input_units)
    layer = hk.Conv2D(
        num_features,
        kernel_shape=kernel_shape,
        stride=stride,
        w_init=initializer,
        b_init=initializer,
        padding='VALID')
    return layer(inputs)

  return net_fn


def linear(num_outputs: int, with_bias: bool = True) -> NetworkFn:
  """Linear layer with DQN's legacy weight initialization scheme."""

  def net_fn(inputs):
    """Function representing linear layer with DQN's legacy initialization."""
    initializer = _dqn_default_initializer(inputs.shape[-1])
    layer = hk.Linear(
        num_outputs,
        with_bias=with_bias,
        w_init=initializer,
        b_init=initializer)
    return layer(inputs)

  return net_fn


def linear_with_shared_bias(num_outputs: int) -> NetworkFn:
  """Linear layer with single shared bias instead of one bias per output."""

  def layer_fn(inputs):
    """Function representing a linear layer with single shared bias."""
    initializer = _dqn_default_initializer(inputs.shape[-1])
    bias_free_linear = hk.Linear(
        num_outputs, with_bias=False, w_init=initializer)
    linear_output = bias_free_linear(inputs)
    bias = hk.get_parameter('b', [1], inputs.dtype, init=initializer)
    bias = jnp.broadcast_to(bias, linear_output.shape)
    return linear_output + bias

  return layer_fn


def dqn_torso() -> NetworkFn:
  """DQN convolutional torso.

  Includes scaling from [`0`, `255`] (`uint8`) to [`0`, `1`] (`float32`)`.

  Returns:
    Network function that `haiku.transform` can be called on.
  """

  def net_fn(inputs):
    """Function representing convolutional torso for a DQN Q-network."""
    network = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        conv(32, kernel_shape=(8, 8), stride=(4, 4)),
        jax.nn.relu,
        conv(64, kernel_shape=(4, 4), stride=(2, 2)),
        jax.nn.relu,
        conv(64, kernel_shape=(3, 3), stride=(1, 1)),
        jax.nn.relu,
        hk.Flatten(),
    ])
    return network(inputs)

  return net_fn


def dqn_value_head(num_actions: int, shared_bias: bool = False) -> NetworkFn:
  """Regular DQN Q-value head with single hidden layer."""

  last_layer = linear_with_shared_bias if shared_bias else linear

  def net_fn(inputs):
    """Function representing value head for a DQN Q-network."""
    network = hk.Sequential([
        linear(512),
        jax.nn.relu,
        last_layer(num_actions),
    ])
    return network(inputs)

  return net_fn


def sketch_atari_network(
    num_actions: int,
    identity_coeffs: chex.Array,
    feature_type: str,
    affine: bool = False,
) -> NetworkFn:
  """Sketch-DQN network.

  Args:
    num_actions: Number of actions.
    identity_coeffs: The linear coefficients for approximating the identity
      function from sketch features.
    feature_type: Base feature function.
    affine: Whether the feature contains a bias term.

  Returns:
    A network function to be.
  """

  chex.assert_rank(identity_coeffs, 2)

  # Set the last layer of the network to be consistent on value range.
  if feature_type in ['tanh', 'morbid']:
    # These features have range [-1, 1]
    final_nl_type = 'tanh'
  elif feature_type in ['gaussian', 'parabolic', 'quartic', 'sigmoid', 'erf']:
    # These features have range [0, 1]
    final_nl_type = 'sigmoid'
  else:
    raise ValueError(f'Unsupported feature type: {feature_type}')

  # The zero'th axis may contain a bias for affine transformation.
  num_features = identity_coeffs.shape[0]
  if affine:
    # The network should only predict the linear part, not the bias part.
    num_features -= 1

  if final_nl_type == 'sigmoid':
    final_nl = jax.nn.sigmoid
  elif final_nl_type == 'tanh':
    final_nl = jnp.tanh
  elif final_nl_type == 'softmax':
    final_nl = lambda x: x
  else:
    raise ValueError('Final nonlinearity is invalid.')

  def net_fn(inputs):
    """Function representing Sketch-DQN Q-network."""

    network = hk.Sequential(
        [dqn_torso(),
         dqn_value_head(num_features * num_actions),
         hk.Reshape([num_features, num_actions]),
         final_nl,
        ]
    )

    q_sketch = network(inputs)

    if final_nl_type == 'softmax':
      q_sketch = jax.nn.softmax(q_sketch, axis=1)
      q_sketch = jnp.cumsum(q_sketch, axis=1)[:, ::-1, :]

    if affine:
      q_sketch = jnp.pad(
          q_sketch, ((0, 0), (0, 1), (0, 0)), constant_values=1.0
      )

    # The identity coefficients have a trailing dimension with shape 1.
    q_values = jnp.einsum('ikj,kl->ijl', q_sketch, identity_coeffs)
    q_values = jnp.squeeze(q_values, axis=-1)
    q_values = jax.lax.stop_gradient(q_values)
    return SketchNetworkOutputs(q_sketch=q_sketch, q_values=q_values)

  return net_fn
