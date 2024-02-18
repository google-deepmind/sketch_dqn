# Copyright 2024 DeepMind Technologies Limited
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

"""Sketch basis functions."""

from typing import Callable, Optional, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np


def coeff_by_regression(
    grid: chex.Array,
    feature_fun: Callable[[chex.Array], chex.Array],
    target_fun: Callable[[chex.Array], chex.Array],
    regularization_weight: float = 0.0,
    error_tol: Optional[float] = None,
) -> tuple[chex.Array, float, Optional[int], Optional[int]]:
  """Compute the regression coefficients to approximate the target function.

  Args:
    grid: The grid points on which to measure the squared error.
    feature_fun: Feature functions.
    target_fun: The target function to be approximated by feature functions.
    regularization_weight: If greater than 0, then this is the l-2 weight of the
      l-2 regularizer.
    error_tol: Tolerance of the maximum squared error over the grid.

  Returns:
    The coefficients, the max. absolute error, and optionally the rank and
    singular values.
  """

  feature_fv = np.asarray(feature_fun(grid), dtype='float64')
  target_fv = np.asarray(target_fun(grid), dtype='float64')

  chex.assert_rank(target_fv, 2)
  chex.assert_equal_shape_prefix([grid, feature_fv, target_fv], 1)
  num_train = grid.shape[0]

  num_features = feature_fv.shape[1]

  if regularization_weight > 0:
    # If regularise, then use the solve function.
    reg = np.eye(num_features) * regularization_weight
    coeff = np.linalg.solve(
        feature_fv.T @ feature_fv / num_train + reg,
        feature_fv.T @ target_fv / num_train)
    rank = None
    sing_vals = None
  else:
    # If not, then use lstsq.
    coeff, _, rank, sing_vals = np.linalg.lstsq(feature_fv, target_fv)

  # Test if the coeffs give a small prediction error.
  pred = feature_fv @ coeff
  error = np.abs(pred - target_fv).max()

  if error_tol and error > error_tol:
    raise ValueError(
        f'The error {error:.3f} is larger than the tolerance {error_tol}.'
        ' Update the feature parameters or increase the tolerance.'
    )

  return coeff, error, rank, sing_vals


def _stretch_biases(biases: chex.Array, power: float) -> chex.Array:
  """Stretches the biases with a power function to get non-uniform spacing.

  New bias is proportional to b^power, scaled to have the same max as the
  original biases.

  Args:
    biases: A sorted array with anti-symmetric entries where biases[i] =
      biases[-i-1].
    power: the exponent of the power function.

  Returns:
    New biases with scaled spacings.
  """

  chex.assert_trees_all_close(biases[::-1], -biases, atol=1e-5, rtol=1e-5)

  max_bias = biases.max()
  normalised_bias = biases / max_bias
  biases = jnp.sign(normalised_bias) * jnp.abs(normalised_bias) ** power
  biases *= max_bias
  return biases


def build_feature_map(
    name: str,
    biases: chex.Array,
    slope: float,
    power: float = 1.0,
    append_one: bool = False,
) -> Callable[[chex.Array], chex.Array]:
  """Builds a feature function from parameters.

  Args:
    name: Type of the nonlinear function.
    biases: The biases of each feature (shifts along the input domain).
    slope: The steepness of the functions.
    power: Power index to stretch the biases.
    append_one: Whether to append a constant 1 to the features.

  Returns:
    A feature function.
  """

  biases = _stretch_biases(biases, power)
  def feature_map(x):

    x = (x[:, None] - biases[None, :]) * slope

    if name == 'sigmoid':
      fun_value = jax.nn.sigmoid(x)

    elif name == 'tanh':
      fun_value = jnp.tanh(x)

    elif name == 'gaussian':
      fun_value = jnp.exp(- x ** 2 / 2)

    elif name == 'morbid':
      fun_value = jnp.exp(- x ** 2 / 2) * jnp.cos(2 * x)

    elif name == 'erf':
      fun_value = (jax.lax.erf(0.5 * x) + 1) / 2.0

    elif name == 'parabolic':
      x = jnp.abs(x)
      fun_value = jnp.where(x < 1, 0.75 * (1 - x**2), 0.0)

    elif name == 'quartic':
      x = jnp.abs(x)
      fun_value = jnp.where(x < 1, 15.0 / 16.0 * (1 - x**2) ** 2, 0.0)

    else:
      raise ValueError('Invalid feature function name.')

    if append_one:
      fun_value = np.pad(fun_value, ((0, 0), (0, 1)), constant_values=1)

    return fun_value

  return feature_map


def build_coefficients(
    grid_lim: float,
    grid_count: int,
    bias_lim: float,
    num_features: int,
    valid_rewards: Sequence[float],
    valid_gammas: Sequence[float],
    slope: float,
    power: float,
    affine: bool,
    feature_type: str,
    reg_weight: float,
    error_tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Build mean and Bellman coefficients for the sketch network."""

  grid = jnp.linspace(-grid_lim, grid_lim, grid_count)
  biases = jnp.linspace(-bias_lim, bias_lim, num_features)

  feature_fun = build_feature_map(
      name=feature_type,
      biases=biases,
      slope=slope,
      power=power,
      append_one=affine,
  )
  identity_fun = lambda x: x[:, None]
  identity_coeff, _, _, _ = coeff_by_regression(
      grid=grid,
      feature_fun=feature_fun,
      target_fun=identity_fun,
      regularization_weight=reg_weight,
      error_tol=error_tol,
  )
  identity_coeff = jnp.asarray(identity_coeff)

  bellman_coeffs = []

  for r in valid_rewards:
    bellman_coeffs.append([])
    for gamma in valid_gammas:
      shifted_feature_fun = lambda x: feature_fun(r + x * gamma)  # pylint:disable=cell-var-from-loop
      coeff, _, _, _ = coeff_by_regression(
          grid=grid,
          feature_fun=feature_fun,
          target_fun=shifted_feature_fun,
          regularization_weight=reg_weight,
          error_tol=error_tol,
      )
      bellman_coeffs[-1].append(coeff)
  bellman_coeffs = jnp.asarray(bellman_coeffs)
  return identity_coeff, bellman_coeffs
