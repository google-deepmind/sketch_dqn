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

"""A sketch DQN agent training on Atari."""

import collections
import itertools
import sys
import typing

from absl import app
from absl import flags
from absl import logging
import chex
import dm_env
import haiku as hk
import jax
import jax.config
import numpy as np
import optax

from dqn_zoo import atari_data
from dqn_zoo import gym_atari
from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib
from dqn_zoo.sketch_dqn import agent
from dqn_zoo.sketch_dqn import feature_maps
from dqn_zoo.sketch_dqn import networks


# Relevant flag values are expressed in terms of environment frames.
_ENVIRONMENT_NAME = flags.DEFINE_string('environment_name', 'pong', '')
_ENVIRONMENT_HEIGHT = flags.DEFINE_integer('environment_height', 84, '')
_ENVIRONMENT_WIDTH = flags.DEFINE_integer('environment_width', 84, '')
_REPLAY_CAPACITY = flags.DEFINE_integer('replay_capacity', int(1e6), '')
_COMPRESS_STATE = flags.DEFINE_bool('compress_state', True, '')
_MIN_REPLAY_CAPACITY_FRACTION = flags.DEFINE_float(
    'min_replay_capacity_fraction', 0.05, ''
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, '')
_MAX_FRAMES_PER_EPISODE = flags.DEFINE_integer(
    'max_frames_per_episode', 108000, ''
)  # 30 mins.
_NUM_ACTION_REPEATS = flags.DEFINE_integer('num_action_repeats', 4, '')
_NUM_STACKED_FRAMES = flags.DEFINE_integer('num_stacked_frames', 4, '')
_EXPLORATION_EPSILON_BEGIN_VALUE = flags.DEFINE_float(
    'exploration_epsilon_begin_value', 1.0, ''
)
_EXPLORATION_EPSILON_END_VALUE = flags.DEFINE_float(
    'exploration_epsilon_end_value', 0.01, ''
)
_EXPLORATION_EPSILON_DECAY_FRAME_FRACTION = flags.DEFINE_float(
    'exploration_epsilon_decay_frame_fraction', 0.02, ''
)
_EVAL_EXPLORATION_EPSILON = flags.DEFINE_float(
    'eval_exploration_epsilon', 0.001, ''
)
_TARGET_NETWORK_UPDATE_PERIOD = flags.DEFINE_integer(
    'target_network_update_period', int(4e4), ''
)
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.00005, '')
_OPTIMIZER_EPSILON = flags.DEFINE_float('optimizer_epsilon', 0.01 / 32, '')
_ADDITIONAL_DISCOUNT = flags.DEFINE_float('additional_discount', 0.99, '')
_MAX_GLOBAL_GRAD_NORM = flags.DEFINE_float('max_global_grad_norm', 10.0, '')
_SEED = flags.DEFINE_integer('seed', 1, '')  # GPU may introduce nondeterminism.
_NUM_ITERATIONS = flags.DEFINE_integer('num_iterations', 200, '')
_NUM_TRAIN_FRAMES = flags.DEFINE_integer(
    'num_train_frames', int(1e6), ''
)  # Per iteration.
_NUM_EVAL_FRAMES = flags.DEFINE_integer(
    'num_eval_frames', int(5e5), ''
)  # Per iteration.
_LEARN_PERIOD = flags.DEFINE_integer('learn_period', 16, '')
_RESULTS_CSV_PATH = flags.DEFINE_string(
    'results_csv_path', '/tmp/results.csv', ''
)

# Sketch DQN settings
_FEATURE_TYPE = flags.DEFINE_enum(
    'feature_type',
    'sigmoid',
    ['sigmoid', 'gaussian', 'parabolic', 'quartic', 'erf'],
    '',
)
_NUM_FEATURES = flags.DEFINE_integer('num_features', 401, '')
_REG_WEIGHT = flags.DEFINE_float('reg_weight', 1e-9, '')
_ERROR_TOL = flags.DEFINE_float('error_tol', 0.01, '')
_GRID_LIM = flags.DEFINE_float('grid_lim', 10.0, '')
_BIAS_LIM = flags.DEFINE_float('bias_lim', 12.0, '')
_GRID_COUNT = flags.DEFINE_integer('grid_count', 10000, '')
_SLOPE = flags.DEFINE_float('slope', 10.0, '')
_AFFINE = flags.DEFINE_bool('affine', True, '')
_POWER = flags.DEFINE_float('power', 1.0, '')


_MAX_ABS_REWARD_VALUE = 1.0
_VALID_REWARDS = [-1, 0, 1]


def main(argv):
  """Trains Sketch-DQN agent on Atari."""
  del argv
  logging.info(
      'Sketch-DQN on Atari on %s.', jax.lib.xla_bridge.get_backend().platform
  )
  random_state = np.random.RandomState(_SEED.value)
  rng_key = jax.random.PRNGKey(
      random_state.randint(-sys.maxsize - 1, sys.maxsize + 1, dtype=np.int64)
  )

  if _RESULTS_CSV_PATH.value:
    writer = parts.CsvWriter(_RESULTS_CSV_PATH.value)
  else:
    writer = parts.NullWriter()

  def environment_builder():
    """Creates Atari environment."""
    env = gym_atari.GymAtari(
        _ENVIRONMENT_NAME.value, seed=random_state.randint(1, 2**32)
    )
    return gym_atari.RandomNoopsEnvironmentWrapper(
        env,
        min_noop_steps=1,
        max_noop_steps=30,
        seed=random_state.randint(1, 2**32),
    )

  env = environment_builder()

  logging.info('Environment: %s', _ENVIRONMENT_NAME.value)
  logging.info('Action spec: %s', env.action_spec())
  logging.info('Observation spec: %s', env.observation_spec())

  valid_gammas = [0, _ADDITIONAL_DISCOUNT.value]
  valid_rewards = [-1.0, 0.0, 1.0]
  identity_coeff, bellman_coeffs = feature_maps.build_coefficients(
      grid_lim=_GRID_LIM.value,
      grid_count=_GRID_COUNT.value,
      bias_lim=_BIAS_LIM.value,
      num_features=_NUM_FEATURES.value,
      valid_gammas=valid_gammas,
      valid_rewards=valid_rewards,
      slope=_SLOPE.value,
      power=_POWER.value,
      affine=_AFFINE.value,
      feature_type=_FEATURE_TYPE.value,
      reg_weight=_REG_WEIGHT.value,
      error_tol=_ERROR_TOL.value,
  )

  num_actions = env.action_spec().num_values
  network_fn = networks.sketch_atari_network(
      num_actions,
      identity_coeff,
      feature_type=_FEATURE_TYPE.value,
      affine=_AFFINE.value,
  )

  network = hk.transform(network_fn)

  def preprocessor_builder():
    return processors.atari(
        additional_discount=_ADDITIONAL_DISCOUNT.value,
        max_abs_reward=_MAX_ABS_REWARD_VALUE,
        resize_shape=(_ENVIRONMENT_HEIGHT.value, _ENVIRONMENT_WIDTH.value),
        num_action_repeats=_NUM_ACTION_REPEATS.value,
        num_pooled_frames=2,
        zero_discount_on_life_loss=True,
        num_stacked_frames=_NUM_STACKED_FRAMES.value,
        grayscaling=True,
    )

  # Create sample network input from sample preprocessor output.
  sample_processed_timestep = preprocessor_builder()(env.reset())
  sample_processed_timestep = typing.cast(
      dm_env.TimeStep, sample_processed_timestep
  )
  sample_network_input = sample_processed_timestep.observation
  chex.assert_shape(
      sample_network_input,
      (
          _ENVIRONMENT_HEIGHT.value,
          _ENVIRONMENT_WIDTH.value,
          _NUM_STACKED_FRAMES.value,
      ),
  )

  exploration_epsilon_schedule = parts.LinearSchedule(
      begin_t=int(
          _MIN_REPLAY_CAPACITY_FRACTION.value
          * _REPLAY_CAPACITY.value
          * _NUM_ACTION_REPEATS.value
      ),
      decay_steps=int(
          _EXPLORATION_EPSILON_DECAY_FRAME_FRACTION.value
          * _NUM_ITERATIONS.value
          * _NUM_TRAIN_FRAMES.value
      ),
      begin_value=_EXPLORATION_EPSILON_BEGIN_VALUE.value,
      end_value=_EXPLORATION_EPSILON_END_VALUE.value,
  )

  if _COMPRESS_STATE.value:

    def encoder(transition):
      return transition._replace(
          s_tm1=replay_lib.compress_array(transition.s_tm1),
          s_t=replay_lib.compress_array(transition.s_t),
      )

    def decoder(transition):
      return transition._replace(
          s_tm1=replay_lib.uncompress_array(transition.s_tm1),
          s_t=replay_lib.uncompress_array(transition.s_t),
      )

  else:
    encoder = None
    decoder = None

  replay_structure = replay_lib.Transition(
      s_tm1=None,
      a_tm1=None,
      r_t=None,
      discount_t=None,
      s_t=None,
  )

  replay = replay_lib.TransitionReplay(
      _REPLAY_CAPACITY.value, replay_structure, random_state, encoder, decoder
  )

  optimizer = optax.adam(
      learning_rate=_LEARNING_RATE.value, eps=_OPTIMIZER_EPSILON.value
  )
  if _MAX_GLOBAL_GRAD_NORM.value > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(_MAX_GLOBAL_GRAD_NORM.value), optimizer
    )

  train_rng_key, eval_rng_key = jax.random.split(rng_key)

  train_agent = agent.SketchDqn(
      preprocessor=preprocessor_builder(),
      sample_network_input=sample_network_input,
      network=network,
      valid_rewards=_VALID_REWARDS,
      valid_gammas=valid_gammas,
      bellman_coeffs=bellman_coeffs,
      optimizer=optimizer,
      transition_accumulator=replay_lib.TransitionAccumulator(),
      replay=replay,
      batch_size=_BATCH_SIZE.value,
      exploration_epsilon=exploration_epsilon_schedule,
      min_replay_capacity_fraction=_MIN_REPLAY_CAPACITY_FRACTION.value,
      learn_period=_LEARN_PERIOD.value,
      target_network_update_period=_TARGET_NETWORK_UPDATE_PERIOD.value,
      rng_key=train_rng_key,
  )
  eval_agent = parts.EpsilonGreedyActor(
      preprocessor=preprocessor_builder(),
      network=network,
      exploration_epsilon=_EVAL_EXPLORATION_EPSILON.value,
      rng_key=eval_rng_key,
  )

  # Set up checkpointing.
  checkpoint = parts.NullCheckpoint()

  state = checkpoint.state
  state.iteration = 0
  state.train_agent = train_agent
  state.eval_agent = eval_agent
  state.random_state = random_state
  state.writer = writer
  if checkpoint.can_be_restored():
    checkpoint.restore()

  while state.iteration <= _NUM_ITERATIONS.value:
    # New environment for each iteration to allow for determinism if preempted.
    env = environment_builder()

    logging.info('Training iteration %d.', state.iteration)
    train_seq = parts.run_loop(train_agent, env, _MAX_FRAMES_PER_EPISODE.value)
    num_train_frames = 0 if state.iteration == 0 else _NUM_TRAIN_FRAMES.value
    train_seq_truncated = itertools.islice(train_seq, num_train_frames)
    train_trackers = parts.make_default_trackers(train_agent)
    train_stats = parts.generate_statistics(train_trackers, train_seq_truncated)

    logging.info('Evaluation iteration %d.', state.iteration)
    eval_agent.network_params = train_agent.online_params
    eval_seq = parts.run_loop(eval_agent, env, _MAX_FRAMES_PER_EPISODE.value)
    eval_seq_truncated = itertools.islice(eval_seq, _NUM_EVAL_FRAMES.value)
    eval_trackers = parts.make_default_trackers(eval_agent)
    eval_stats = parts.generate_statistics(eval_trackers, eval_seq_truncated)

    # Logging and checkpointing.
    human_normalized_score = atari_data.get_human_normalized_score(
        _ENVIRONMENT_NAME.value, eval_stats['episode_return']
    )
    capped_human_normalized_score = np.amin([1.0, human_normalized_score])
    log_output = [
        ('iteration', state.iteration, '%3d'),
        ('frame', state.iteration * _NUM_TRAIN_FRAMES.value, '%5d'),
        ('eval_episode_return', eval_stats['episode_return'], '% 2.2f'),
        ('train_episode_return', train_stats['episode_return'], '% 2.2f'),
        ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
        ('train_num_episodes', train_stats['num_episodes'], '%3d'),
        ('eval_frame_rate', eval_stats['step_rate'], '%4.0f'),
        ('train_frame_rate', train_stats['step_rate'], '%4.0f'),
        ('train_exploration_epsilon', train_agent.exploration_epsilon, '%.3f'),
        ('train_state_value', train_stats['state_value'], '%.3f'),
        ('normalized_return', human_normalized_score, '%.3f'),
        ('capped_normalized_return', capped_human_normalized_score, '%.3f'),
        ('human_gap', 1.0 - capped_human_normalized_score, '%.3f'),
    ]
    log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
    logging.info(log_output_str)
    writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
    state.iteration += 1
    checkpoint.save()

  writer.close()


if __name__ == '__main__':
  jax.config.update('jax_platform_name', 'gpu')  # Default to GPU.
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  jax.config.config_with_absl()
  app.run(main)
