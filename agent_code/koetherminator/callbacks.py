import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.agents.ppo import ppo_agent
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork

from BombermanEnvironment import BombermanEnvironment
from BombermanEnvironment import BombermanGame


def setup(self):
    self.policy = tf.compat.v2.saved_model.load('../../policy')
    self.policy_state = self.policy.get_initial_state(batch_size=1)




def act(self, game_state: dict):
    self.policy.action()

    return action