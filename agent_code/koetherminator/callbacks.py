import tensorflow as tf

from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.agents.ppo import ppo_agent
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


def setup(self):
    self.policy = tf.compat.v2.saved_model.load('../../policy')
    self.policy_state = self.policy.get_initial_state(batch_size=1)




def act(self, game_state: dict):
    self.policy.action()

    return action