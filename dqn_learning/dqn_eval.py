import base64

import matplotlib.pyplot as plt
import numpy as np

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
from tf_agents.networks.q_network import QNetwork

from BombermanEnvironment import BombermanEnvironment


# from tf docu
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]




if __name__ == '__main__':
    eval_tf_env = tf_py_environment.TFPyEnvironment(BombermanEnvironment(replay=True))

    time_step = eval_tf_env.reset()

    policy = tf.compat.v2.saved_model.load("policy")
    policy_state = policy.get_initial_state(batch_size=eval_tf_env.batch_size)

    while not time_step.is_last():
        policy_step = policy.action(time_step, policy_state)
        policy_state = policy_step.state
        time_step = eval_tf_env.step(policy_step.action)