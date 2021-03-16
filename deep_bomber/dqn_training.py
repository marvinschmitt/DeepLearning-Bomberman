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
from tf_agents.networks.q_network import QNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.networks.q_network import QNetwork

from adapter.bomberman_adapter import BombermanEnvironment

N_PARALLEL_ENVIRONMENTS = 4 # not yet (sadFace)
INITIAL_COLLECT_STEPS = 20000


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)

    for iteration in range(n_iterations):
        current_metrics = []

        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)

        train_loss = agent.train(trajectories)
        all_train_loss.append(train_loss.loss.numpy())

        for i in range(len(train_metrics)):
            current_metrics.append(train_metrics[i].result().numpy())

        all_metrics.append(current_metrics)

        if iteration % 500 == 0:
            print("\nIteration: {}, loss:{:.2f}".format(iteration, train_loss.loss.numpy()))

            for i in range(len(train_metrics)):
                print('{}: {}'.format(train_metrics[i].name, train_metrics[i].result().numpy()))


if __name__ == '__main__':
    eval_tf_env = tf_py_environment.TFPyEnvironment(BombermanEnvironment())
    # tf_env = tf_py_environment.TFPyEnvironment(
    #    parallel_py_environment.ParallelPyEnvironment(
    #        [BombermanEnvironment] * N_PARALLEL_ENVIRONMENTS
    #    ))

    tf_env = tf_py_environment.TFPyEnvironment(BombermanEnvironment())

    q_net = QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        conv_layer_params=[(32, 8, 1), (32, 4, 1)],
        fc_layer_params=[32, 64, 128]
    )

    train_step = tf.Variable(0)
    update_period = 4
    optimizer = tf.keras.optimizers.Adam()  # todo fine tune

    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0,
        decay_steps=250000 // update_period,
        end_learning_rate=0.01
    )

    agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=0.99,
        train_step_counter=train_step,
        epsilon_greedy=lambda: epsilon_fn(train_step)
    )

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=1000000
    )
    replay_buffer_observer = replay_buffer.add_batch

    train_metrics = [
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric()
    ]

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer_observer]+train_metrics,
        num_steps=update_period
    )

    initial_collect_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

    initial_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch, ShowProgress(INITIAL_COLLECT_STEPS)],
        num_steps=INITIAL_COLLECT_STEPS
    )
    final_time_step, final_policy_state = initial_driver.run()

    dataset = replay_buffer.as_dataset(sample_batch_size=64, num_steps=2, num_parallel_calls=3).prefetch(3)

    all_train_loss = []
    all_metrics = []

    # training here
    train_agent(50000)

    policy_saver.PolicySaver(agent.policy).save("policy")




    #saved_model = policy_saver.PolicySaver(eval_policy)
    #saved_model.save("policy")
