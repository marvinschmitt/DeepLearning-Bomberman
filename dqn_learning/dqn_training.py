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

from dqn_learning.BombermanEnvironment import BombermanEnvironment

N_PARALLEL_ENVIRONMENTS = 4 # not yet (sadFace)
INITIAL_COLLECT_STEPS = 100
COLLECT_EPISODES_PER_ITERATION = 6


def create_actor_value_networks(tf_env):
    actor_net = ActorDistributionRnnNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        conv_layer_params=[(32, 8, 1), (16, 4, 1)],
        input_fc_layer_params=(256,),
        lstm_size=(256,),
        output_fc_layer_params=(128,)
    )

    value_net = ValueRnnNetwork(
        tf_env.observation_spec(),
        conv_layer_params=[(32, 8, 1), (16, 4, 1)],
        input_fc_layer_params=(256,),
        lstm_size=(256,),
        output_fc_layer_params=(128,),
        activation_fn=tf.nn.elu
    )

    return actor_net, value_net

def create_q_network():
    q_net = None
    return q_net


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


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


def train_step():
    trajectories = replay_buffer.as_dataset(
        sample_batch_size=tf_env.batch_size,
        single_deterministic_pass=True
    )
    return agent.train(experience=trajectories)


def evaluate():
    pass


if __name__ == '__main__':
    eval_tf_env = tf_py_environment.TFPyEnvironment(BombermanEnvironment())
    # tf_env = tf_py_environment.TFPyEnvironment(
    #    parallel_py_environment.ParallelPyEnvironment(
    #        [BombermanEnvironment] * N_PARALLEL_ENVIRONMENTS
    #    ))

    tf_env = tf_py_environment.TFPyEnvironment(BombermanEnvironment())
    optimizer = tf.keras.optimizers.Adam()  # todo fine tune


    """
    q_network = QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec()# ,
        # conv_layer_params=[(4, 4, 1)]
    )
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_network,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter
    )
    """

    actor_net, value_net = create_actor_value_networks(tf_env)

    agent = ppo_agent.PPOAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        optimizer,
        actor_net,
        value_net,
        num_epochs=25,
        gradient_clipping=0.5,
        entropy_regularization=1e-2,
        importance_ratio_clipping=0.2,
        use_gae=True,
        use_td_lambda_return=True
    )


    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy


    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=401
    )

    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]


    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env, collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=COLLECT_EPISODES_PER_ITERATION
    )

    saved_model = policy_saver.PolicySaver(eval_policy)

    for ep in range(100):
        collect_driver.run()
        total_loss, _ = train_step()
        replay_buffer.clear()
        print(f"Finished Ep {ep}")

    saved_model.save("policy")


    1 + 1
