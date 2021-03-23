import tensorflow as tf
import pickle

from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.networks.q_network import QNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork

from adapter.bomberman_adapter import BombermanEnvironment


N_PARALLEL_ENVIRONMENTS = 4  # not yet (sadFace)
INITIAL_COLLECT_STEPS = 1000
EVAL_INTERVAL = 1000
NUM_EVAL_EPISODES = 50
BATCH_SIZE = 64
N_STEPS = 1000


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        state = policy.get_initial_state(environment.batch_size)
        episode_return = 0.0

        while not time_step.is_last():
            policy_step = policy.action(time_step, state)
            state = policy_step.state
            time_step = environment.step(policy_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def create_networks(tf_env):
    actor_net = ActorDistributionRnnNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        conv_layer_params=[(16, 4, 2), (32, 2, 1)],
        input_fc_layer_params=(256,),
        lstm_size=(256,),
        output_fc_layer_params=(128,))

    value_net = ValueRnnNetwork(
        tf_env.observation_spec(),
        conv_layer_params=[(16, 4, 2), (32, 2, 1)],
        input_fc_layer_params=(256,),
        lstm_size=(256,),
        output_fc_layer_params=(128,),
        activation_fn=tf.nn.elu)

    return actor_net, value_net


def train_agent(n_steps, save_each=10000, print_each=1000):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)

    #for iteration in range(1, n_iterations+1):
    step = agent.train_step_counter.numpy()
    while step < n_steps:
        step = agent.train_step_counter.numpy()
        current_metrics = []

        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)

        train_loss = agent.train(trajectories)
        all_train_loss.append(train_loss.loss.numpy())

        for i in range(len(train_metrics)):
            current_metrics.append(train_metrics[i].result().numpy())

        all_metrics.append(current_metrics)

        if step % print_each == 0:
            print("\nIteration: {}, loss:{:.2f}".format(step, train_loss.loss.numpy()))

            for i in range(len(train_metrics)):
                print('{}: {}'.format(train_metrics[i].name, train_metrics[i].result().numpy()))

        if step % EVAL_INTERVAL == 0:
            avg_return = compute_avg_return(eval_tf_env, agent.policy, NUM_EVAL_EPISODES)
            print(f'Step = {step}, Average Return = {avg_return}')
            returns.append((step, avg_return))

        if step % save_each == 0:
            policy_save_handler.save("policies/policy_ppo")


if __name__ == '__main__':
    # tf_env = tf_py_environment.TFPyEnvironment(
    #   parallel_py_environment.ParallelPyEnvironment(
    #       [BombermanEnvironment] * N_PARALLEL_ENVIRONMENTS
    #   ))

    tf_env = tf_py_environment.TFPyEnvironment(BombermanEnvironment(mode="no_bomb"))
    eval_tf_env = tf_py_environment.TFPyEnvironment(BombermanEnvironment(mode="no_bomb"))

    actor_net, value_net = create_networks(tf_env)

    train_step = tf.Variable(0)
    update_period = 4
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # todo fine tune


    agent = ppo_agent.PPOAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=25,
        gradient_clipping=0.5,
        entropy_regularization=1e-2,
        importance_ratio_clipping=0.2,
        use_gae=True,
        use_td_lambda_return=True
    )

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=10000  # todo finetune
    )
    replay_buffer_observer = replay_buffer.add_batch

    train_metrics = [
        tf_metrics.AverageReturnMetric(batch_size=tf_env.batch_size),  # todo: doesn't work. just sums rewards
        tf_metrics.AverageEpisodeLengthMetric(batch_size=tf_env.batch_size)
    ]

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer_observer]+train_metrics,
        num_episodes=10
    )

    # initial_collect_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
    #
    # initial_driver = dynamic_step_driver.DynamicStepDriver(
    #     tf_env,
    #     initial_collect_policy,
    #     observers=[replay_buffer.add_batch, ShowProgress(INITIAL_COLLECT_STEPS)],
    #     num_steps=INITIAL_COLLECT_STEPS
    # )
    # final_time_step, final_policy_state = initial_driver.run()

    dataset = replay_buffer.\
        as_dataset(sample_batch_size=BATCH_SIZE, num_steps=2, num_parallel_calls=3).\
        prefetch(3)

    agent.train = common.function(agent.train)

    all_train_loss = []
    all_metrics = []
    returns = []

    checkpoint_dir = "checkpoints/checkpoint_ppo"
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=train_step
    )
    # train_checkpointer.initialize_or_restore()
    # train_step = tf.compat.v1.train.get_global_step()
    policy_save_handler = policy_saver.PolicySaver(agent.policy)

    # training here
    train_agent(N_STEPS)


    # save at end in every case

    policy_save_handler.save("policies/policy_ppo")
