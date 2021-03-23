import os

import tensorflow as tf
from tf_agents.environments import tf_py_environment

from adapter.bomberman_adapter import BombermanEnvironment, BombermanGame


if __name__ == '__main__':
    eval_tf_env = tf_py_environment.TFPyEnvironment(BombermanEnvironment(mode='imitator', replay=True))

    time_step = eval_tf_env.reset()

    policy = tf.saved_model.load("policy_imitator")
    policy_state = policy.get_initial_state(batch_size=eval_tf_env.batch_size)

    while not time_step.is_last():
        policy_step = policy.action(time_step, policy_state)
        policy_state = policy_step.state
        print(BombermanGame.ACTIONS[int(policy_step.action)])
        time_step = eval_tf_env.step(int(policy_step.action))

    replay_files = os.listdir("replays")
    replay_file = sorted(replay_files)[-1]

    command = f"python main.py replay \"replays/{replay_file}\" --update-interval 0.03 --fps 60"
    os.system(command)
