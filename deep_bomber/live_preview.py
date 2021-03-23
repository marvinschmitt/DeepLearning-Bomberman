import os

import tensorflow as tf
from tf_agents.environments import tf_py_environment

from adapter.bomberman_adapter import BombermanEnvironment, BombermanGame


if __name__ == '__main__':
    eval_tf_env = tf_py_environment.TFPyEnvironment(BombermanEnvironment(mode='no_bomb', live_preview=True))

    time_step = eval_tf_env.reset()

    policy = tf.saved_model.load("policies/policy_ppo")
    policy_state = policy.get_initial_state(batch_size=eval_tf_env.batch_size)

    while not time_step.is_last():
        policy_step = policy.action(time_step, policy_state)
        policy_state = policy_step.state
        time_step = eval_tf_env.step(int(policy_step.action))