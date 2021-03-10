import tensorflow as tf

from tf_agents.environments import tf_py_environment

from dqn_learning.BombermanEnvironment import BombermanEnvironment


if __name__ == '__main__':
    eval_tf_env = tf_py_environment.TFPyEnvironment(BombermanEnvironment(replay=True))

    time_step = eval_tf_env.reset()

    policy = tf.compat.v2.saved_model.load("policy")
    policy_state = policy.get_initial_state(batch_size=eval_tf_env.batch_size)

    while not time_step.is_last():
        policy_step = policy.action(time_step, policy_state)
        policy_state = policy_step.state
        time_step = eval_tf_env.step(policy_step.action)