from absl import app
from absl import flags
from absl import logging
import os
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment

from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.networks.q_network import QNetwork
from tf_agents.system.default.multiprocessing_core import handle_main
from tf_agents.system.multiprocessing_test import XValStateSaver

from adapter.bomberman_adapter import BombermanEnvironment

# fixme: !nice

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), 'Root directory for writing summaries and checkpoints')
FLAGS = flags.FLAGS

INITIAL_COLLECT_STEPS = 1000
EVAL_INTERVAL = 1000
NUM_EVAL_EPISODES = 100
BATCH_SIZE = 64


def train_eval_bomberman(
        root_dir,
        num_parallel_environments=4,
        summary_interval=1000
):
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')
    ckpt_dir = os.path.join(root_dir, 'checkpoint')
    policy_dir = os.path.join(root_dir, 'policy')

    train_summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=1000)
    train_summary_writer.set_as_default()
    eval_summary_writer = tf.summary.create_file_writer(eval_dir)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=10),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=10)
    ]

    global_step = tf.Variable(0)

    with tf.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
        tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
            [BombermanEnvironment] * num_parallel_environments
        ))
        eval_tf_env = BombermanEnvironment()
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


def main(_):
    if FLAGS.root_dir is None:
        raise AttributeError("require root_dir!")
    
    train_eval_bomberman(root_dir=FLAGS.root_dir)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    handle_main(main, extra_state_savers=[])
    # app.run(main)
