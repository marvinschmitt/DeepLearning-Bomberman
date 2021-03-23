import tensorflow as tf
import numpy as np
from tf_agents.trajectories import time_step
from adapter.bomberman_adapter import BombermanGame

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

# fixme: !important


def construct_timestep_from_gamestate(game_state: dict):
    observation = BombermanGame.get_observation_from_state(game_state)

    # timestep = time_step.TimeStep(
    #     step_type=time_step.StepType.MID,
    #     reward=np.array(reward, dtype=np.float32),
    #     discount=np.array(discount, dtype=np.float32),
    #     observation=tf.convert_to_tensor(observation, dtype=tf.float32)
    # )

    return time_step.transition(observation, 0.0)


def setup(self):
    self.policy = tf.compat.v2.saved_model.load('policy')
    self.policy_state = self.policy.get_initial_state(batch_size=1)


def act(self, game_state: dict):
    timestep = construct_timestep_from_gamestate(game_state)
    policy_step = self.policy.action(timestep, self.policy_state)
    self.policy_state = policy_step.state
    action = ACTIONS[policy_step.action]

    return action
