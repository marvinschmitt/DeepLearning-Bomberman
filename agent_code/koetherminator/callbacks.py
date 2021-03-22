import tensorflow as tf
import tf_agents
import numpy as np

from adapter.bomberman_adapter import BombermanGame

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

# fixme: !important

def construct_timestep_from_gamestate(game_state: dict):
    step_type = tf_agents.trajectories.time_step.StepType.MID
    reward = 0.0
    discount = 1.0
    observation = BombermanGame.get_observation_from_state(game_state)

    time_step = tf_agents.trajectories.time_step.transition(
       tf.convert_to_tensor(observation, dtype=tf.float32),
       np.array(reward, dtype=np.float32)
    )

    # time_step = tf_agents.trajectories.time_step.TimeStep(
    #     tf_agents.trajectories.time_step.StepType.MID,
    #     tf.nest.map_structure(_as_float32_array, [reward]),
    #     _as_float32_array(discount),
    #     observation
    # )

    return time_step


def setup(self):
    self.policy = tf.compat.v2.saved_model.load('policy')
    self.policy_state = self.policy.get_initial_state(batch_size=1)


def act(self, game_state: dict):
    time_step = construct_timestep_from_gamestate(game_state)
    policy_step = self.policy.action(time_step, self.policy_state)
    self.policy_state = policy_step.state
    action = ACTIONS[policy_step.action]

    return action
