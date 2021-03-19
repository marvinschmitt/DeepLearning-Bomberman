import tensorflow as tf


def construct_timestep_from_gamestate(game_state: dict):
    return None


def setup(self):
    self.policy = tf.compat.v2.saved_model.load('../../policy')
    self.policy_state = self.policy.get_initial_state(batch_size=1)




def act(self, game_state: dict):
    time_step = construct_timestep_from_gamestate(game_state)
    action = self.policy.action(time_step)

    return action