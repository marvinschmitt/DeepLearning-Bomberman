from abc import ABC

import numpy as np

from collections import namedtuple

from tf_agents.environments import py_environment, utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step

from environment import BombeRLeWorld
import settings as s
import events as e
from typing import List


class BombermanGame:
    def __init__(self, make_video=False, replay=False):
        self._actions = ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
        self.ROWS, self.COLS = s.ROWS, s.COLS

        args = namedtuple("args",
                          ["no_gui", "fps", "turn_based", "update_interval", "save_replay", "replay", "make_video",
                           "continue_without_training"])
        args.continue_without_training = False
        args.save_replay = False

        if make_video:
            args.no_gui = False
            args.make_video = True
            args.fps = 15
            args.update_interval = 0.1
            args.turn_based = False

        else:
            args.no_gui = True
            args.make_video = False

        if replay:
            args.save_replay = True

        # agents = [("user_agent", True)] + [("rule_based_agent", False)] * (s.MAX_AGENTS-1)
        agents = [("user_agent", True)] + [("peaceful_agent", False)] * (s.MAX_AGENTS - 1)

        self._world = BombeRLeWorld(args, agents)
        self._agent = self._world.agents[0]

    def actions(self):
        """
        getter for available actions in the Bomberman Game
        Returns: private list containing the possible actions
        """
        return self._actions

    def make_action(self, agent_action : str):
        """

        Args:
            agent_action: action to be taken.

        Returns:
            reward: reward resulting from the action that has been taken.

        """
        self._world.do_step(agent_action)
        
        events = self._agent.events

        reward = BombermanGame.reward_from_events(events)

        return np.array(reward, dtype=np.float32)

    def get_world_state(self):
        return self._world.get_state_for_agent(self._agent)

    def get_observation(self):
        return BombermanGame.get_observation_from_state(self.get_world_state())

    @staticmethod
    def reward_from_events(events: List[str]) -> float:
        """
        *This is not a required function, but an idea to structure your code.*

        Here you can modify the rewards your agent get so as to en/discourage
        certain behavior.
        """
        game_rewards = {
            e.COIN_COLLECTED: 1,
            e.KILLED_OPPONENT: 5,
            # positive auxiliary rewards
            e.BOMB_DROPPED: 0.001,
            e.COIN_FOUND: 0.01,
            # e.SURVIVED_ROUND: 0.5,
            e.CRATE_DESTROYED: 0.1,
            e.MOVED_LEFT: 0.0001,
            e.MOVED_RIGHT: 0.0001,
            e.MOVED_UP: 0.0001,
            e.MOVED_DOWN: 0.0001,
            # negative auxiliary rewards
            e.INVALID_ACTION: -0.0002,
            e.WAITED: -0.0002,
            e.GOT_KILLED: -1,
            e.KILLED_SELF: -1
        }

        reward_sum = 0
        for event in events:
            if event in game_rewards:
                reward_sum += game_rewards[event]
        return reward_sum

    @staticmethod
    def get_observation_from_state(state):
        """
        Build a tensor of the observed board state for the agent.
        Layers:
        0: field with walls and crates
        1: revealed coins
        2: bombs
        3: agents (self and others)

        Returns: observation tensor

        """
        cols, rows = state['field'].shape[0], state['field'].shape[1]
        observation = np.zeros([rows, cols], dtype=np.float32)

        # write field with crates
        observation[:, :] = state['field']

        # write revealed coins
        if state['coins']:
            coins_x, coins_y = zip(*state['coins'])
            observation[list(coins_y), list(coins_x)] = 2  # revealed coins

        # write ticking bombs
        if state['bombs']:
            bombs_xy, bombs_t = zip(*state['bombs'])
            bombs_x, bombs_y = zip(*bombs_xy)
            observation[list(bombs_y), list(bombs_x)] = -2  # list(bombs_t)

        """
        bombs_xy = [xy for (xy, t) in state['bombs']]
        bombs_t = [t for (xy, t) in state['bombs']]
        bombs_x, bombs_y = [x for x, y in bombs_xy], [y for x, y in bombs_xy]
        observation[2, bombs_x, bombs_y] = bombs_t or 0
        """

        # write agents
        if state['self']:   # let's hope there is...
            _, _, _, (self_x, self_y) = state['self']
            observation[self_y, self_x] = 3

        if state['others']:
            _, _, _, others_xy = zip(*state['others'])
            others_x, others_y = zip(*others_xy)
            observation[others_y, others_x] = -3

        return observation
        # return tf.convert_to_tensor(observation, dtype=np.int32)

    def new_episode(self):
        # todo: End the world/game properly
        # if self._world.time_to_stop():
        #    self._world.end_round()

        self._world.new_round()

    def is_episode_finished(self):
        return self._world.time_to_stop()

    def set_user_input(self, new_user_input):
        self._world.user_input = new_user_input


class BombermanEnvironment(py_environment.PyEnvironment, ABC):  # todo: which methods of ABC are actually required?
    def __init__(self, make_video=False, replay=False):
        super().__init__()

        self._game = BombermanGame(make_video, replay)

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=5, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(self._game.ROWS, self._game.COLS),
                                                             dtype=np.float32, minimum=-3, maximum=3, name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._game.set_user_input(None)
        observation = self._game.get_observation()

        self._game.new_episode()

        return time_step.restart(observation)

    def _step(self, action):
        """
        Perform one step of the game
        Args:
            action: int

        Returns:
        """

        if self._game.is_episode_finished():
            self.reset()

        agent_action = self._game.actions()[action]

        reward = self._game.make_action(agent_action)

        observation = self._game.get_observation()

        if self._game.is_episode_finished():
            return time_step.termination(observation, reward)
        else:
            return time_step.transition(observation, reward)


if __name__ == "__main__":
    environment = BombermanEnvironment()
    utils.validate_py_environment(environment, episodes=10)
    print("Everything is fine.")
