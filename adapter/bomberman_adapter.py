from abc import ABC

import numpy as np

from collections import namedtuple

from tf_agents.environments import py_environment, utils
from agents import Agent, SequentialAgentBackend

from environment import BombeRLeWorld
import settings as s
import events as e
from typing import List
from time import sleep

from fallbacks import pygame


class BombermanGame:
    ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

    def __init__(self, make_video=False, replay=False, live_preview=False):
        self._actions = self.ACTIONS
        self.ROWS, self.COLS = s.ROWS, s.COLS
        self._live_preview = False

        args = namedtuple("args",
                          ["no_gui", "fps", "log_dir", "turn_based", "update_interval", "save_replay", "replay",
                           "make_video",
                           "continue_without_training"])
        args.continue_without_training = False
        args.save_replay = False
        args.log_dir = "agent_code/koetherminator"

        if make_video:  # not working yet!
            args.no_gui = False
            args.make_video = True  # obviously gotta change to True if ffmpeg issue is fixed
            args.fps = 15
            args.update_interval = 0.1
            args.turn_based = False

        elif live_preview:
            self._live_preview = True
            args.no_gui = False
            args.make_video = False
            args.fps = 15
            args.update_interval = 1
            args.turn_based = False

        else:
            args.no_gui = True
            args.make_video = False

        if replay:
            args.save_replay = True

        # agents = [("user_agent", True)] + [("rule_based_agent", False)] * (s.MAX_AGENTS-1)
        agents = [("user_agent", True)] + [("peaceful_agent", False)] * (s.MAX_AGENTS - 1)

        if not args.no_gui:
            pygame.init()

        self._world = BombeRLeWorld(args, agents)
        self._agent = self._world.agents[0]

        rb_agent_cfg = {"color": "blue", "name": "rule_based_agent"}
        rb_agent_backend = SequentialAgentBackend(False, rb_agent_cfg['name'], rb_agent_cfg['name'])
        rb_agent_backend.start()
        self._rb_agent = Agent(rb_agent_cfg['color'], rb_agent_cfg['name'], rb_agent_cfg['name'], train=False,
                               backend=rb_agent_backend)

    def actions(self):
        """
        getter for available actions in the Bomberman Game
        Returns: private list containing the possible actions
        """
        return self._actions

    def make_action(self, agent_action: str):
        """

        Args:
            agent_action: action to be taken.

        Returns:
            reward: reward resulting from the action that has been taken.

        """
        self._world.do_step(agent_action)

        events = self._agent.events

        reward = self.reward(events)

        if self._live_preview:
            self._world.render()
            self._world.gui.render_text(f"ACTION: {agent_action}", 800, 490, (255, 255, 255))
            self._world.gui.render_text(f"REWARD: {reward}", 800, 520, (50, 255, 50) if reward > 0 else (255, 50, 50))
            pygame.display.flip()
            sleep(0.03)

        return np.array(reward, dtype=np.float32)

    def get_world_state(self):
        return self._world.get_state_for_agent(self._agent)

    def get_observation(self):
        return self.get_observation_from_state(self.get_world_state())

    @staticmethod
    def reward(events: List[str]) -> float:
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
            # e.COIN_FOUND: 0.01,
            # e.SURVIVED_ROUND: 0.5,
            e.CRATE_DESTROYED: 0.1,
            e.MOVED_LEFT: 0.001,
            e.MOVED_RIGHT: 0.001,
            e.MOVED_UP: 0.001,
            e.MOVED_DOWN: 0.001,
            # negative auxiliary rewards
            e.INVALID_ACTION: -0.002,
            e.WAITED: -0.002,
            e.GOT_KILLED: -1,
            e.KILLED_SELF: -1
        }

        reward_sum = 0
        for event in events:
            if event in game_rewards:
                reward_sum += game_rewards[event]
        return reward_sum

    @classmethod
    def get_observation_from_state(cls, state):
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
        observation = np.zeros([rows, cols, 1], dtype=np.float32)

        # write field with crates
        observation[:, :, 0] = state['field']

        # write revealed coins
        if state['coins']:
            coins_x, coins_y = zip(*state['coins'])
            observation[list(coins_y), list(coins_x), 0] = 2  # revealed coins

        # write ticking bombs
        if state['bombs']:
            bombs_xy, bombs_t = zip(*state['bombs'])
            bombs_x, bombs_y = zip(*bombs_xy)
            observation[list(bombs_y), list(bombs_x), 0] = -2  # list(bombs_t)

        """
        bombs_xy = [xy for (xy, t) in state['bombs']]
        bombs_t = [t for (xy, t) in state['bombs']]
        bombs_x, bombs_y = [x for x, y in bombs_xy], [y for x, y in bombs_xy]
        observation[2, bombs_x, bombs_y] = bombs_t or 0
        """

        # write agents
        if state['self']:  # let's hope there is...
            _, _, _, (self_x, self_y) = state['self']
            observation[self_y, self_x, 0] = 3

        if state['others']:
            _, _, _, others_xy = zip(*state['others'])
            others_x, others_y = zip(*others_xy)
            observation[others_y, others_x, 0] = -3

        return observation

    def new_episode(self):
        if self._world.running:
            self._world.end_round()

        if not self._world.running:
            self._world.ready_for_restart_flag.wait()
            self._world.ready_for_restart_flag.clear()
            self._world.new_round()

    def is_episode_finished(self):
        return self._world.time_to_stop()

    def set_user_input(self, new_user_input):
        self._world.user_input = new_user_input


class BombermanGameImitator(BombermanGame):
    def __init__(self, make_video=False, replay=False, live_preview=False):
        super().__init__(make_video, replay, live_preview=live_preview)

    def make_action(self, agent_action: str):
        game_state = self._world.get_state_for_agent(self._agent)
        self._rb_agent.act(game_state)
        rb_agent_action, _ = self._rb_agent.backend.get_with_time("act")

        self._world.do_step(agent_action)

        events = self._agent.events
        reward = self.reward(action=agent_action, target_action=rb_agent_action, events=events)

        return np.array(reward, dtype=np.float32)

    @classmethod
    def reward(cls, action, target_action, events=[]):
        """
        slight overkill at this point. may make more sophisticated
        """

        imitation_reward = 1.0
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
        game_rewards = {}

        reward_sum = float(action == target_action) * imitation_reward
        for event in events:
            if event in game_rewards:
                reward_sum += game_rewards[event]
        return reward_sum


class BombermanGameFourChannel(BombermanGame):
    def __init__(self, make_video=False, replay=False, live_preview=False):
        super().__init__(make_video, replay, live_preview=live_preview)

    @classmethod
    def get_observation_from_state(cls, state):
        """
        Build a tensor of the observed board state for the agent.
        Layers:
        0: field with walls and crates
        1: revealed coins
        2: bombs
        3: agents (self and others)

        Returns: observation tensor

        """
        rows, cols = state['field'].shape[0], state['field'].shape[1]
        observation = np.zeros([rows, cols, 4], dtype=np.float32)

        # write field with crates
        observation[:, :, 0] = state['field']

        # write revealed coins
        if state['coins']:
            coins_x, coins_y = zip(*state['coins'])
            observation[list(coins_y), list(coins_x), 1] = 1  # revealed coins

        # write ticking bombs
        if state['bombs']:
            bombs_xy, bombs_t = zip(*state['bombs'])
            bombs_x, bombs_y = zip(*bombs_xy)
            observation[list(bombs_y), list(bombs_x), 2] = list(bombs_t)

        """
        bombs_xy = [xy for (xy, t) in state['bombs']]
        bombs_t = [t for (xy, t) in state['bombs']]
        bombs_x, bombs_y = [x for x, y in bombs_xy], [y for x, y in bombs_xy]
        observation[2, bombs_x, bombs_y] = bombs_t or 0
        """

        # write agents (self: 1, others: -1)
        if state['self']:  # let's hope there is...
            _, _, _, (self_x, self_y) = state['self']
            observation[self_y, self_x, 3] = 1

        if state['others']:
            _, _, _, others_xy = zip(*state['others'])
            others_x, others_y = zip(*others_xy)
            observation[others_y, others_x, 3] = -1

        return observation


class BombermanEnvironment:
    def __init__(self, mode='base', make_video=False, replay=False, live_preview=False):
        super().__init__()
        self.observation_shape = (s.COLS, s.ROWS, 1)

        if mode == 'base':
            self._game = BombermanGame(make_video=make_video, replay=replay, live_preview=live_preview)
            self.actions = self._game.actions()

        elif mode == 'imitator':
            self._game = BombermanGameImitator(make_video=make_video, replay=replay, live_preview=live_preview)
            self.actions = self._game.actions()

        elif mode == 'fourchannel':
            self._game = BombermanGameFourChannel(make_video=make_video, replay=replay, live_preview=live_preview)
            self.actions = self._game.actions()
            self.observation_shape = (s.COLS, s.ROWS, 4)

        elif mode == 'no_bomb':
            self._game = BombermanGame(make_video=make_video, replay=replay, live_preview=live_preview)
            self.actions = self._game.actions()
            self.actions[-1] = 'WAIT'

        else:
            raise ValueError("Please specify a valid mode!")

    def reset(self):
        self._game.new_episode()
        self._game.set_user_input(None)
        observation = self._game.get_observation()

        return observation, 0.0

    def step(self, action):
        """
        Perform one step of the game
        Args:
            action: int

        Returns:
        """

        if self._game.is_episode_finished():
            self.reset()

        agent_action = self._game.actions()[int(action)]
        assert agent_action in self._game.actions(), "Action not in action space!"

        reward = self._game.make_action(agent_action)
        observation = self._game.get_observation()

        return observation, reward

    def is_finished(self):
        return self._game.is_episode_finished()


if __name__ == "__main__":
    environment = BombermanEnvironment()
    utils.validate_py_environment(environment, episodes=5)
    print("Everything is fine with base env")