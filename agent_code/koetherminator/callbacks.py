import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import numpy as np

from collections import namedtuple

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


class ReplayBuffer:
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    # state_ is next state
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, input_dims):
    inputs = tf.keras.Input(shape=input_dims)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(n_actions, activation=None)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model


class DQNAgent:
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=1e-3, epsilon_end=0.01,
                 mem_size=1000000, fname='dqn_model_koetherminator.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_net = build_dqn(lr, n_actions, input_dims)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_net(observation[np.newaxis, :])
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            # only learn if buffer is full enough
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_net(states)
        q_next = self.q_net(states_)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * dones

        self.q_net.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self):
        self.q_net.save(self.model_file)

    def load_model(self):
        self.q_net = load_model(self.model_file)


def setup(self):
    self.agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=1e-3,
                          input_dims=(17, 17, 1), epsilon_dec=1e-6,
                          n_actions=6, mem_size=100000, batch_size=64,
                          epsilon_end=0.01, fname='dqn_model_koetherminator.h5')
    self.agent.load_model()
    self.actions = BombermanGame.ACTIONS


def act(self, game_state: dict):
    observation = BombermanGame.get_observation_from_state(game_state)
    action = self.agent.choose_action(observation)

    return self.actions[action]
