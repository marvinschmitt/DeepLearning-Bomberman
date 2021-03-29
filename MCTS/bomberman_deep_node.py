from __future__ import annotations
import random
from MCTS.mcts import Node
from typing import Tuple, List, Set
from copy import deepcopy

import queue

import numpy as np
from collections import namedtuple

from environment_fast import BombeRLeWorld
from environment_fast import MOVES, BOMB, WAIT

from MCTS.bomberman_node import BombermanNode

WAIT_CHANCE = 1
BOMB_CHANCE = 1


class BombermanDeepNode(Node):
    """
    A representation of a single state of the Bomberman game.
    Needs to be able to direct all player actions.

    This is basically the actual board state + the agent who acts next + the action that was taken to get here.
    """

    class DummyTerminalNode:
        def __init__(self, reward):
            self.reward = reward

        def is_terminal(self):
            return True

        def get_reward(self):
            return self.reward

        def get_actor(self):
            return -1

    def __init__(self, world: BombeRLeWorld, predictor, actor_queue: queue.deque, action: str):
        self.world = world
        self.predictor = predictor

        if not actor_queue:
            self.world.do_step()
            active_agent_indices = [a for a in range(len(self.world.agents)) if self.world.agents[a] in self.world.active_agents]
            actor_queue = queue.deque(np.random.permutation(active_agent_indices))
        self.actor_queue = actor_queue

        self.actor = self.actor_queue.popleft() if self.actor_queue else None # (terminal state)
        self.action = action

        self.bomb_suppressed = False

    def get_actor(self) -> int:
        "Return who is taking the action. (index in 0,..,3)"
        return self.actor

    def get_action(self) -> str:
        "Return what action was taken to get to this state."
        return self.action

    def get_reward(self) -> np.array:
        "Return cumulated rewards up to this node."
        return 0

    def find_children(self) -> Set[Node]:
        "All possible successors of this board state."
        if self.is_terminal():
            return set()

        return set([self.make_move(self.actor, action) for action in self.get_valid_actions()])

    def find_child_for_rollout(self) -> Node:
        "Returns a dummy node that instantly ends rollout and uses predictor value instead."
        reward = np.zeros(4)
        for a in range(len(self.world.agents)):
            reward[a] = self.predictor.predict(self.world.get_observation(a))

        return self.DummyTerminalNode(reward)

    def is_terminal(self) -> bool:
        "Returns True if the node has no children."
        return not self.world.running

    def __hash__(self) -> int:
        "Nodes must be hashable"
        return id(self)

    def __eq__(self, node2) -> bool:
        "Nodes must be comparable"
        if self.world != node2.world:
            return False

        if self.actor_queue != node2.actor_queue:
            return False

        if self.actor != node2.actor:
            return False

        if self.action != node2.action:
            return False

        return True

    def make_move(self, agent_idx, action):
        world = deepcopy(self.world)
        a = world.agents[agent_idx]
        world.perform_agent_action(a, action)
        node = BombermanDeepNode(world, self.predictor, self.actor_queue, action)
        return node

    def suppress_bomb(self):
        self.bomb_suppressed = True

    def get_valid_actions(self):
        moves = [move for move in MOVES if self.is_action_valid(move)]
        bomb = [BOMB] * self.is_action_valid(BOMB) * np.random.choice([0, 1], p=[1-BOMB_CHANCE, BOMB_CHANCE]) \
            * (not self.bomb_suppressed)
        wait = [WAIT] * np.random.choice([0, 1], p=[1-WAIT_CHANCE, WAIT_CHANCE])

        return moves + bomb + wait or [WAIT]

    def is_action_valid(self, action):
        agent = self.world.agents[self.actor]
        return self.world.is_action_valid(agent, action)
