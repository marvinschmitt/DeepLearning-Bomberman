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


class BombermanDeepNode(BombermanNode):
    """
    A representation of a single state of the Bomberman game.
    Needs to be able to direct all player actions.

    This is basically the actual board state + the agent who acts next + the action that was taken to get here.
    """
    def __init__(self, world: BombeRLeWorld, reward: np.array, actor_queue: queue.deque, action: str, rollout=False):
        super.__init__(world, reward, actor_queue, action, rollout)

    def get_reward(self) -> np.array:
        "Return what action was taken to get to this state."
        return self.reward

    def find_children(self) -> Set[Node]:
        "All possible successors of this board state."
        if self.is_terminal():
            return set()

        return set([self.make_move(self.actor, action) for action in self.get_valid_actions()])

    def find_child_for_rollout(self) -> Node:
        "Successor of this board state (for now just a randomly selected action)+ reward"
        action = random.choice(self.get_valid_actions())
        node = self.make_move(self.actor, action, rollout=True)
        return node

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

    def make_move(self, agent_idx, action, rollout=False):
        world = self.world if self.rollout else deepcopy(self.world)
        a = world.agents[agent_idx]
        world.perform_agent_action(a, action)
        node = BombermanNode(world, self.reward, self.actor_queue, action, rollout)
        return node
