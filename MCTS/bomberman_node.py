from __future__ import annotations

from MCTS.mcts import Node
from typing import Tuple
from copy import deepcopy

import queue

import numpy as np
from collections import namedtuple

from environment import BombeRLeWorld

from MCTS.mcts_rewards import reward_from_events

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]


class BombermanNode(Node):
    """
    A representation of a single state of the Bomberman game.
    Needs to be able to direct all player actions.

    This is basically the actual board state + the agent who acts next + the action that was taken to get here.

    Implementation Hint: Maybe call 'perform_agent_action' from 'GenericWorld' directly?
    """
    def __init__(self, world: BombeRLeWorld, actor_queue: queue.deque):
        self.world = deepcopy(world)
        self.actor_queue = actor_queue or queue.deque(np.random.permutation(len(self.world.active_agents)))

        # make sure that no dead agent will be polled. todo: test
        for a in self.actor_queue:
            if a not in self.world.active_agents:
                self.actor_queue.remove(a)

        self.actor = self.actor_queue.pop()

    def get_actor(self) -> int:
        "Return who is taking the action. (index in 0,..,3)"
        return self.actor

    def get_action(self) -> str:
        "Return what action was taken. (index in 0,..,5)"
        pass

    def find_children(self) -> set[Node]:
        "All possible successors of this board state."
        pass

    def find_child_for_rollout(self) -> Tuple[Node, float]:
        "Successor of this board state (for now just a randomly selected action)+ reward"
        if self.is_terminal():
            return
        else:
            valid_actions = [action for action in ACTIONS if self.is_action_valid(action)]
            results = [self.make_move(self.actor, action) for action in valid_actions]
            nodes = [(BombermanNode(world, self.actor_queue), reward) for world, reward in results]
        return tuple(nodes)

    def is_terminal(self) -> bool:
        "Returns True if the node has no children."
        pass
        # incorporate time_to_stop or world.running (but on t+1!)

    def __hash__(self) -> int:
        "Nodes must be hashable"
        pass

    def __eq__(self, node2) -> bool:
        "Nodes must be comparable"
        pass

    def make_move(self, agent_idx, action):
        a = self.world.active_agents[agent_idx]
        self.world.perform_agent_action(a, action) # todo: problem – world has advanced here...
        events = a.events
        reward = reward_from_events(events)
        return self.world, reward

    def is_action_valid(self, action):
        pass


args = namedtuple("args",
                  ["no_gui", "fps", "log_dir", "turn_based", "update_interval", "save_replay", "replay",
                   "make_video",
                   "continue_without_training"])
args.continue_without_training = False
args.save_replay = False
args.log_dir = "log/"
args.no_gui = True
args.make_video = False

agents = [("user_agent", True)]*4

initial_world = BombeRLeWorld(args, agents)
initial_actor_queue = queue.deque(np.random.permutation(len(initial_world.active_agents)))

bomberman_node = BombermanNode(
    world=initial_world,
    actor_queue=initial_actor_queue
)
