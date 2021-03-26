import logging
import pickle
import random
from collections import namedtuple
from datetime import datetime
from logging.handlers import RotatingFileHandler
from os.path import dirname
from threading import Event
from time import time
from typing import List, Union

import numpy as np

import events as e
import settings as s
from agents_fast import Agent, SequentialAgentBackend
from fallbacks import pygame
from items_fast import Coin, Explosion, Bomb

WorldArgs = namedtuple("WorldArgs",
                       ["fps", "turn_based", "update_interval", "save_replay", "replay", "make_video", "continue_without_training"])


class GenericWorld:
    def __init__(self, args: WorldArgs):

        self.running: bool = False
        self.step: int

        self.active_agents: List[Agent]
        self.arena: np.ndarray
        self.coins: List[Coin]
        self.bombs: List[Bomb]
        self.explosions: List[Explosion]

        self.args = args

        self.colors = s.AGENT_COLORS

        self.round = 0
        self.running = False
        self.ready_for_restart_flag = Event()


    def new_round(self):
        raise NotImplementedError()

    def add_agent(self, agent: Agent):
        assert len(self.agents) < s.MAX_AGENTS

        self.agents.append(agent)

    def tile_is_free(self, x, y):
        is_free = (self.arena[x, y] == 0)
        if is_free:
            for obstacle in self.bombs + self.active_agents:
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free

    def perform_agent_action(self, agent: Agent, action: str):
        # Perform the specified action if possible, wait otherwise
        if action == 'UP' and self.tile_is_free(agent.x, agent.y - 1):
            agent.y -= 1
            agent.add_event(e.MOVED_UP)
        elif action == 'DOWN' and self.tile_is_free(agent.x, agent.y + 1):
            agent.y += 1
            agent.add_event(e.MOVED_DOWN)
        elif action == 'LEFT' and self.tile_is_free(agent.x - 1, agent.y):
            agent.x -= 1
            agent.add_event(e.MOVED_LEFT)
        elif action == 'RIGHT' and self.tile_is_free(agent.x + 1, agent.y):
            agent.x += 1
            agent.add_event(e.MOVED_RIGHT)
        elif action == 'BOMB' and agent.bombs_left:
            self.bombs.append(Bomb((agent.x, agent.y), agent, s.BOMB_TIMER, s.BOMB_POWER))
            agent.bombs_left = False
            agent.add_event(e.BOMB_DROPPED)
        elif action == 'WAIT':
            agent.add_event(e.WAITED)
        else:
            agent.add_event(e.INVALID_ACTION)

    def poll_and_run_agents(self):
        raise NotImplementedError()

    def do_step(self):
        self.step += 1

        self.poll_and_run_agents()

        self.collect_coins()
        self.update_bombs()
        self.evaluate_explosions()

        if self.time_to_stop():
            self.end_round()

    def collect_coins(self):
        for coin in self.coins:
            if coin.collectable:
                for a in self.active_agents:
                    if a.x == coin.x and a.y == coin.y:
                        coin.collectable = False
                        a.update_score(s.REWARD_COIN)
                        a.add_event(e.COIN_COLLECTED)

    def update_bombs(self):
        """
        Count down bombs placed
        Explode bombs at zero timer.

        :return:
        """
        for bomb in self.bombs:
            if bomb.timer <= 0:
                # Explode when timer is finished
                bomb.owner.add_event(e.BOMB_EXPLODED)
                blast_coords = bomb.get_blast_coords(self.arena)

                # Clear crates
                for (x, y) in blast_coords:
                    if self.arena[x, y] == 1:
                        self.arena[x, y] = 0
                        bomb.owner.add_event(e.CRATE_DESTROYED)
                        # Maybe reveal a coin
                        for c in self.coins:
                            if (c.x, c.y) == (x, y):
                                c.collectable = True
                                bomb.owner.add_event(e.COIN_FOUND)

                # Create explosion
                self.explosions.append(Explosion(blast_coords, bomb.owner, s.EXPLOSION_TIMER))
                bomb.active = False
                bomb.owner.bombs_left = True
            else:
                # Progress countdown
                bomb.timer -= 1
        self.bombs = [b for b in self.bombs if b.active]

    def evaluate_explosions(self):
        # Explosions
        agents_hit = set()
        for explosion in self.explosions:
            # Kill agents
            if explosion.timer > 1:
                for a in self.active_agents:
                    if (not a.dead) and (a.x, a.y) in explosion.blast_coords:
                        agents_hit.add(a)
                        # Note who killed whom, adjust scores
                        if a is explosion.owner:
                            a.add_event(e.KILLED_SELF)
                        else:
                            explosion.owner.update_score(s.REWARD_KILL)
                            explosion.owner.add_event(e.KILLED_OPPONENT)
            # Show smoke for a little longer
            if explosion.timer <= 0:
                explosion.active = False

            # Progress countdown
            explosion.timer -= 1
        for a in agents_hit:
            a.dead = True
            self.active_agents.remove(a)
            a.add_event(e.GOT_KILLED)
            for aa in self.active_agents:
                if aa is not a:
                    aa.add_event(e.OPPONENT_ELIMINATED)
        self.explosions = [exp for exp in self.explosions if exp.active]

    def time_to_stop(self):
        # Check round stopping criteria
        if len(self.active_agents) == 0:
            return True

        if (len(self.active_agents) == 1
                and (self.arena == 1).sum() == 0
                and all([not c.collectable for c in self.coins])
                and len(self.bombs) + len(self.explosions) == 0):
            return True

        if not any([a.train for a in self.active_agents]):
            return True

        if self.step >= s.MAX_STEPS:
            return True

        return False

class BombeRLeWorld(GenericWorld):
    def __init__(self, args: WorldArgs, agents: List[Agent]):
        super().__init__(args)

        for agent in agents:
            self.add_agent(agent)

    def new_round(self):
        if self.running:
            self.end_round()

        self.round += 1

        # Bookkeeping
        self.step = 0
        self.active_agents = []
        self.bombs = []
        self.explosions = []

        # Arena with wall and crate layout
        self.arena = (np.random.rand(s.COLS, s.ROWS) < s.CRATE_DENSITY).astype(int)
        self.arena[:1, :] = -1
        self.arena[-1:, :] = -1
        self.arena[:, :1] = -1
        self.arena[:, -1:] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    self.arena[x, y] = -1

        # Starting positions
        start_positions = [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]
        random.shuffle(start_positions)
        for (x, y) in start_positions:
            for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if self.arena[xx, yy] == 1:
                    self.arena[xx, yy] = 0

        # Distribute coins evenly
        self.coins = []

        x_split = np.array_split(np.arange(s.ROWS), 3)
        y_split = np.array_split(np.arange(s.COLS), 3)
        for x_slice in x_split:
            for y_slice in y_split:
                n_crates = (self.arena[x_slice[:, np.newaxis], y_slice] == 1).sum()
                while True:
                    x, y = np.random.choice(x_slice), np.random.choice(y_slice)
                    if n_crates == 0 and self.arena[x, y] == 0:
                        self.coins.append(Coin((x, y)))
                        self.coins[-1].collectable = True
                        break
                    elif self.arena[x, y] == 1:
                        self.coins.append(Coin((x, y)))
                        break

        # Reset agents and distribute starting positions
        for agent in self.agents:
            agent.start_round()
            self.active_agents.append(agent)
            agent.x, agent.y = start_positions.pop()

        self.running = True

    def get_state_for_agent(self):
        state = {
            'round': self.round,
            'step': self.step,
            'field': np.array(self.arena),
            'agents': [agent.get_state() for agent in self.active_agents],
            'bombs': [bomb.get_state() for bomb in self.bombs],
            'coins': [coin.get_state() for coin in self.coins if coin.collectable],
        }

        explosion_map = np.zeros(self.arena.shape)
        for exp in self.explosions:
            for (x, y) in exp.blast_coords:
                explosion_map[x, y] = max(explosion_map[x, y], exp.timer)
        state['explosion_map'] = explosion_map

        return state

    def send_training_events(self):
        # Send events to all agents that expect them, then reset and wait for them
        for a in self.agents:
            if a.train:
                if not a.dead:
                    a.process_game_events(self.get_state_for_agent(a))
                for enemy in self.active_agents:
                    if enemy is not a:
                        pass
                        # a.process_enemy_game_events(self.get_state_for_agent(enemy), enemy)
        for a in self.agents:
            if a.train:
                if not a.dead:
                    a.wait_for_game_event_processing()
                for enemy in self.active_agents:
                    if enemy is not a:
                        pass
                        # a.wait_for_enemy_game_event_processing()
        for a in self.active_agents:
            a.store_game_state(self.get_state_for_agent(a))
            a.reset_game_events()

    def poll_and_run_agents(self):
        self.send_training_events()

        # Tell agents to act
        for a in self.active_agents:
            if a.available_think_time > 0:
                a.act(self.get_state_for_agent(a))

        # Give agents time to decide
        perm = np.random.permutation(len(self.active_agents))
        self.replay['permutations'].append(perm)
        for i in perm:
            a = self.active_agents[i]
            if a.available_think_time > 0:
                action, think_time = a.wait_for_act()
                self.logger.info(f'Agent <{a.name}> chose action {action} in {think_time:.2f}s.')
                if think_time > a.available_think_time:
                    self.logger.warning(f'Agent <{a.name}> exceeded think time by {s.TIMEOUT - think_time}s. Setting action to "WAIT" and decreasing available time for next round.')
                    action = "WAIT"
                    a.available_think_time = s.TIMEOUT - (think_time - a.available_think_time)
                else:
                    self.logger.warning(f'Agent <{a.name}> stayed within acceptable think time.')
                    a.available_think_time = s.TIMEOUT
            else:
                self.logger.info(f'Skipping agent <{a.name}> because of last slow think time.')
                a.available_think_time += s.TIMEOUT
                action = "WAIT"

            self.replay['actions'][a.name].append(action)
            self.perform_agent_action(a, action)

    def end_round(self):
        assert self.running, "End of round requested while not running"
        super().end_round()

        self.logger.info(f'WRAPPING UP ROUND #{self.round}')
        # Clean up survivors
        for a in self.active_agents:
            a.add_event(e.SURVIVED_ROUND)

        # Send final event to agents that expect them
        for a in self.agents:
            if a.train:
                a.round_ended()

        # Save course of the game for future replay
        if self.args.save_replay:
            self.replay['n_steps'] = self.step
            with open(f'replays/{self.round_id}.pt', 'wb') as f:
                pickle.dump(self.replay, f)

        # Mark round as ended
        self.running = False

        self.logger.debug('Setting ready_for_restart_flag')
        self.ready_for_restart_flag.set()

    def end(self):
        if self.running:
            self.end_round()
        self.logger.info('SHUT DOWN')
        for a in self.agents:
            # Send exit message to shut down agent
            self.logger.debug(f'Sending exit message to agent <{a.name}>')
