"""This module serves as dummy test to verify CI/CD"""
import unittest
from environment_fast import BombeRLeWorld
from agents_fast import Agent
from items_fast import Bomb

class TestBombermanNode(unittest.TestCase):
    def test_get_oberservation(self):
        a = Agent(train=True)
        agents = [a, Agent(), Agent(), Agent()]
        world = BombeRLeWorld(agents)
        world.bombs.append(Bomb((1, 2), a, 4, 3))
        print(world.get_observation()[:, :, 0])
        world.do_step()
        print(world.get_observation()[:, :, 0])
        world.do_step()
        print(world.get_observation()[:,:,0])