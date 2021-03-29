from functools import cached_property
from time import time

import settings as s
from fallbacks import pygame


class Item(object):
    def __init__(self):
        pass

    def get_state(self) -> tuple:
        raise NotImplementedError()


class Coin(Item):

    def __init__(self, pos, collectable=False):
        super(Coin, self).__init__()
        self.x = pos[0]
        self.y = pos[1]
        self.collectable = collectable

    def get_state(self):
        return self.x, self.y


class Bomb(Item):

    def __init__(self, pos, owner, timer, power):
        super(Bomb, self).__init__()
        self.x = pos[0]
        self.y = pos[1]
        self.owner = owner
        self.timer = timer
        self.power = power

        self.active = True

    def __eq__(self, bomb2) -> bool:
        if self.x != bomb2.x:
            return False

        if self.y != bomb2.y:
            return False

        if self.owner != bomb2.owner:
            return False

        if self.timer != bomb2.timer:
            return False

        if self.power != bomb2.power:
            return False

        if self.active != bomb2.active:
            return False

        return True

    def get_state(self):
        return (self.x, self.y), self.timer

    def get_blast_coords(self, arena):
        x, y = self.x, self.y
        blast_coords = [(x, y)]

        for i in range(1, self.power + 1):
            if arena[x + i, y] == -1:
                break
            blast_coords.append((x + i, y))
        for i in range(1, self.power + 1):
            if arena[x - i, y] == -1:
                break
            blast_coords.append((x - i, y))
        for i in range(1, self.power + 1):
            if arena[x, y + i] == -1:
                break
            blast_coords.append((x, y + i))
        for i in range(1, self.power + 1):
            if arena[x, y - i] == -1:
                break
            blast_coords.append((x, y - i))

        return blast_coords


class Explosion(Item):

    def __init__(self, blast_coords, owner, timer):
        super().__init__()
        self.blast_coords = blast_coords
        self.owner = owner
        self.timer = timer
        self.active = True

    def __eq__(self, exp2) -> bool:
        if self.blast_coords != exp2.blast_coords:
            return False

        if self.owner != exp2.owner:
            return False

        if self.timer != exp2.timer:
            return False

        if self.active != exp2.active:
            return False

        return True