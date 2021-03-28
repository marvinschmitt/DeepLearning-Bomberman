"""This module serves as dummy test to verify CI/CD"""
import unittest
import numpy as np

class TestGenerateData(unittest.TestCase):
    def test_load(self):
        obervation = np.load('observation.npy')
        value = np.load('value.npy')
        q = np.load('q.npy')