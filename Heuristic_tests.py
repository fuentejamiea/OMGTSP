import unittest
from Heuristic import *
from My_Graph import *


class HeuristicMethods(unittest.TestCase):
    def test_MST(self):
        graph = Graph("tests/MST_test.txt")
        mst, weight = min_spanning_tree(graph)
        self.assertEqual(len(mst), len(graph.nodes) - 1)
        self.assertEqual(weight, 37)

    def test_euler(self):
        graph = Graph("tests/Euler_test.txt")
        tour = eulerian_tour(graph)
        print(tour)

    def test_christofides(self):
        graph = Graph("instances/pr76.txt")
        christofides(graph)
