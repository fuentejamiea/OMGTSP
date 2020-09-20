import unittest
from collections import Counter
from Heuristic.Christofides import *
from Graph.My_Graph import *


class HeuristicMethods(unittest.TestCase):
    def test_MST(self):
        graph = Graph("instances/MST_test.txt")
        mst, weight = min_spanning_tree(graph)
        self.assertEqual(len(mst), len(graph.nodes) - 1)
        self.assertEqual(weight, 37)

    def test_euler(self):
        graph = Graph("instances/Euler_test.txt")
        tour = eulerian_tour(graph)
        self.assertEqual(len(set(tour)), len(graph.edges))

    def test_christofides(self):
        graph = Graph("../../instances/ulysses22.txt")
        tour = christofides(graph)
        cnt = [0]*len(graph.nodes)
        self.assertEqual(len(graph.nodes), len(tour))
        for e in tour:
            cnt[e.from_node.num] += 1
            cnt[e.to_node.num] += 1
        for num in cnt:
            self.assertEqual(2, num)



