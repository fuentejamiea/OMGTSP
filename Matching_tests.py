import unittest
from Matching import *
import numpy as np
import pickle
import gurobipy


def clean_matching(mate):
    """
    :param mate:
        Matching in M[node] = e(node, node_mate)  format
    :return:
        Matching in {(node1,node2), (node3,node4)... ] format with tuples sorted in increasing order
    """
    matching = set()
    for e in set(mate.values()):
        if e.from_node.num < e.to_node.num:
            matching.add((e.to_node.num, e.from_node.num))
        else:
            matching.add((e.from_node.num, e.to_node.num))
    return matching


def random_cardinality_matching(n, num, denom, pickle_path=""):
    """
    :param n:
        number of Nodes in test Graph
    :param num:
        numerator in num/denom = probability there exists an edge between any two distinct nodes
    :param denom:
        denominator in num/denom = probability there exists an edge between any two distinct nodes
    :param pickle_path:
        (optional) path to pickled random mat representing node adj. matrix
    :return:
        diff = difference between maximal matching algorithm and cardinality of matching found by gurobi ILP
        invalid = number edges in matching not in graph
    """
    if pickle_path:
        with open(pickle_path, 'rb') as f:
            mat = pickle.load(f)
            n, num, denom = map(int, pickle_path.split("_")[:3])
    else:
        mat = []

    cutoff = denom - num
    graph = Matching()
    model = gurobipy.Model()
    model.setParam("OutputFlag", False)
    graph.add_nodes(n)
    con_map = {i: set() for i in range(n)}

    for i in range(n):
        if not pickle_path:
            new_row = np.random.randint(0, denom, i)
            mat.append(new_row)
        for j in range(i):
            if mat[i][j] >= cutoff:
                n1 = graph.get_node(i)
                n2 = graph.get_node(j)
                graph.add_edge(n1, n2, 0)
                new_var = model.addVar(obj=-1, vtype=gurobipy.GRB.BINARY, name="({},{})".format(i, j))
                model.update()
                con_map[i].add(new_var)
                con_map[j].add(new_var)

    for nd, const in con_map.items():
        model.addConstr(gurobipy.quicksum(const) <= 1)
    model.update()

    mate, _, _,  = graph.maximal_matching()
    matching = clean_matching(mate)
    count1 = len(matching)
    matching = {m for m in matching if mat[m[0]][m[1]] >= cutoff}
    count2 = len(matching)
    invalid = count1 - count2
    model.optimize()
    grb_match = [v.VarName for v in model.getVars() if v.x != 0]
    diff = len(matching) - len(grb_match)
    if diff != 0 or invalid != 0:
        print("{}_{}_{}cardinality matching failure".format(n, num, denom))
        if not pickle_path:
            with open('{}_{}_{}_problem_mat.pkl'.format(n, num, denom), 'wb') as f:
                pickle.dump(mat, f)
    return diff, invalid


def random_weighted_matching(n, rand_range, pickle_path=""):
    """
    :param n:
        number of Nodes in test Graph
    :param rand_range:
        edge weight for each node pair is in [1, rand_range)
    :param pickle_path:
        (optional) path to pickled random mat representing node adj. matrix
    :return:
        boolean indicating if matching weight is equal to weight calculated with gurobi ILP
    """
    if pickle_path:
        with open(pickle_path, 'rb') as f:
            mat = pickle.load(f)
            n, rand_range = map(int, pickle_path.split("_")[:2])
    else:
        mat = []
    graph = Matching()
    model = gurobipy.Model()
    model.setParam("OutputFlag", False)
    graph.add_nodes(n)
    con_map = {i: set() for i in range(n)}

    for i in range(n):
        if not pickle_path:
            new_row = np.random.randint(1, rand_range, i)
            mat.append(new_row)
        for j in range(i):
            n1 = graph.get_node(i)
            n2 = graph.get_node(j)
            graph.add_edge(n1, n2, mat[i][j])
            new_var = model.addVar(obj=mat[i][j], vtype=gurobipy.GRB.BINARY, name="({},{})".format(i, j))
            model.update()
            con_map[i].add(new_var)
            con_map[j].add(new_var)

    for nd, const in con_map.items():
        model.addConstr(gurobipy.quicksum(const) == 1)
    model.update()
    model.optimize()
    opt_val = model.getObjective().getValue()
    mate, my_val = graph.weighted_matching()
    opt_flag = opt_val == my_val

    if not opt_flag:
        print("{}_{}_weighted matching failure".format(n, rand_range))
        if not pickle_path:
            with open('{}_{}_problem_mat.pkl'.format(n, rand_range), 'wb') as f:
                pickle.dump(mat, f)

    return opt_flag


class MatchingMethods(unittest.TestCase):
    def test_aps(self):
        graph = Matching("Tests/matching_test1.txt")
        mate, _, _ = graph.maximal_matching()
        matching = set(mate.values())
        self.assertEqual(len(matching), 6)

    def test_blossom(self):
        graph = Matching("Tests/blossom_test0.txt")
        mate = {}
        for edge in graph.edges:
            if edge.weight:
                mate[edge.to_node] = edge
                mate[edge.from_node] = edge
        mate, _, _ = graph.maximal_matching(mate=mate)
        self.assertEqual(clean_matching(mate), {(1, 0), (4, 2), (5, 3), (7, 6), (9, 8),
                                                (11, 10), (13, 12), (15, 14)})

    def test_weighted_matching(self):
        graph = Matching("Tests/weighted_matching.txt")
        _, my_val = graph.weighted_matching()
        self.assertEqual(my_val, 44)

    def test_random_maximal(self):
        for n in [10, 15, 20, 50, 75, 100]:
            for num in [1, 3, 7]:
                self.assertEqual((0, 0), random_cardinality_matching(n, num, 10))

    def test_random_weights(self):
        for n in [10, 20, 50, 80, 100]:
            for r in [20, 50, 100]:
                self.assertTrue(random_weighted_matching(n, r))


if __name__ == '__main__':
    unittest.main()
