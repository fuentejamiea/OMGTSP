import gurobipy as grb
import unittest
import Heuristic.Christofides
import Graph.My_Graph as My_Graph
import Simplex.My_Model as My_Model
import numpy as np

def test_instance():
    graph = My_Graph.Graph("../../instances/ulysses22.txt")
    tour = Heuristic.Christofides.christofides(graph)
    grb_model = grb.Model()
    test_model = My_Model.MyModel(len(graph.nodes), len(graph.edges), obj=np.array(edge.weight for edge in graph.edges),
                          rhs=2 * np.ones((len(graph.nodes,))))
    test_model.x[[e.num for e in tour]] = 1
    for edge in graph.edges:
        test_model.A[edge.from_node.num, edge.num] = 1
        test_model.A[edge.to_node.num, edge.num] = 1
        grb_model.addVar(lb=0, obj=edge.weight, vtype=grb.GRB.CONTINUOUS)

    test_model.compute_bfs()
    grb_model.update()
    var_list = grb_model.getVars()
    for node in graph.nodes:
        grb_model.addConstr(lhs=grb.LinExpr([(1, var_list[edge.num]) for edge in node.edges]),
                             sense=grb.GRB.EQUAL, rhs=2)
    grb_model.update()
    grb_model.setObjective(grb_model.getObjective(), grb.GRB.MINIMIZE)
    #grb_model.optimize()




#test_instance()

z = np.array([1, 1, 1, 1, 1])
A = np.array([[3, 2, 1, 0, 0],
              [5, 1, 1, 1, 0],
              [2, 5, 1, 0, 1]])
B = np.linalg.inv(A[:, [2, 3, 4]])
#print(B)
z = z[[0,1]] - z[[2,3,4]]@B@(A[:, [0, 1]])
#print(z)
#print(B@A)
entering = 1
leaving = 0
e = np.zeros(3)
e[leaving] = 1
y = B.T@e
x = y@(A[:, [0, 1]])
ratio = z[entering]/x[entering]
#print(-ratio* y@A[:,2])
#print(z - (x * ratio))


test_model = My_Model.MyModel(3, 5, obj=np.ones(5),rhs=np.array([1, 3, 4]), A=A)
test_model.compute_bfs()

