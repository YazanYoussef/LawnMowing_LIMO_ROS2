from concorde.problem import Problem
from concorde.concorde import Concorde

def solve_concorde(matrix):
    problem = Problem.from_matrix(matrix)
    solver = Concorde()
    solution = solver.solve(problem, concorde_exe='/home/yazanyoussef/Documents/concorde_build/TSP/concorde')
    return solution