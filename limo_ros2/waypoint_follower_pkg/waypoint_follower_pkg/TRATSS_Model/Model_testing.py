import torch
from .VAL_Script import set_decode_type
from .utils import load_model
import os
# from SOLVER import solve_concorde
import numpy as np
from copy import deepcopy
from numpy import array
from numpy import linalg as LA
import random

class TRATSS_PLAN:
    def __init__(self, current_location, areas_corners, areas_centers,
                 current_area, visited_areas, new_areas, batch_size=1, path=None, transform=False):
        """
        Input:
            current_location: list of tuple with the (x,y) coordinates of the agent's current position
            areas_corners: list of the (x,y) coordinates of the corners of the areas
            areas_centers: list of the (x,y) coordinates of the centers of the areas
        """
        self.current_location = current_location
        self.current = current_area
        self.visited = visited_areas
        self.new = new_areas
        self.areas = areas_corners
        self.centers = areas_centers
        self.batch_size = batch_size
        self.plan_cost = None
        self.plan = None

        if path is None:
            self.path = ('/home/navinst/Desktop/LIMO_Experiment/Single_Agent_Code/TRATSS_Model/outputs/TA_11-20/run_Mon_2024_03_18_T_16_49_46')
        else:
            self.path = path
        self.TRATSS, _ = load_model(self.path)
        set_decode_type(self.TRATSS, "greedy")
        if transform:
            self.transformation(self.current_location, self.areas, self.centers)
        self.obtaine_solution()

    #---------------------------------------------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------------------------------------------
    def get_adjacency_matrix(self, num_areas, batch_size):
        Ai = np.ones((num_areas,num_areas))
        np.fill_diagonal(Ai,0)

        A = np.repeat(Ai[np.newaxis], batch_size,axis=0)

        return A.tolist()
    #---------------------------------------------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------------------------------------------
    def transformation(self, agent_location, areas, centers):
        num_points = int(len(areas[0])/2)
        # Extracting x,y information of the agent's location
        x_agent = agent_location[0]
        y_agent = agent_location[1]
        #  the x,y information of the corners of the areas
        for i in range(len(areas)):
            centers[i][0] -= x_agent
            centers[i][1] -= y_agent
            for p in range(num_points):
                areas[i][2*p] -= x_agent
                areas[i][2*p+1] -= y_agent
        
        self.areas = areas
        self.centers = centers
    #---------------------------------------------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------------------------------------------
    def obtaine_solution(self):
        batch_size = self.batch_size
        num_areas = len(self.areas)

        X = torch.FloatTensor(deepcopy(self.areas))[None,:,:]
        C = torch.FloatTensor(deepcopy(self.centers))[None,:,:]
        A1 = self.get_adjacency_matrix(num_areas,batch_size)[0]
        A = torch.tensor(A1, dtype=torch.int64)
        with torch.no_grad():
            self.plan_cost, _, self.plan = self.TRATSS(X,C,A,self.current, self.current_location, self.visited, self.new)
        
        save_path = '/home/navinst/Desktop/LIMO_Experiment/Fully_online_with_true_path_no_obstacles.pt'
        torch.save(self.plan, save_path)

    # print('My generated tour is: ', pi[:,:,0]+1)
    # print('My generated tour cost is: ', cost2)
    # print('MY TOUR:')
    # print(cost2.mean())
    # print(torch.std(cost2))
    # save_path = '/home/navinst/Desktop/LIMO_Experiment/solution.pt'
    # torch.save(pi, save_path)
    # print(len(pi[0]))
    # print(pi)
    # print(cost2)
