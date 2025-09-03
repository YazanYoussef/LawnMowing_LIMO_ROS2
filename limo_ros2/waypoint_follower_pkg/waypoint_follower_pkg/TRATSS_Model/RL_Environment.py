from __future__ import print_function
from __future__ import annotations 

import sys
import random
from contextlib import closing
from io import StringIO
from typing import Optional
import numpy as np
import gym
from gym import Env, spaces
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation 
import itertools as tools
from IPython import display
from math import comb
from random import randint
from abc import abstractmethod
from copy import deepcopy
from typing import Generic, Optional, SupportsFloat, Tuple, TypeVar, Union
from gym import spaces
from gym.logger import deprecation
from gym.utils import seeding
from gym.utils.seeding import np_random
import math
import torch
from torch import nn
from torch import linalg as LA
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from icecream import ic 
import ctypes

from collections import Counter



#We will start here by defining some useful functions:

def get_key_of(value,dictionary):
    keys_list = list(dictionary)
    key = keys_list[value]
    return key
#---------------------------------------------------------------------------------------------------------------------------------------
def get_location(points, areas):
    '''
        points: (batch_size x 1)
        areas:  (batch_size x 8)
    '''
    b = torch.arange(points.size(0), device= points.device)
    num_of_points = int(areas.shape[1]/2)
    x_indices = torch.tensor([2*n for n in range(num_of_points)] , dtype=torch.int64, device=points.device).expand(points.size(0), -1)
    y_indices = torch.tensor([2*n+1 for n in range(num_of_points)], dtype=torch.int64, device=points.device).expand(points.size(0), -1)
    x_chosen = x_indices[b, points.squeeze()-1]
    y_chosen = y_indices[b, points.squeeze()-1]
    x_pos    = deepcopy(areas[b, x_chosen])
    y_pos    = deepcopy(areas[b, y_chosen])
    starting_points = torch.cat((x_pos[:,None], y_pos[:,None]), dim=1)
    return starting_points

#---------------------------------------------------------------------------------------------------------------------------------------
def end_point(points, areas, centers, pattern):
    '''
    point: number of the starting point (batch_size x 1)
    areas: (batch_size x 8)
    centers: (batch_size x 2)
    pattern: the selected patterns for the batch to follow (batch_size x 1)
    '''
    #Making the size of the following (batch_size x 1) instead of (batch_size)
    points = deepcopy(points[:,None])
    pattern = deepcopy(pattern[:,None])

    b = torch.arange(areas.size(0), device=points.device)[:,None]
    num_of_points = int(areas.shape[1]/2)
    ind = torch.arange(num_of_points, dtype=torch.float, device=points.device)
    ind = ind.repeat(areas.size(0),1)
    x_indices = torch.tensor([2*n for n in range(num_of_points)] , dtype=torch.int64, device=points.device).expand(points.size(0), -1)
    y_indices = torch.tensor([2*n+1 for n in range(num_of_points)], dtype=torch.int64, device=points.device).expand(points.size(0), -1)
    if num_of_points > 1:
        x_remaining = x_indices[ind != points-1].reshape(-1, x_indices.size(1)-1)
        y_remaining = y_indices[ind != points-1].reshape(-1, y_indices.size(1)-1)
        x_points    = deepcopy(areas[b,x_remaining])
        y_points    = deepcopy(areas[b,y_remaining])
    else:
        x_points    = deepcopy(areas[b,x_indices])
        y_points    = deepcopy(areas[b,y_indices])
    last_points = deepcopy(get_location(points, areas))

    #Defining the patterns here to know later which pattern is assigned to each area
    p1 = torch.tensor([1.], device= areas.device)
    p2 = torch.tensor([2.], device= areas.device)
    p3 = torch.tensor([3.], device= areas.device)
    
    #Getting the indices (as a boolean tensor) of the areas which got patterns 1, 2, and 3 respectively
    areas_with_p1 = pattern.unsqueeze(2) == p1.unsqueeze(1)
    areas_with_p1 = areas_with_p1.squeeze(-1)

    areas_with_p2 = pattern.unsqueeze(2) == p2.unsqueeze(1)
    areas_with_p2 = areas_with_p2.squeeze(-1)

    areas_with_p3 = pattern.unsqueeze(2) == p3.unsqueeze(1)
    areas_with_p3 = areas_with_p3.squeeze(-1)

    #Getting the ending point for the areas with pattern 3 "Spiral":
    last_points[areas_with_p3.squeeze()] = deepcopy(centers[areas_with_p3.squeeze()])

    #Getting the ending point for the areas with pattern 2 "ZicZac_V":
    '''
    Gets the point that is furthest from the starting point
    '''
    if torch.any(areas_with_p2.squeeze()):
        b_p2 = b[areas_with_p2.squeeze()].squeeze()
        num_starting_point_p2 = points[areas_with_p2].reshape(-1,1)
        starting_x_p2 = deepcopy(areas[b_p2.reshape(-1,1),x_indices[b_p2.reshape(-1,1),num_starting_point_p2-1]])
        starting_y_p2 = deepcopy(areas[b_p2.reshape(-1,1),y_indices[b_p2.reshape(-1,1),num_starting_point_p2-1]])

        delta_x_p2 = torch.abs(starting_x_p2-x_points[areas_with_p2.squeeze(-1)])
        delta_y_p2 = torch.abs(starting_y_p2-y_points[areas_with_p2.squeeze(-1)])
        v_p2 = torch.cat((delta_x_p2[:,:,None],delta_y_p2[:,:,None]), dim=-1)
        n_p2 = LA.norm(v_p2, dim=-1)
        idx_of_last_points_p2 = torch.argmax(n_p2, dim=1)
        last_points[areas_with_p2.squeeze()] = torch.cat([x_points[b_p2,idx_of_last_points_p2][:,None],y_points[b_p2,idx_of_last_points_p2][:,None]], dim=-1)

    #Getting the ending point for the areas with pattern 1 "ZicZac_H":
    '''
    Gets the point that has the minimum delta_x from the starting point
    '''
    if torch.any(areas_with_p1.squeeze()):
        b_p1 = b[areas_with_p1.squeeze()].squeeze()
        num_starting_point_p1 = points[areas_with_p1].reshape(-1,1)
        starting_x_p1 = deepcopy(areas[b_p1.reshape(-1,1),x_indices[b_p1.reshape(-1,1),num_starting_point_p1-1]])

        delta_x_p1 = torch.abs(starting_x_p1-x_points[areas_with_p1.squeeze(-1)])
        idx_of_last_points_p1 = torch.argmin(delta_x_p1, dim=1)
        last_points[areas_with_p1.squeeze()] = torch.cat([x_points[b_p1,idx_of_last_points_p1][:,None],y_points[b_p1,idx_of_last_points_p1][:,None]], dim=-1)

    return last_points
#There was a function here which was used to get the center of the corners. Now, we already have 
#the center of each area.

#-------------------------------------------------------------------------------------------------------------------------------------------

class LawnMowing(gym.Env):
    """
    The Lawn Mowing Problem that we have:
    
    ### Description
    We will assume here that we have a robot designated to perform lawn mowing task to a certain area.
    
    
    ### Actions
    Given "a = number of areas" and "p = number of possible patterns", the action space will be of the following form:
    [area_i, point_k, pattern_j].


    The actions will be: [i,k,j] with i = 1:a, k = 1:4, and j = 1:p
    
    ### Observations
    The number of possible states will be: ((a+1)*sum([aCr: r=0:a])), where "a: number of areas" and the sum will be
    the possibile combinations of the areas. All these states are reachable. The episode ends when "Done = 1". 
    
    Current area could be:
    0:Not a given area
    1:First area
    .
    .
    .
    a:last area 
    ### Rewards
    The rewards are set as negative the distance such that the agent minimizes the covered distance.
    
    ### Rendering
    
    
    ### Arguments
    
    
    """

    metadata = {"render_modes": ["auto"], "render_fps": 4}
    def __init__(self, areas, pos, patterns, starting_point):
        '''
        Parameters
        ----------
        areas = batch_size x N x 8
        pos   = batch_size x N x 2
        patterns = 1 x 3
        starting_point = batch_size x 2
        '''

        self.Map = deepcopy(areas)
        self.Points = deepcopy(pos)
        self.patterns = deepcopy(patterns)

        p_names = ['ZicZac_H','ZicZac_V','Spiral']
        #Creating a dictionary for the input patterns:
        self.patterns_dict = {}
        for i in range(patterns.shape[0]):
            self.patterns_dict[patterns[i]] = p_names[i]
       
        self.starting_loc = deepcopy(starting_point)
        
        #Number of check points within an area:
        check_points = int(self.Map.shape[2]/2)
        #The number of areas that we have:
        self.num_areas = self.Map.shape[1]

        #Current area (it is the starting point and we check whether it is in any area or not):
        self.batch_size = areas.size()[0]
        self.device = areas.device
        self.CA = torch.zeros((self.batch_size, 1), dtype=torch.int64, device=areas.device)
        self.CA1 = deepcopy(starting_point)
        #Visited areas initialized as an empty set (fresh start):
        self.VA = torch.zeros((self.batch_size, 1), dtype=torch.int64, device=areas.device)
        #New areas which are all the given areas (fresh start): 
        NA1 = torch.arange(self.num_areas, dtype=torch.int64, device= self.device)
        self.NA = NA1.repeat(self.batch_size,1)
        #"done" flag initialized to zero:
        self.done = torch.zeros((self.batch_size, 1), dtype=torch.bool).to(self.device)
        #The initial state:
        self.initial_state = deepcopy([self.CA, self.CA1, self.VA, self.NA, self.done])
        

        self.negative_reward = 5


    def step(self, a):
        '''
        Parameters
        ----------
        a : [batch_size x 3]
        '''
        
        batch_size = a.size()[0]
        i = a[:, 0]
        j = a[:, 1]
        k = a[:, 2]

        batch_dim = torch.arange(batch_size, device=self.device)
        areas = self.Map[batch_dim, i]
        pos = self.Points[batch_dim, i]
        area_num = i
        points = j
        current = area_num[:,None]

        # Initializing the reward tensor
        reward = torch.zeros((batch_size, 1), dtype=torch.float, device=self.device)

        self.CA = deepcopy(current)

        starting_points = get_location(points, areas)

        #Getting the distance between the chosen starting points and the current points
        distances = LA.vector_norm(self.CA1-starting_points, dim=-1)[:,None]

        stopping_points = end_point(points, areas, pos, k)
        
        self.CA1 = deepcopy(stopping_points)
        
        reward = deepcopy(distances)

        #Updating the visited areas
        if torch.equal(self.VA, torch.zeros((self.batch_size, 1), dtype=torch.int64, device=areas.device)) and (self.num_areas-self.NA.size(1) == 0):
            self.VA = deepcopy(self.CA)
        else:
            self.VA = torch.cat((deepcopy(self.VA),deepcopy(self.CA)), dim=1)
        
        
        
        #The following updates the new areas
        sorted_VA, ind_VA = torch.sort(deepcopy(self.VA), dim=-1)
        sorted_NA, ind_NA = torch.sort(deepcopy(self.NA), dim=-1)
        idx = sorted_VA.unsqueeze(2) != sorted_NA.unsqueeze(1)
        idx_bool = torch.all(idx, dim=1)
        ind_of_NA_to_keep = ind_NA[idx_bool].reshape(batch_size,(self.num_areas-self.VA.size(1)))

        self.NA = self.NA[batch_dim.reshape(-1,1), ind_of_NA_to_keep]

        #Updating the done flag
        A = torch.arange(self.num_areas, dtype= torch.int64, device= self.device)
        given_areas = A.repeat(batch_size,1)
        done_idx1 = sorted_VA.unsqueeze(2) == given_areas.unsqueeze(1)
        done_idx = torch.all(torch.any(done_idx1, dim=1),dim=1)
        self.done[done_idx] = True
        
        #Adding the return reward to the done areas
        starting_locs = deepcopy(self.starting_loc[done_idx])
        return_distances = LA.vector_norm(self.CA1[done_idx]-starting_locs, dim=-1)[:,None]
        reward[done_idx] += deepcopy(return_distances)

        # Updating the state
        new_state = [self.CA,self.CA1,self.VA,self.NA,self.done]
        
        #Naming new variables to be returned by the step function
        s = deepcopy(new_state)
        r = deepcopy(reward)

        #We need here to update the masking
        mask_indices1 = sorted_VA.unsqueeze(2) == given_areas.unsqueeze(1)
        mask_indices = torch.any(mask_indices1, dim=1)
        self.nodes_mask[mask_indices] = True
        return (s, r, deepcopy(self.done), deepcopy(self.nodes_mask))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        '''
        The action masking is eliminated now as it is not used so far in the code
        '''
        self.nodes_mask = torch.zeros((self.batch_size,self.num_areas), dtype=bool, device= self.device)
        # self.nodes_mask[:,0] = True
        # self.distance = 0
        s = deepcopy(self.initial_state)
        self.lastaction = None
        if not return_info:
            return (s,deepcopy(self.nodes_mask)) 
        else:
            return (s, deepcopy(self.nodes_mask))

    def render(self, mode="auto"):
        x_data = self.path[:,0]
        y_data = self.path[:,1]

        fig = plt.figure()

        lines = plt.plot([])
        line = lines[0]
        
        coord = [self.Map[i][0].T for i in range(len(self.Map))]
        
        def scatter_plot(list):
            x = []
            y = []
            colors = []
            labels = []
            counter = 0
            for i in list:
                x = i[0]
                y = i[1]
                rand_color = '#%06X' % randint(0, 0xFFFFFF)
                plt.scatter(x,y,label = 'area_'+chr(counter+65),color = rand_color)
                plt.legend()
                plt.show
                counter += 1

        def animation_plot(i):
            scatter_plot(coord)
            line.set_data((x_data[0:i],y_data[0:i]))

        anim = FuncAnimation(fig, animation_plot, frames=len(self.path)+1, interval=200)
        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()
