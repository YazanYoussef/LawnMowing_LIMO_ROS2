import os
import time
from tqdm import tqdm
import math
import json
import argparse
import pprint as pp
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torch.nn import DataParallel


#--------------------------------------------------------------------------------------------------------
def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)
#---------------------------------- Rollout Functions ---------------------------------------------------
def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)

def get_node_dim(model, nodes):
        print(nodes.shape[1])
        model.node_dim = nodes.shape[1]

def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    def eval_model_bat(bat):
        with torch.no_grad():
            x = move_to(bat['nodes'], opts.device)
            graph = move_to(bat['graph'], opts.device)
            pos = move_to(bat['pos'], opts.device)
            cost, _, _  = model(x, pos ,graph)
        return cost.detach().cpu()
    
    val_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(val_loader, disable=opts.no_progress_bar, ascii=True)
    ], 0)


def rollout_groundtruth(problem, dataset, opts):
    return torch.cat([
        bat['tour_costs']
        for bat in DataLoader(
            dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    ], 0)
#------------------------------------------------------------------------------------------------


#--------------------------------- Validation Function ------------------------------------------
def validate(model, dataset, problem, opts):
    # Validate
    print(f'\nValidating on {len(dataset)} samples ...') 
    # from {dataset.filename}
    cost = rollout(model, dataset, opts)
    gt_cost = torch.sum(rollout_groundtruth(problem, dataset, opts), axis=-1)
    opt_gap = ((cost/gt_cost - 1) * 100)
    
    print('Validation groundtruth cost: {:.3f} +- {:.3f}'.format(
        gt_cost.mean(), torch.std(gt_cost)))
    print('Validation average cost: {:.3f} +- {:.3f}'.format(
        cost.mean(), torch.std(cost)))
    
    print('Validation optimality gap: {:.3f}% +- {:.3f}'.format(
        opt_gap.mean(), torch.std(opt_gap)))

    return cost.mean(), opt_gap.mean()
#------------------------------------------------------------------------------------------------
def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model
