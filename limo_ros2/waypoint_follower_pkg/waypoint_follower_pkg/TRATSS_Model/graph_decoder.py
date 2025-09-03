import math
import numpy as np
from typing import NamedTuple
import time

import threading
import multiprocessing
import torch
from torch import linalg as LA
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import DataParallel
from waypoint_follower_pkg.TRATSS_Model.RL_Environment import LawnMowing
from copy import deepcopy

from icecream import ic

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
class AttentionModelFixed(NamedTuple):
    # """
    # Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    # This class allows for efficient indexing of multiple Tensors at once
    # """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return tuple.__getitem__(self,key)


class AttentionModel(nn.Module):

    def __init__(self,
                 problem,
                 embedding_dim,
                 encoder_class,
                 n_encode_layers,
                 aggregation="sum",
                 aggregation_graph="mean",
                 normalization="layer",
                 learn_norm=True,
                 track_norm=False,
                 gated=True,
                 n_heads=8,
                 tanh_clipping=10.0,
                 mask_inner=True,
                 mask_logits=True,
                 mask_graph=False,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 extra_logging=False,
                 *args, **kwargs):
        """
        Models with a GNN/Transformer/MLP encoder and the Autoregressive decoder using attention mechanism

        Args:
            problem: TSP
            embedding_dim: Hidden dimension for encoder/decoder
            encoder_class: GNN/Transformer/MLP encoder
            n_encode_layers: Number of layers for encoder
            aggregation: Aggregation function for GNN encoder
            aggregation_graph: Graph aggregation function
            normalization: Normalization scheme ('batch'/'layer'/'none')
            learn_norm: Flag for enabling learnt affine transformation during normalization
            track_norm: Flag to enable tracking training dataset stats instead of using batch stats during normalization
            gated: Flag to enbale anisotropic GNN aggregation
            n_heads: Number of attention heads for Transformer encoder/MHA in decoder
            tanh_clipping: Constant value to clip decoder logits with tanh
            mask_inner: Flag to use visited mask during inner function of decoder
            mask_logits: Flag to use visited mask during log computation of decoder
            mask_graph: Flag to use graph mask during decoding
            checkpoint_encoder: Whether to use checkpoints for encoder embeddings
            shrink_size: N/A
            extra_logging: Flag to perform extra logging, used for plotting histograms of embeddings

        References:
            - W. Kool, H. van Hoof, and M. Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.
            - M. Deudon, P. Cournut, A. Lacoste, Y. Adulyasak, and L.-M. Rousseau. Learning heuristics for the tsp by policy gradient. In International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 170–181. Springer, 2018.
        """

        super(AttentionModel, self).__init__()
        
        self.problem = problem
        self.embedding_dim = embedding_dim
        self.encoder_class = encoder_class
        self.n_encode_layers = n_encode_layers
        self.aggregation = aggregation
        self.aggregation_graph = aggregation_graph
        self.normalization = normalization
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.mask_graph = mask_graph
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        
        # Extra logging updates self variables with batch statistics (without returning them)
        self.extra_logging = extra_logging
        
        self.decode_type = None
        self.temp = 1.0
        # TSP
        assert problem.NAME in ("tsp", "tspsl"), "Unsupported problem: {}".format(problem.NAME)

        step_context_dim = 2 * embedding_dim  # Embedding of first and last node
        node_dim = 8  # x, y

        # Learned input symbols for first action
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
    
        # Input embedding layer
        self.init_embed = nn.Linear(node_dim, embedding_dim, bias=True)
        # self.val_init_embed = nn.Linear(node_dim_val, embedding_dim, bias=True)         
        
        # Encoder model
        self.embedder = self.encoder_class(n_layers=n_encode_layers, 
                                           n_heads=n_heads,
                                           hidden_dim=embedding_dim, 
                                           aggregation=aggregation, 
                                           norm=normalization, 
                                           learn_norm=learn_norm,
                                           track_norm=track_norm,
                                           gated=gated)

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # For each point in the area, we compute (glimpse key, glimpse value, logit key) so 3 * 2
        #I think this can be also used for the patterns.
        point_embed_dim = deepcopy(embedding_dim)
        self.init_point_embed = nn.Linear(2, point_embed_dim, bias=True)
        self.project_point_embeddings = nn.Linear(2, 3 * point_embed_dim, bias=False)

        self.point_project_out = nn.Linear(point_embed_dim, point_embed_dim, bias=False)

        self.init_pat_embed = nn.Linear(2, point_embed_dim, bias=True)
        self.project_pat_embeddings = nn.Linear(2, 3 * point_embed_dim, bias=False)

        self.pat_project_out = nn.Linear(point_embed_dim, point_embed_dim, bias=False)         

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
    
    def get_node_dim(self, nodes):
        self.node_dim = nodes.shape[1]

    def forward(self, nodes, pos, graph, current_area=None, current_location=None, visited_areas=None,
                new_areas=None, supervised=False, targets=None, class_weights=None, return_pi=False):
        """
        Args:
            nodes: Input graph nodes (B x N x 2)
            graph: Graph as adjacency matrices (B x N x N)
            supervised: Toggles SL training, teacher forcing and NLL loss computation
            targets: Targets for teacher forcing and NLL loss
            return_pi: Toggles returning the output sequences 
                       (Not compatible with DataParallel as the results
                        may be of different lengths on different GPUs)
        """
        device = nodes.device
        if self.checkpoint_encoder:
            embeddings = checkpoint(self.embedder, self._init_embed(nodes), graph)
        else:
            embeddings = self.embedder(self._init_embed(nodes), graph)
        
        if self.extra_logging:
            self.embeddings_batch = embeddings
        
        # Reinforcement learning or inference
         # Run inner function
        patterns = torch.tensor([1.,2.,3.], dtype=torch.float, device= device)

        _log_p, pi, cost = self._inner(patterns, nodes, pos, graph, embeddings,
                                       current_area, current_location, visited_areas, new_areas)

        if self.extra_logging:
            self.log_p_batch = self._log_p
            self.log_p_sel_batch = self._log_p.gather(2, self.pi.unsqueeze(-1)).squeeze(-1)
        
         # Log likelihood is calculated within the model since 
         # returning it per action does not work well with DataParallel 
         # (since sequences can be of different lengths)
        log_p = _log_p.squeeze(-1)
        assert (log_p > -1000).detach().all()
        ll = log_p.sum(1)
        cost2 = cost.squeeze(-1)
        if return_pi:
            return cost2, ll, pi
        return cost2, ll, pi

    def beam_search(self, *args, **kwargs):
        """Helper method to call beam search
        """
        return self.problem.beam_search(*args, **kwargs, model=self)


    def _calc_log_likelihood(self, _log_p, a, mask):
        
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(1, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)
#---------------------------------------------------------------------------------------------------------------------------------------
    def _init_embed(self, nodes):
        return self.init_embed(nodes)
    
#---------------------------------------------------------------------------------------------------------------------------------------
    def _inner(self, patterns, nodes, pos, graph, embeddings,
               current_area, current_location, visited_areas, new_areas, supervised=False, targets=None,):
        outputs = []
        sequences = []
        costs = []
        tasks = []
        device = nodes.device

        # Create problem state for masking (tracks which nodes have been visited)
        batch_size, num_nodes, _ = nodes.shape
        starting_points = torch.zeros((batch_size,2), dtype=torch.float, device=device)
        env = LawnMowing(areas=nodes, pos=pos, patterns=patterns, starting_point=starting_points)
        state, mask = env.reset()
        fixed = self._precompute(embeddings)
        
        # Perform decoding steps
	#---------------------------------------------------------------------------------------------------------------
        """
        The following was added to facilitate the online execution. If online execution is being used, the following
        section will be executeed only when online implementation is running
        """
        if current_location is not None:
            #Converting lists into tensors
            CA = torch.tensor(current_area, dtype=torch.int64, device=device).view(batch_size,-1)
            CL = torch.tensor(current_location, dtype=torch.float, device=device).view(batch_size,-1)
            visited = torch.tensor(visited_areas, dtype=torch.int64, device=device).view(batch_size,-1)
            new = torch.tensor(new_areas, dtype=torch.int64, device=device).view(batch_size,-1)
            flag = state[4]
            #Update the state
            state = [CA, CL, visited, new, flag]
            #Doing some other updates:
            # sequences.append(visited_areas)
            env.CA = CA
            env.CA1 = CL
            env.VA = visited
            env.NA = new
            if new.size(1) == num_nodes:
                i = 0
                env.nodes_mask = mask
            else:
                i = state[2].shape[1]
                A = torch.arange(num_nodes, dtype= torch.int64, device= device)
                given_areas = A.repeat(batch_size,1)
                sorted_VA, ind_VA = torch.sort(deepcopy(visited), dim=-1)
                mask_indices1 = sorted_VA.unsqueeze(2) == given_areas.unsqueeze(1)
                mask_indices = torch.any(mask_indices1, dim=1)
                mask[mask_indices] = True
        #---------------------------------------------------------------------------------------------------------------
        else:
            i=0
        score = torch.zeros((batch_size,1),dtype=torch.float, device=device)
        while not torch.all(state[4]):
            if self.shrink_size is not None:
                unfinished = state[3]
                if unfinished.size() == 0:
                    break
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]
            # Get log probabilities of next action
            class current_state:
                masks = mask[:,None,:].detach().clone()
                current_step = i
                ca = deepcopy(state[0])
                current_node = ca

                if (len(sequences) > 0) or (state[3].shape[1] != num_nodes):
                    num_steps = int(state[2].shape[1])
                    a1 = deepcopy(sequences[0][:,0]) if len(sequences) > 0 else state[2][:,0]
                    tour = [sequences[idx][:,0].reshape(-1,1) for idx in range(len(sequences))]
                    tour = torch.cat(tour, dim=-1) if tour else []
                    first_node = a1#.detach().clone()
                    first_node = first_node[:,None]
                else:
                    num_steps = i

            #This part gets probability of choosing area i [P(i)]:
            log_p_i = self._get_log_p(fixed, current_state, current_state.masks)
            mask2 = current_state.masks
        
            # Select the indices of the next nodes in the sequences
            selected_node, log_p_i2 = self._select_node(
                log_p_i.exp()[:,0,:], mask2[:,0,:])  # Squeeze out steps dimension
            #------------------------------------------------------------------------------------------------------------------------
            #We need now to get the probability of choosing point j [P(j|i)]:
            '''
            h_s: The point which the agent is currently at
            [batch_sizex1x2]
            H_j: The coordinates of the possible starting points of the selected area
            [batch_sizex4x2] where 4 is number of points at which the agent can start
            '''
            j_dim = 4
            b_idx = torch.arange(batch_size, dtype= torch.int64, device= device)
            h_s = state[1][:,None,:]
            H_j = deepcopy(nodes[b_idx,selected_node].reshape(batch_size,j_dim,2))
            log_p_j = self._get_log_p_j(H_j, h_s)
            selected_point, log_p_j2 = self._select_point(
                    log_p_j.exp()[:, 0, :])  # Squeeze out steps dimension

            #We need now to get the probability of choosing pattern k [P(k|i,j)]:
            '''
            h_s: The point which the agent will start the pattern at
            [batch_sizex1x2]
            H_j: The coordinates of the possible points at which the agent can stop
            [batch_sizex3x2] where 3 is number of points at which the agent can start
            '''
            k_dim = 3
            h_sp = deepcopy(H_j[b_idx, selected_point][:,None,:])
            patterns_base = torch.ones((batch_size), dtype= torch.int64, device= device)
            H_k = torch.cat([
                end_point(selected_point+1, nodes[b_idx,selected_node], pos[b_idx,selected_node], n*patterns_base)[:,None,:] for n in range(1,k_dim+1)
            ],1)
            log_p_k = self._get_log_p_k(H_k, h_sp)
            selected_pattern, log_p_k2 = self._select_point(
                    log_p_k.exp()[:, 0, :])  # Squeeze out steps dimension

            log_p = torch.log(log_p_i2*log_p_j2*log_p_k2)
            selected_a = torch.cat([selected_node[:,None], selected_point[:,None]+1, selected_pattern[:,None]+1], dim = 1).to(torch.int64)
            # Update problem state
            state, reward, done, mask = env.step(selected_a)
            # extra_distance = LA.norm(state[1]-exit_point_coord, dim=-1)[:,None]
            # env.CA1 = deepcopy(exit_point_coord)
            # starting_mask = mask_i[:,0].clone().detach()
            # mask = torch.cat([mask_i, starting_mask[:,None]], dim=1)
            score += reward #+extra_distance

            # Make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:,None])
            costs.append(deepcopy(score))
            sequences.append(deepcopy(selected_a))
            i += 1

        # Collected lists, return Tensor
        # torch.stack(costs[-1]),
        _log_p = torch.stack(outputs, dim=1)
        pi = torch.stack(sequences,1)
        cost = costs[-1]
        return _log_p, pi, cost
#---------------------------------------------------------------------------------------------------------------------------------------
    def _select_node(self, probs, mask):
        # probs = probs2[:,:-1]
        # mask = mask2[:,:-1]
        assert (probs == probs).all(), "Probs should not contain any NaNs"
        if self.decode_type == "greedy":
            probabilities, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).detach().any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            b = torch.arange(probs.size(0))
            probabilities = probs[b,selected]

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).detach().any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
                b = torch.arange(probs.size(0))
                probabilities = probs[b,selected]

        else:
            assert False, "Unknown decode type"
        
        return selected, probabilities
#---------------------------------------------------------------------------------------------------------------------------------------    
    def _select_point(self, probs):
        assert (probs == probs).all(), "Probs should not contain any NaNs"
        if self.decode_type == "greedy":
            probabilities, selected = probs.max(1)

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            b = torch.arange(probs.size(0))
            probabilities = probs[b,selected]
        
        return selected, probabilities
#---------------------------------------------------------------------------------------------------------------------------------------
    def _precompute(self, embeddings, num_steps=1):
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        if self.aggregation_graph == "sum":
            graph_embed = embeddings.sum(1)
        elif self.aggregation_graph == "max":
            graph_embed = embeddings.max(1)[0]
        elif self.aggregation_graph == "mean":
            graph_embed = embeddings.mean(1)
        else:  # Default: dissable graph embedding
            graph_embed = embeddings.sum(1) * 0.0
        
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
#---------------------------------------------------------------------------------------------------------------------------------------
    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, unlike torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(
                log_p.size(0), 1)[:, None, :]
        )
#---------------------------------------------------------------------------------------------------------------------------------------
    def _get_log_p(self, fixed, state, mask, normalize=True):
        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
        
        # Compute the mask, for masking next action based on previous actions
        # mask = state.get_mask()
        
        graph_mask = None
        if self.mask_graph:
            # Compute the graph mask, for masking next action based on graph structure 
            graph_mask = state.get_graph_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, graph_mask)

        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)
        
        assert not torch.isnan(log_p).any()
        return log_p
#---------------------------------------------------------------------------------------------------------------------------------------    
    def _get_log_p_j(self, Hj, Hs, normalize=True, num_steps=1):
        # Compute query = current point embedding
        query = self.init_point_embed(Hs)

        #Trying here to compute the possible points embeddings:
        glimpse_key, glimpse_val, logit_key = \
            self.project_point_embeddings(Hj[:, None, :, :]).chunk(3, dim=-1)
        
        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = (
            self._make_heads_j(glimpse_key, num_steps),
            self._make_heads_j(glimpse_val, num_steps),
            logit_key.contiguous()
        )
        
        graph_mask = None

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._j_one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, graph_mask)

        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p
#---------------------------------------------------------------------------------------------------------------------------------------    
    def _get_log_p_k(self, Hk, Hs, normalize=True, num_steps=1):
        # Compute query = current point embedding
        query = self.init_pat_embed(Hs)

        #Trying here to compute the possible points embeddings:
        glimpse_key, glimpse_val, logit_key = \
            self.project_pat_embeddings(Hk[:, None, :, :]).chunk(3, dim=-1)
        
        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = (
            self._make_heads_k(glimpse_key, num_steps),
            self._make_heads_k(glimpse_val, num_steps),
            logit_key.contiguous()
        )
        
        graph_mask = None

        # Compute logits (unnormalized log_p)
        log_p_k, glimpse = self._k_one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, graph_mask)

        if normalize:
            log_p_k = F.log_softmax(log_p_k / self.temp, dim=-1)

        assert not torch.isnan(log_p_k).any()

        return log_p_k
#---------------------------------------------------------------------------------------------------------------------------------------
    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once 
        (for efficient evaluation of the model)
        """
        batch_size = embeddings.size(0)
        num_steps = 1
        current_node = state.current_node

        if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
            if state.current_step == 0:
                # First and only step, ignore prev_a (this is a placeholder)
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                return embeddings.gather(
                    1,
                    torch.cat((state.first_node, current_node), 1)[:,:,None].expand(batch_size, 2, embeddings.size(-1))
                ).view(batch_size, 1, -1)
        
        # More than one step, assume always starting with first
        embeddings_per_step = embeddings.gather(
            1,
            state.tour[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
        )
        return torch.cat((
            # First step placeholder, cat in dim 1 (time steps)
            self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
            # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
            torch.cat((
                embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                embeddings_per_step
            ), 2)
        ), 1)
#---------------------------------------------------------------------------------------------------------------------------------------
    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, graph_mask=None):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads
        # Compute the glimpse, rearrange dimensions to (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        
        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        # compatibility = compatibility_i[:, :, :, :, 1:]
        
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -1e10
            if self.mask_graph:
                compatibility[graph_mask[None, :, :, None, :].expand_as(compatibility)] = -1e10

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        
        # From the logits compute the probabilities by masking the graph, clipping, and masking visited
        if self.mask_logits and self.mask_graph:
            logits[graph_mask] = -1e10 
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -1e10

        return logits, glimpse.squeeze(-2)
#---------------------------------------------------------------------------------------------------------------------------------------    
    def _j_one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, graph_mask=None):
        batch_size, _, embed_dim = query.size()
        key_size = val_size = embed_dim

        # Compute the glimpse, rearrange dimensions to (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, 1, 1, 1, key_size).permute(2, 0, 1, 3, 4)
        
        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.point_project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, 1, 1, 1 * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        

        return logits, glimpse.squeeze(-2)
#---------------------------------------------------------------------------------------------------------------------------------------    
    def _k_one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, graph_mask=None):
        batch_size, _, embed_dim = query.size()
        key_size = val_size = embed_dim

        # Compute the glimpse, rearrange dimensions to (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, 1, 1, 1, key_size).permute(2, 0, 1, 3, 4)
        
        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.pat_project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, 1, 1, val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        

        return logits, glimpse.squeeze(-2)
#---------------------------------------------------------------------------------------------------------------------------------------
    def _get_attention_node_data(self, fixed, state):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key
#---------------------------------------------------------------------------------------------------------------------------------------
    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
    
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
 #---------------------------------------------------------------------------------------------------------------------------------------   
    def _make_heads_j(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
    
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), 1, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), 1, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, corners_size, head_dim)
        )
 #---------------------------------------------------------------------------------------------------------------------------------------   
    def _make_heads_k(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
    
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), 1, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), 1, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, stopping_points_size, head_dim)
        )
