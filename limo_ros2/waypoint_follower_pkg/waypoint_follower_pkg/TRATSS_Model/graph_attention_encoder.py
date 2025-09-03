import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from typing import Tuple


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        return input + self.module(input, mask=mask)
#########################################################################################
#Gate from the gated transformer paper
#########################################################################################
class Gate(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int):
        """Gate Layer for both the Encoder & Decoder

        Parameters
        ----------
        input_dim: int
            input dimension of the input tensor
        hid_dim: int
            hidden dimension of the GRU layer
        """
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hid_dim)

    def forward(self, output, original_input):
        """Feed-forward function of the Gate Layer

        Parameters
        ----------
        output: [batch size, src len, hid dim]
            the output from either Attention Layer or Positionwise Layer

        original_input.shape: [batch size, src len, hid dim]
            the input preprocessed text tokens
        """
        b, f, s = original_input.shape

        # Permute the x and y so that the shape is now [B,S,F]
        original_input_permuted = original_input.permute(0, 2, 1)
        output_permuted = output.permute(0, 2, 1)

        # We really just need the GRU to weight between the input and the self attention. So we
        # resize to be [B * S, 1, F] so that we essentially have a massive batch of samples with
        # sequence length 1 and F features
        gate_output, hidden = self.gru(
            output,
            original_input.contiguous(),
        )

        return gate_output
#########################################################################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                     Attention mask should contain 1 if attention is not possible (additive attention)
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            compatibility[mask[None, :, :, :].expand_as(compatibility)] = -1e10

        attn = F.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch', learn_norm=True, track_norm=False):
        super(Normalization, self).__init__()

        self.normalizer = {
            "layer": nn.LayerNorm(embed_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(embed_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(normalization, None)

    def forward(self, input, mask=None):
        if self.normalizer:
            return self.normalizer(
                input.view(-1, input.size(-1))
            ).view(*input.size())
        else:
            return input


class PositionWiseFeedforward(nn.Module):

    def __init__(self, embed_dim, feed_forward_dim):
        super(PositionWiseFeedforward, self).__init__()
        self.sub_layers = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim, bias=True),
        )

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        return self.sub_layers(input)


class MultiHeadAttentionLayer(nn.Module):
    """Implements a configurable Transformer layer

    References:
        - W. Kool, H. van Hoof, and M. Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.
        - M. Deudon, P. Cournut, A. Lacoste, Y. Adulyasak, and L.-M. Rousseau. Learning heuristics for the tsp by policy gradient. In International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 170–181. Springer, 2018.
    """

    def __init__(self, n_heads, embed_dim, feed_forward_dim, 
                 norm='batch', learn_norm=True, track_norm=False):
        super(MultiHeadAttentionLayer, self).__init__()
        self.self_attention = SkipConnection(
            MultiHeadAttention(
                    n_heads=n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            )
        # self.first_gate = Gate(input_dim=embed_dim, hid_dim=embed_dim) #WAS ADDED BY ME
        self.norm1 = Normalization(embed_dim, norm, learn_norm, track_norm)

        self.positionwise_ff = SkipConnection(
            PositionWiseFeedforward(
                   embed_dim=embed_dim,
                   feed_forward_dim=feed_forward_dim
                )
            )
        # self.second_gate = Gate(input_dim=feed_forward_dim, hid_dim=embed_dim)#WAS ADD BY ME TOO
        self.norm2 = Normalization(embed_dim, norm, learn_norm, track_norm)

    def forward(self, h, mask):
        # x = h
        h = self.self_attention(h, mask=mask)
        # h = self.first_gate(h,x)
        h = self.norm1(h, mask=mask)
        # x2 = h
        h = self.positionwise_ff(h, mask=mask)
        # h = self.second_gate(h, x2)
        h = self.norm2(h, mask=mask)
        return h


class GraphAttentionEncoder(nn.Module):

    def __init__(self, n_layers, n_heads, hidden_dim, norm='batch', 
                 learn_norm=True, track_norm=False, *args, **kwargs):
        super(GraphAttentionEncoder, self).__init__()
        
        feed_forward_hidden = hidden_dim * 4
        
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(n_heads, hidden_dim, feed_forward_hidden, norm, learn_norm, track_norm)
                for _ in range(n_layers)
        ])

    def forward(self, x, graph):
        for layer in self.layers:
            x = layer(x, graph)
        return x









        