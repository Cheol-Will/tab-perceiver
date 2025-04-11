from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, GELU, Sequential, Parameter, Dropout
import torch.nn.functional as F

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.conv import FTTransformerConvs
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

def attend(q, k, v, dropout_prob=0.0):
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    attention = F.softmax(torch.einsum("bhqd,bhkd->bhqk", q, k) / (head_dim**(0.5)), dim=-1) 
    attention = torch.dropout(attention, dropout_prob, train=True)
    weighted_v = torch.einsum("bhqk,bhkd->bqhd", attention, v) 
    weighted_v = weighted_v.reshape(batch_size, seq_len, -1) 

    return weighted_v


class MLP(Module):
    """A dense module following attention in Transformer block."""
    
    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: int,
        dropout_prob: float = 0.0,
    ):
        super(MLP, self).__init__()

        self.fc1 = Linear(hidden_dim, hidden_dim*mlp_ratio)
        self.act = GELU()
        self.drop1 = Dropout(dropout_prob)
        self.norm = LayerNorm(hidden_dim*mlp_ratio)
        self.fc2 = Linear(hidden_dim*mlp_ratio, hidden_dim)
        self.drop2 = Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x        


class Attention(Module):
    """{Cross, Self}-Attention Module"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        input_qdim: int = None,
        input_kdim: int = None,
        dropout_prob: float = 0.0,
    ): 
        super(Attention, self).__init__()
        if input_qdim is None:
            input_qdim = hidden_dim
        if input_kdim is None:
            input_kdim = input_qdim

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads 
        self.dropout_prob = dropout_prob

        self._query = Linear(input_qdim, hidden_dim)
        self._key = Linear(input_kdim, hidden_dim)
        self._value = Linear(input_kdim, hidden_dim)  
        self.proj = Linear(hidden_dim, hidden_dim)
        self.qnorm = LayerNorm(self.head_dim)
        self.knorm = LayerNorm(self.head_dim)
        
    def forward(self, query, key=None, value=None):
        if key is None:
            # self-attention
            key = query
            value = query
        else:
            # corss-attention
            if value is None:
                value = key
        
        B, N, D = query.shape
        H = self.num_heads
        head_dim = self.head_dim

        # (B, N, D) -> (B, N, H, D') -> (B, H, N, D')
        Q = self._query(query).reshape(B, -1, H, head_dim).transpose(1,2)
        K = self._key(key).reshape(B, -1, H, head_dim).transpose(1,2)
        V = self._value(value).reshape(B, -1, H, head_dim).transpose(1,2)
        Q, K = self.qnorm(Q), self.knorm(K)

        out = attend(Q, K, V, self.dropout_prob)
        out = self.proj(out)
        return out


class SelfAttention(Module):
    """Self Attention Module including Norm, dropout, MLP"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: int,
        input_dim: int = None,
        dropout_prob: float = 0.0,
    ):
        super(SelfAttention, self).__init__()
        if input_dim is None:
            input_dim = hidden_dim

        self.attention = Attention(hidden_dim, num_heads, input_dim, input_dim, dropout_prob)
        self.norm1 = LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout_prob)    
        self.norm2 = LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(Module):
    """Cross Attention Module including Norm, dropout, MLP"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: int,
        input_qdim: int = None,
        input_kdim: int = None,
        dropout_prob: float = 0.0,
    ):
        super(CrossAttention, self).__init__()
        if input_qdim is None:
            input_qdim = hidden_dim
        if input_kdim is None:
            input_kdim = hidden_dim
        
        self.attention = Attention(hidden_dim, num_heads, input_qdim, input_kdim, dropout_prob)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout_prob)    

        self.q_norm = LayerNorm(input_qdim)
        self.kv_norm = LayerNorm(input_kdim)
        self.mlp_norm = LayerNorm(hidden_dim)

    def forward(self, query, key):
        x = self.q_norm(query)
        key = self.kv_norm(key)
        x = x + self.attention(x, key)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TabPerceiver(Module):
    """
    Args: 
        channels (int): Tensor frame embedding dimensionality.
        out_channels (int): Number of classes.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer blocks.
        num_latent_array (int): Number of latent array.
        latent_channels (int): Latent space dimensionality.
        dropout_prob (float): dropout probability
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        num_latent_array: int,
        latent_channels: int,
        dropout_prob: float,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
    ) -> None:
        super(TabPerceiver, self).__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }
        
        self.tensor_frame_encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        
        # Encoder and Decoder query with shape of (N, D) and (1, D)
        self.encoder_query = Parameter(torch.empty(num_latent_array, latent_channels))
        self.decoder_query = Parameter(torch.empty(1, latent_channels)) 

        # cross attention for latent encoder query
        self.encoder = CrossAttention(
            hidden_dim=latent_channels, 
            num_heads=num_heads, 
            input_kdim=channels,
            mlp_ratio=4,
            dropout_prob=dropout_prob,
            )
        self.blocks = Sequential(
            *[SelfAttention(
                hidden_dim=latent_channels, 
                num_heads=num_heads,
                mlp_ratio=4,
                dropout_prob=dropout_prob,
            )
            for _ in range(num_layers)]
        )
        self.decoder = CrossAttention(
            hidden_dim=latent_channels,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout_prob=dropout_prob,
        )
        self.proj = Sequential(
            LayerNorm(latent_channels),
            Linear(latent_channels, out_channels)
        )

    def reset_parameters(self) -> None:
        # tensor_frame embedding parameter reset
        self.tensor_frame_encoder.reset_parameters()

        # truncated normal with std=0.02(default) from PerceiverIO 
        nn.init.trunc_normal_(self.encoder_query, std=0.02)
        nn.init.trunc_normal_(self.decoder_query, std=0.02)

        # add weight initialization

    def forward(self, tf):
        # pre-processing with shape (batch_size, colummns, 1)
        # batch_size = tf.__len__()
        batch_size = len(tf)
        x, _ = self.tensor_frame_encoder(tf)

        # Encode input into latent of shape (batch_size, N, K) where N, K are hyperparamters of latent space.
        encoder_query = self.encoder_query.repeat(batch_size, 1, 1)
        x = self.encoder(encoder_query, x)
        
        # Multihead Attention
        x = self.blocks(x)

        # decode latent into decoder query shape (batch_size, num_classes, K)
        decoder_query = self.decoder_query.repeat(batch_size, 1, 1)
        x = self.decoder(decoder_query, x).reshape(batch_size, -1)

        # projection: (batch_size, hidden_dim) -> (batch_size, num_classes)
        x = self.proj(x)
        return x