from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn import LayerNorm, Linear, Module, GELU, Sequential, Parameter, Dropout
import torch.nn.functional as F

import torch_frame
from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

def attend(query, key, value, dropout_prob=0.0, train=True):
    batch_size, num_heads, query_len, head_dim = query.shape
    
    attention = F.softmax(torch.einsum("bhqd,bhkd->bhqk", query, key) / (head_dim**(0.5)), dim=-1) 
    attention = F.dropout(attention, p=dropout_prob, training=train)
    weighted_sum = torch.einsum("bhqk,bhkd->bhqd", attention, value) # (batch_size, num_heads, query_len, head_dim)
    return weighted_sum


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

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

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

    def reset_parameters(self):
        self._query.reset_parameters()
        self._key.reset_parameters()
        self._value.reset_parameters()
        self.proj.reset_parameters()
        
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

        # (batch_size, query_len, hidden_dim) -> (batch_size, num_heads, query_len, head_dim)
        Q = self._query(query).reshape(B, -1, H, head_dim).transpose(1,2)
        K = self._key(key).reshape(B, -1, H, head_dim).transpose(1,2)
        V = self._value(value).reshape(B, -1, H, head_dim).transpose(1,2)

        # (batch_size, num_heads, query_len, head_dim) -> (batch_size, query_len, hidden_dim)
        out = attend(Q, K, V, self.dropout_prob, self.training) 
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
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

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.mlp.reset_parameters()

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

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, query, key):
        x = query + self.attention(self.q_norm(query), self.kv_norm(key))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TabPerceiver(Module):
    r"""

    Args:
        out_channels (int): Output channels dimensionality
        num_heads (int): Number of heads in the self-attention layer.
        num_layers (int): Number of self-attention layers
        num_latents (int): Number of latents
        hidden_dim (int): Embedding Dimensionality
    """
    def __init__(
        self,
        out_channels: int,
        num_features: int,
        num_heads: int,
        num_layers: int,
        num_latents: int,
        hidden_dim: int,
        dropout_prob: float,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
    ) -> None:
        super(TabPerceiver, self).__init__()
        self.hidden_dim = hidden_dim
        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
        }
        
        # (num_features, 1) -> (num_features, hidden_dim) 
        self.tensor_frame_encoder = StypeWiseFeatureEncoder(
            out_channels=hidden_dim,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        # Positional embedding 
        self.pos_embedding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        
        # Latents and Decoder query
        self.latents = Parameter(torch.empty(1, num_latents, hidden_dim))
        self.queries = Parameter(torch.empty(1, 1, hidden_dim)) 
        self.encoder = CrossAttention(
            hidden_dim=hidden_dim, 
            num_heads=num_heads, 
            input_kdim=hidden_dim,
            mlp_ratio=4,
            dropout_prob=dropout_prob,
        )
        self.blocks = Sequential(
            *[SelfAttention(
                hidden_dim=hidden_dim, 
                num_heads=num_heads,
                mlp_ratio=4,
                dropout_prob=dropout_prob,
            )
            for _ in range(num_layers)]
        )
        self.decoder = CrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout_prob=dropout_prob,
        )
        self.proj = Sequential(
            LayerNorm(hidden_dim),
            Linear(hidden_dim, out_channels)
        )
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        self.tensor_frame_encoder.reset_parameters()
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.proj[0].reset_parameters()
        self.proj[1].reset_parameters()
        nn.init.normal_(self.pos_embedding)
        nn.init.trunc_normal_(self.latents, std=0.02)
        nn.init.trunc_normal_(self.queries, std=0.02)

        for block in self.blocks:
            block.reset_parameters()

    def forward(self, tf):
        # pre-processing with shape (batch_size, num_colummns, hidden_dim)
        batch_size = len(tf)
        x, _ = self.tensor_frame_encoder(tf)
        x = x + self.pos_embedding

        # Encode input into latent of shape (batch_size, num_latents, hidden_dim) 
        latents = self.latents.repeat(batch_size, 1, 1)
        x = self.encoder(latents, x)
        
        # Transformer Blocks
        x = self.blocks(x)

        # Decode and projection: (batch_size, hidden_dim) -> (batch_size, num_classes)
        queries = self.queries.repeat(batch_size, 1, 1)
        x = self.decoder(queries, x).reshape(batch_size, -1)
        x = self.proj(x)
        return x


class TabPerceiverTransfer(TabPerceiver):
    r"""
        TabPerceiver for being trained on several datasets.
    """
    def reconstructIO(
        self,
        out_channels: int,
        num_features: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
    ):
        """ Reconstruct Input, Output layers, and positional encoding """
        stype_encoder_dict = {
            stype.categorical: EmbeddingEncoder(),
            stype.numerical: LinearEncoder(),
        }
        self.tensor_frame_encoder = StypeWiseFeatureEncoder(
            out_channels=self.hidden_dim,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        self.pos_embedding = Parameter(torch.empty(1, num_features, self.hidden_dim))
        self.proj = Sequential(
            LayerNorm(self.hidden_dim),
            Linear(self.hidden_dim, out_channels)
        )           
        self.freeze_transformer()
        self.reset_parameters_finetune()

    def freeze_transformer(self):
        """ Freeze Transformer blocks for finetuning """
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.latents.requires_grad = False
        self.queries.requires_grad = False
        
    def reset_parameters_finetune(self):
        """ initialize re-defined parameters"""
        self.tensor_frame_encoder.reset_parameters()
        torch.nn.init.normal_(self.pos_embedding)
        self.proj[0].reset_parameters()
        self.proj[1].reset_parameters()