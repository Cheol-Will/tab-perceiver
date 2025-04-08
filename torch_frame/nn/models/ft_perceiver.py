from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential, Parameter
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

class MultiheadAttention(Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        input_kdim: int = None,
    ): 
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        if input_kdim is None:
            input_kdim = hidden_dim
        # if vdim is None:
        #     vdim = hidden_dim
        
        self.head_dim = hidden_dim // num_heads 
        self._query = Linear(hidden_dim, hidden_dim)
        self._key = Linear(input_kdim, hidden_dim)
        self._value = Linear(input_kdim, hidden_dim)  
        self.proj = Linear(hidden_dim, hidden_dim)
        
    def forward(self, query, key, value):

        B, N, D = query.shape
        H = self.num_heads
        head_dim = self.head_dim

        # (B, N, D) -> (B, N, H, D') -> (B, H, N, D')
        # In cross attention in PerceiverIO, seq_len can differ.
        Q = self._query(query).reshape(B, -1, H, head_dim).transpose(1,2)
        K = self._key(key).reshape(B, -1, H, head_dim).transpose(1,2)
        V = self._value(value).reshape(B, -1, H, head_dim).transpose(1,2)

        # (batch, head, seq_qeury, dim), (batch, head, seq_key, dim)
        A = F.softmax(torch.einsum("bhij,bhkj->bhik", Q, K) / (head_dim**(0.5)), dim=-1) 
        V = torch.einsum("bhij,bhjk->bhik", A, V) 
        V = V.transpose(1, 2).reshape(B, N, D) # (batch, seq_qeury, head, head_dim) -> (batch_size, seq_query, hidden_dim)
        out = self.proj(V)

        return out, A

class BasicDecoder(Module):
    def __init__(
        self, 
        seq_len: int,
        hidden_dim: int
    ):
        super(BasicDecoderQuery, self).__init__()
        # self.
    def decoder_query(self, inputs):
        pass

class FTPerceiver(Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        num_latent_array: int,
        latent_channels: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
    ) -> None:
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }
        
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        self.latent_encoder_query = Parameter(torch.empty(num_latent_array, latent_channels))
        self.latent_decoder_query = Parameter(torch.empty(out_channels, latent_channels)) # output shape of C x 1 
        # In pytorch frame benchmark setting, out_channels == num_classes
        
        # cross attention for latent encoder query
        self.latent_encoder = MultiheadAttention(latent_channels, num_heads, input_kdim=channels)
        self.backbone = FTTransformerConvs(channels=latent_channels, num_layers=num_layers)
        self.decoder = MultiheadAttention(latent_channels, num_heads)
        self.proj = Sequential(
            LayerNorm(latent_channels),
            ReLU(),
            Linear(latent_channels, 1)
        )
        # cross attention for latent decoder query

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        # self.latent_encoder.reset_parameters()
        self.backbone.reset_parameters()
        for m in self.proj:
            if not isinstance(m, ReLU):
                m.reset_parameters()

    def forward(self, tf):
        # pre-processing with shape (batch_size, colummns, 1)
        # do not embed here. 
        batch_size = tf.__len__()
        x, _ = self.encoder(tf)

        # Encode input into latent of shape (batch_size, N, K) where N, K are hyperparamters of latent space.
        latent_encoder_query = self.latent_encoder_query.repeat(batch_size, 1, 1)
        x, _ = self.latent_encoder(latent_encoder_query, x, x)
        
        # column-wise interaction (technically, latent interaction)
        # FTTransformer uses x_cls for prediction, but here we just use other tokens with decoder query array.
        x, x_cls = self.backbone(x)

        # decode latent into decoder query shape (batch_size, num_classes, K)
        latent_decoder_query = self.latent_decoder_query.repeat(batch_size, 1, 1)
        x, _ = self.decoder(latent_decoder_query, x, x)

        # project it into (batch_size, num_classes, 1) -> (batch_size, num_classes)
        x = self.proj(x).reshape(batch_size, -1)

        return x