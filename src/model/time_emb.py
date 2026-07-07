import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):

    def __init__(
        self,
        embed_dim
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )


    def forward(self, t):

        half_dim = self.embed_dim // 2


        frequencies = torch.exp(
            -math.log(10000)
            *
            torch.arange(
                half_dim,
                device=t.device
            )
            /
            half_dim
        )


        x = (
            t[:, None]
            *
            frequencies[None]
        )


        embedding = torch.cat(
            [
                torch.sin(x),
                torch.cos(x)
            ],
            dim=-1
        )


        embedding = self.mlp(
            embedding
        )


        return embedding