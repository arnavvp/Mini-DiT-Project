import torch
import torch.nn as nn
import math

from src.model.patch_emb import PatchEmb
from src.model.transformer import Transformer
from src.model.time_emb import TimeEmbedding

class VisionTransformer(nn.Module):

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        channels=3,
        embed_dim=128,
        depth=6,
        num_heads=8,
        num_classes=10
    ):
        super().__init__()
        self.patch_emb = PatchEmb(img_size, patch_size, channels, embed_dim)
        self.img_size = img_size
        self.patch_size = patch_size
        self.channels = channels
        num_patches = self.patch_emb.num_patches
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.time_embedded = TimeEmbedding(embed_dim)
        self.transformerBlocks = nn.Sequential(
            *[Transformer(embed_dim, num_heads) for _ in range(depth)]
        )
        self.normfinal = nn.LayerNorm(embed_dim)
        self.patch_decoder = nn.Linear(embed_dim, patch_size*patch_size*channels)

    def unpatchify(self, x):

        B = x.shape[0]

        p = self.patch_size
        c = self.channels

        h = self.img_size // p
        w = self.img_size // p


        x = x.reshape(
            B,
            h,
            w,
            p,
            p,
            c
        )


        x = x.permute(
            0,
            5,
            1,
            3,
            2,
            4
        )


        x = x.reshape(
            B,
            c,
            self.img_size,
            self.img_size
        )

        return x


    def forward(self, x, t):
        x = self.patch_emb(x)
        batch_size = x.shape[0]
        x = x + self.position_embedding
        t = self.time_embedded(t)
        t = t.unsqueeze(1)
        x = x + t
        x = self.transformerBlocks(x)
        x = self.normfinal(x)
        x = self.patch_decoder(x)
        x = self.unpatchify(x)
        return x

if __name__ == "__main__":

    img = torch.randn(32,3,32,32)

    model = VisionTransformer()

    t = torch.randint(
        0,
        1000,
        (32,)
    )

    y = model(img,t)

    print(y.shape)