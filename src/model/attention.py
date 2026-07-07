import torch
import torch.nn as nn
import math

class att(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_out = nn.Linear(embed_dim, embed_dim)
        

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv_proj(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, self.embed_dim)
        return self.attn_out(out)
    
def test_attention():
    x = torch.randn(32,65,128)

    attention = att()

    y = attention(x)

    print(y.shape)

if __name__ == "__main__":
    test_attention()
