import torch
import torch.nn as nn

from src.model.attention import att
class Transformer(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = att(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self,x):

        x = x + self.attn(
            self.norm1(x)
        )
        x = x + self.mlp(self.norm2(x))
        x = self.dropout(x)
        return x

def test_transformer():
    block = Transformer()
    x = torch.randn(1, 10, 128)  # (batch_size, seq_length, embed_dim)
    y = block(x)
    assert y.shape == x.shape, "Output shape must match input shape"
    print("Transformer block test passed.")

if __name__ == "__main__":
    test_transformer()