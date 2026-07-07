import torch
import torch.nn as nn

class PatchEmb(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        color_channels=3,
        embed_dim=128
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.color_channels = color_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            color_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self, x):

        x = self.proj(x)

        x = x.flatten(2)

        x = x.transpose(1,2)

        return x
  
def test_patch_emb():
    model = PatchEmb()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, 64, 128), f"Unexpected output shape: {y.shape}"

if __name__ == "__main__":
    test_patch_emb()