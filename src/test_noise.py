import torch
import matplotlib.pyplot as plt

from src.data.dataset import get_loaders
from src.diff.schedule import NoiseScheduler


# load one CIFAR image
train_loader, _ = get_loaders(
    batch_size=1
)

image, _ = next(iter(train_loader))


scheduler = NoiseScheduler()


timesteps = [
    0,
    100,
    300,
    500,
    700,
    999
]


plt.figure(figsize=(12,3))


for idx, t_value in enumerate(timesteps):

    noise = torch.randn_like(image)


    t = torch.tensor(
        [t_value]
    )


    noisy = scheduler.add_noise(
        image,
        noise,
        t
    )


    img = noisy.squeeze(0)


    img = img.permute(
        1,
        2,
        0
    )


    # undo normalization
    img = (
        img * 0.5
        +
        0.5
    )


    img = img.clamp(
        0,
        1
    )


    plt.subplot(
        1,
        len(timesteps),
        idx+1
    )


    plt.imshow(img)


    plt.title(
        f"t={t_value}"
    )


    plt.axis("off")


plt.show()