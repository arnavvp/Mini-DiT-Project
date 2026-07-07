import torch
import matplotlib.pyplot as plt

from src.data.dataset import get_loaders
from src.model.integrate import VisionTransformer
from src.diff.schedule import NoiseScheduler


device = "cuda" if torch.cuda.is_available() else "cpu"


loader, _ = get_loaders(batch_size=1)

image, _ = next(iter(loader))

image = image.to(device)

scheduler = NoiseScheduler()


noise = torch.randn_like(image)


t = torch.tensor(
    [500],
    device=device
)


noisy = scheduler.add_noise(
    image,
    noise,
    t
)

model = VisionTransformer().to(device)

model.load_state_dict(
    torch.load(
        "dit.pth",
        map_location=device
    )
)

model.eval()

with torch.no_grad():

    predicted_noise = model(
        noisy,
        t
    )

denoised = noisy - predicted_noise

images = [
    image,
    noisy,
    denoised
]

titles = [
    "Original",
    "Noisy",
    "Denoised"
]

print(
    "MSE:",
    ((noise - predicted_noise)**2).mean().item()
)

for i in range(3):

    img = images[i]

    img = img.squeeze(0)
    img = img.cpu()

    img = img.permute(
        1,
        2,
        0
    )

    img = img * 0.5 + 0.5

    img = img.clamp(0,1)


    plt.subplot(
        1,
        3,
        i+1
    )

    plt.imshow(img)

    plt.title(
        titles[i]
    )

    plt.axis("off")


plt.show()