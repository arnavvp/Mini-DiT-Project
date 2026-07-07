import torch
import torch.nn as nn
from tqdm import tqdm
import math
from src.model.integrate import VisionTransformer
from src.data.dataset import get_loaders
from src.diff.schedule import NoiseScheduler


device = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_psnr(
    img1,
    img2
):

    mse = torch.mean(
        (img1 - img2) ** 2
    )


    if mse == 0:
        return float("inf")


    psnr = (
        20
        *
        math.log10(
            2.0 / math.sqrt(mse.item())
        )
    )


    return psnr


model = VisionTransformer().to(device)

model.load_state_dict(
    torch.load(
        "dit.pth",
        map_location=device
    )
)

model.eval()


_, test_loader = get_loaders()


scheduler = NoiseScheduler()

scheduler.alpha_bar = (
    scheduler.alpha_bar.to(device)
)


criterion = nn.MSELoss()


total_loss = 0
total_psnr = 0


with torch.no_grad():

    for images, _ in tqdm(test_loader):

        images = images.to(device)


        noise = torch.randn_like(
            images
        )


        t = torch.randint(
            0,
            scheduler.timesteps,
            (images.shape[0],),
            device=device
        )


        noisy = scheduler.add_noise(
            images,
            noise,
            t
        )


        pred_noise = model(
            noisy,
            t
        )


        loss = criterion(
            pred_noise,
            noise
        )

        alpha_bar = scheduler.alpha_bar[t]

        alpha_bar = alpha_bar.view(
            -1,
            1,
            1,
            1
        )


        denoised = (
            noisy
            -
            torch.sqrt(1 - alpha_bar)
            *
            pred_noise
        )


        denoised = (
            denoised
            /
            torch.sqrt(alpha_bar)
        )
        denoised = torch.clamp(
                denoised,
                -1,
                1
            )
        psnr = calculate_psnr(
            images,
            denoised
        )

        total_psnr += psnr

        total_loss += loss.item()


print(
    "Test Noise MSE:",
    total_loss / len(test_loader)
)

print(
    "Test PSNR:",
    total_psnr / len(test_loader)
)