import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model.integrate import VisionTransformer
from src.diff.schedule import NoiseScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"


model = VisionTransformer().to(device)


model.load_state_dict(
    torch.load(
        "dit_42_colab_gpu.pth",
        map_location=device
    )
)


model.eval()

import torch


def sample(
    model,
    scheduler,
    n,
    device
):

    model.eval()


    x = torch.randn(
        n,
        3,
        32,
        32
    ).to(device)


    scheduler.beta = scheduler.beta.to(device)
    scheduler.alpha = scheduler.alpha.to(device)
    scheduler.alpha_bar = scheduler.alpha_bar.to(device)


    with torch.no_grad():

        for i in reversed(
            range(scheduler.timesteps)
        ):


            t = torch.full(
                (n,),
                i,
                device=device
            )


            predicted_noise = model(
                x,
                t
            )


            alpha = scheduler.alpha[i]
            alpha_bar = scheduler.alpha_bar[i]
            beta = scheduler.beta[i]


            x = (
                1 / torch.sqrt(alpha)
            ) * (
                x
                -
                (
                    (1-alpha)
                    /
                    torch.sqrt(
                        1-alpha_bar
                    )
                )
                *
                predicted_noise
            )


            if i > 0:

                noise = torch.randn_like(x)

                x += (
                    torch.sqrt(beta)
                    *
                    noise
                )


            x = torch.clamp(
                x,
                -1,
                1
            )


    return x

x = torch.randn(
    1,
    3,
    32,
    32
).to(device)

scheduler = NoiseScheduler()


scheduler.beta = scheduler.beta.to(device)
scheduler.alpha = scheduler.alpha.to(device)
scheduler.alpha_bar = scheduler.alpha_bar.to(device)

with torch.no_grad():
    saved_steps = []

    for i in tqdm(
        reversed(range(scheduler.timesteps)),
        total=scheduler.timesteps
    ):


        t = torch.tensor(
            [i],
            device=device
        )


        predicted_noise = model(
            x,
            t
        )



        alpha = scheduler.alpha[i]

        alpha_bar = scheduler.alpha_bar[i]

        beta = scheduler.beta[i]


        x = (
            1 / torch.sqrt(alpha)
        ) * (
            x
            -
            (
                (1-alpha)
                /
                torch.sqrt(
                    1-alpha_bar
                )
            )
            *
            predicted_noise
        )


        if i > 0:

            noise = torch.randn_like(
                x
            )

            x += (
                torch.sqrt(beta)
                *
                noise
            )
        x = torch.clamp(x, -1, 1)

        if i % 100 == 0:
            saved_steps.append(
                x.detach().cpu()
            )

plt.figure(figsize=(15, 3))

for idx, step in enumerate(saved_steps):

    img = step.squeeze(0)

    img = img.permute(
        1,
        2,
        0
    )

    img = img * 0.5 + 0.5

    img = img.clamp(
        0,
        1
    )


    plt.subplot(
        1,
        len(saved_steps),
        idx + 1
    )

    plt.imshow(img)

    plt.title(
        f"{900 - idx*100}"
    )

    plt.axis("off")


from torchvision.utils import make_grid


grid = make_grid(
    x,
    nrow=4,
    normalize=True
)


plt.imshow(
    grid.permute(1,2,0).cpu()
)

plt.axis("off")

plt.show()