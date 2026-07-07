import torch
import torch.nn as nn

from src.model.integrate import VisionTransformer
from src.data.dataset import get_loaders
from src.engine.eval import evaluate
from tqdm import tqdm
from src.diff.schedule import NoiseScheduler
torch.set_num_threads(8)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = VisionTransformer().to(device)


scheduler = NoiseScheduler()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.05
)

epochs = 7
train_loader, _ = get_loaders()

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, _ in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}"
        ):

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


            noisy_images = scheduler.add_noise(
                images,
                noise,
                t
            )


            predicted_noise = model(
                noisy_images,
                t
            )


            loss = criterion(
                predicted_noise,
                noise
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            total_loss += loss.item()


            print(
                "Loss:",
                total_loss / len(train_loader)
            )


            torch.save(
                model.state_dict(),
                "dit.pth"
            )