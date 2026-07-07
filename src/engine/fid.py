import torch
from tqdm import tqdm

from torchmetrics.image.fid import FrechetInceptionDistance

from src.data.dataset import get_loaders
from src.model.integrate import VisionTransformer
from src.diff.schedule import NoiseScheduler
from src.sample import sample

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = VisionTransformer().to(device)


model.load_state_dict(
    torch.load(
        "dit.pth",
        map_location=device
    )
)


model.eval()


scheduler = NoiseScheduler()

fid = FrechetInceptionDistance(
    feature=2048
).to(device)

_, loader = get_loaders()

count = 0


for images, _ in tqdm(loader):


    images = images.to(device)


    batch_size = images.shape[0]


    generated = sample(
        model,
        scheduler,
        batch_size,
        device
    )


    # convert [-1,1] -> [0,255]

    real = (
        (images * 0.5 + 0.5)
        *
        255
    )


    fake = (
        (generated * 0.5 + 0.5)
        *
        255
    )


    real = real.to(
        torch.uint8
    )


    fake = fake.to(
        torch.uint8
    )


    fid.update(
        real,
        real=True
    )


    fid.update(
        fake,
        real=False
    )


    count += batch_size


    if count >= 1000:
        break



print(
    "FID:",
    fid.compute().item()
)