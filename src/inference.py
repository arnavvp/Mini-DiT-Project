import torch
from PIL import Image
from torchvision import transforms

from src.model.integrate import VisionTransformer

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

device = "cuda" if torch.cuda.is_available() else "cpu"


model = VisionTransformer()

model.load_state_dict(
    torch.load(
        "vit.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

transform = transforms.Compose(
    [
        transforms.Resize(
            (32,32)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5,0.5,0.5),
            (0.5,0.5,0.5)
        )
    ]
)

image = Image.open(
    "test.png"
).convert("RGB")

image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)

prob = torch.softmax(
       output,
       dim=1
   )


prediction = prob.argmax(
    dim=1
)



print(
    "Prediction:",
    classes[prediction.item()]
)


print(
    "Confidence:",
    prob.max().item()
)