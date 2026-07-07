from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_loaders(batch_size=64):

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,0.5,0.5),
            (0.5,0.5,0.5)
        )
    ])


    train_dataset = datasets.ImageFolder(
        root="data/CIFAR-10-images/train",
        transform=transform
    )


    test_dataset = datasets.ImageFolder(
        root="data/CIFAR-10-images/test",
        transform=transform
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    return train_loader, test_loader


if __name__ == "__main__":

    train, test = get_loaders()

    images, labels = next(iter(train))

    print(images.shape)
    print(labels.shape)
    print(labels[:10])