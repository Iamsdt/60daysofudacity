from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def calculate_mean_std(path, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    data = datasets.ImageFolder(path, transform=transform)
    print(len(data))
    loader = DataLoader(data, batch_size=batch_size)

    mean = 0.
    std = 0.
    nb_samples = 0.

    for images, _ in loader:
        batch_samples = images.size(0)
        data = images.view(batch_samples, images.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        break

    mean /= nb_samples
    std /= nb_samples

    mean = mean.numpy()
    std = std.numpy()

    print("Mean: ", mean)
    print("Std: ", std)

    return [mean, std]
