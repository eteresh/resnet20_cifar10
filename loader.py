import torch
import torchvision
import torchvision.transforms as transforms


def get_train_dataset_normalization():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8 * 1024, shuffle=True, num_workers=2)

    avg_image = torch.zeros(3, 32, 32)
    avg_image_dispersion = torch.zeros(3, 32, 32)
    n_images = 0
    for Z_batch, y_batch in trainloader:
        avg_image += Z_batch.sum(axis=0)
        avg_image_dispersion += (Z_batch * Z_batch).sum(axis=0)

        n_images += Z_batch.shape[0]
    avg_image /= n_images
    avg_image_dispersion = avg_image_dispersion / n_images - avg_image ** 2
    avg_image_deviation = avg_image_dispersion.sqrt()
    return avg_image, avg_image_deviation


avg_image, avg_image_deviation = get_train_dataset_normalization()


def normalize_image(image):
    return (image - avg_image) / avg_image_deviation


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(normalize_image),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(normalize_image),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
