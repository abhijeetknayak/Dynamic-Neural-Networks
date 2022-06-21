import torch
import torchvision
import torchvision.transforms as T

def get_cifar10_data(transform, batch_size):
    # Get the Train and Test data
    train_set = torchvision.datasets.CIFAR10(root='./data', transform=transform,
                                             train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', transform=transform,
                                            train=False, download=True)

    # Create the DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                               shuffle=False, num_workers=2)

    return train_loader, test_loader
