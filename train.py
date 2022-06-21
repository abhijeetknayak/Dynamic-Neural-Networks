import torch
from utilities.data import *
from utilities.transforms import *
from networks.models import *

def train():
    batch_size = 4
    num_epochs = 3
    lr = 0.001
    device = 'cuda'
    net = Model().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, test_loader = get_cifar10_data(transform=ft, batch_size=batch_size)
    for i in range(num_epochs):
        running_loss = 0.0
        for idx, samples in enumerate(train_loader):
            images, labels = samples

            # Send data to device
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            out = net(images)
            loss = criterion(out, labels)

            running_loss += loss.item()

            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    train()



