import numpy as np
import torch
from sklearn.metrics import accuracy_score

from loader import trainloader, testloader


def get_accuracy(net, loader):
    scores = []
    for Z_batch, y_batch in loader:
        y_pred = net(Z_batch.to('cuda'))
        y_pred = y_pred.cpu().detach().numpy()
        scores.append(accuracy_score(y_batch, np.argmax(y_pred, axis=1)))
    return np.array(scores).mean()


def train_epoch(net, criterion, optimizer):
    net.train()
    for Z_batch, y_batch in trainloader:
        Z_batch, y_batch = Z_batch.to('cuda'), y_batch.to('cuda')

        y_pred = net(Z_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return get_accuracy(net, testloader)
