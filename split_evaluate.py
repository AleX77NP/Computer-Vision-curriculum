from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader
from model import ManualLinearRegression
from sequential import make_train_step
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = np.random.rand(100, 1)
    y = 1 + 2 * x + .1 * np.random.randn(100, 1)

    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(x_tensor, y_tensor)

    train_dataset, val_dataset = random_split(dataset, [80, 20])

    train_loader = DataLoader(dataset=train_dataset, batch_size=16)
    val_loader = DataLoader(dataset=train_dataset, batch_size=20)

    model = ManualLinearRegression().to(device)

    lr = 1e-1
    n_epochs = 1000

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []
    val_losses = []
    train_step = make_train_step(model, loss_fn, optimizer)

    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            loss = train_step(x_batch, y_batch)
            losses.append(loss)

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                model.eval()

                yhat = model(x_val)
                val_loss = loss_fn(y_val, yhat)
                val_losses.append(val_loss.item())

    print(model.state_dict())

