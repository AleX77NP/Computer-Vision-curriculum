import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


if __name__ == '__main__':

    torch.manual_seed(42)

    x = np.random.rand(100, 1)
    y = 1 + 2 * x + .1 * np.random.randn(100, 1)

    # shuffle indices
    idx = np.arange(100)
    np.random.shuffle(idx)

    # 80 samples for training
    train_idx = idx[:80]

    # 20 samples for validation
    val_idx = idx[:80]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)

    model = nn.Sequential(nn.Linear(1, 1)).to(device)

    a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

    optimizer = optim.SGD([a, b], lr=1e-1)

    loss_fn = nn.MSELoss(reduction='mean')

    train_step = make_train_step(model, loss_fn, optimizer)
    losses = []

    for epoch in range(1000):
        loss = train_step(x_train_tensor, y_train_tensor)
        losses.append(loss)

    print(model.state_dict())