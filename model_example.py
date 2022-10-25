import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.a + self.b * x


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

    # Now we can create a model and send it at once to the device
    model = ManualLinearRegression().to(device)
    # We can also inspect its parameters using its state_dict
    print(model.state_dict())

    lr = 1e-1
    n_epochs = 1000

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # What is this?!?
        model.train()

        # No more manual prediction!
        # yhat = a + b * x_tensor
        yhat = model(x_train_tensor)

        loss = loss_fn(y_train_tensor, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(model.state_dict())