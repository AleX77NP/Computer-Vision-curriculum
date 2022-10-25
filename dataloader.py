from torch.utils.data import Dataset, TensorDataset, DataLoader
from model import ManualLinearRegression
from sequential import make_train_step
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    x = np.random.rand(100, 1)
    y = 1 + 2 * x + .1 * np.random.randn(100, 1)

    # shuffle indices
    idx = np.arange(100)
    np.random.shuffle(idx)

    # 80 samples for training
    train_idx = idx[:80]

    # 20 samples for validation
    val_idx = idx[:80]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    train_data = CustomDataset(x_train_tensor, y_train_tensor)
    # print(train_data[0])
    #
    # train_data = TensorDataset(x_train_tensor, y_train_tensor)
    # print(train_data[0])

    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

    model = ManualLinearRegression().to(device)

    lr = 1e-1
    n_epochs = 1000

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []
    train_step = make_train_step(model, loss_fn, optimizer)

    for epoch in range(1000):
        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            loss = train_step(x_batch, y_batch)
            losses.append(loss)

    print(model.state_dict())