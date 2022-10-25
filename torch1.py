import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import numpy as np

# generate data
np.random.seed(42)
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

np.random.seed(42)

a = np.random.randn(1)
b = np.random.randn(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

print(type(x_train), type(x_train_tensor), x_train_tensor.type())