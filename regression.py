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

print(a, b)

lr = 0.1

n_epochs = 1000

# gradient descent
for epoch in range(n_epochs):

    yhat = a + b * x_train

    error = (y_train - yhat)
    loss = (error ** 2).mean()

    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()

    a = a - lr * a_grad
    b = b - lr * b_grad

print(a, b)

print("\n")


# to compare
from sklearn.linear_model import LinearRegression
linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])

# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e


