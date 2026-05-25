import numpy as np
import torch

#  NUMPY SGD


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def update(self, w, b, dw, db):
        w = w - self.lr * dw
        b = b - self.lr * db
        return w, b

    def train_numpy(self, epochs=100):

        print("\n--- NumPy SGD Training ---")

        np.random.seed(0)

        X = np.random.randn(100, 1)
        y = 3 * X + 2 + 0.1 * np.random.randn(100, 1)

        w = np.random.randn(1, 1)
        b = np.zeros((1,))

        for i in range(epochs):

            y_pred = X @ w + b

            error = y_pred - y
            dw = (2 / len(X)) * (X.T @ error)
            db = (2 / len(X)) * np.sum(error)

            w, b = self.update(w, b, dw, db)

            if i % 10 == 0:
                loss = np.mean(error ** 2)
                print(f"Epoch {i}, Loss: {loss:.4f}")

        return w, b


# run NumPy
model = SGD(lr=0.01)
model.train_numpy()


#  PYTORCH SGD


class p_SGD:
    def __init__(self, w, b, lr):
        self.w = w
        self.b = b
        self.lr = lr

    def step(self):
        with torch.no_grad():
            self.w -= self.lr * self.w.grad
            self.b -= self.lr * self.b.grad

    def zero_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()



# PyTorch training function


def train_torch():

    print("\n--- PyTorch SGD Training ---")

    torch.manual_seed(0)

    # parameters
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # optimizer
    opt = p_SGD(w, b, lr=0.01)

    # data
    X = torch.randn(100, 1)
    y = 3 * X + 2 + 0.1 * torch.randn(100, 1)

    for i in range(100):

        # forward
        y_pred = X * w + b
        loss = ((y_pred - y) ** 2).mean()

        # backward
        loss.backward()

        # update
        opt.step()
        opt.zero_grad()

        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {loss.item():.4f}")

    


train_torch()