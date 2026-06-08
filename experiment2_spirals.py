import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# optional but recommended:
# pip install torchdiffeq
try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_ODEINT = True
except:
    HAS_ODEINT = False


# DATA: SPIRAL-LIKE 2D PROBLEM
def make_spiral(n=2000, noise=0.2):
    X, y = make_circles(n_samples=n, noise=noise, factor=0.5)

    theta = np.arctan2(X[:, 1], X[:, 0])
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)

    X[:, 0] = r * np.cos(theta * 3)
    X[:, 1] = r * np.sin(theta * 3)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# BASE MLP (vector field)
class MLP(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, dim)
        )

    def forward(self, x):
        return self.net(x)


# Euler RESNET
class EulerBlock(nn.Module):
    def __init__(self, f, h=0.1):
        super().__init__()
        self.f = f
        self.h = h

    def forward(self, x):
        return x + self.h * self.f(x)


class EulerNet(nn.Module):
    def __init__(self, depth=6):
        super().__init__()
        self.f = MLP()
        self.layers = nn.ModuleList([EulerBlock(self.f) for _ in range(depth)])
        self.classifier = nn.Linear(2, 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)


# RK2 (Heun)
class RK2Block(nn.Module):
    def __init__(self, f, h=0.1):
        super().__init__()
        self.f = f
        self.h = h

    def forward(self, x):
        k1 = self.f(x)
        k2 = self.f(x + self.h * k1)
        return x + 0.5 * self.h * (k1 + k2)


class RK2Net(nn.Module):
    def __init__(self, depth=6):
        super().__init__()
        self.f = MLP()
        self.layers = nn.ModuleList([RK2Block(self.f) for _ in range(depth)])
        self.classifier = nn.Linear(2, 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)

# NEURAL ODE
class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP()

    def forward(self, t, x):
        return self.net(x)


class NeuralODE(nn.Module):
    def __init__(self, adaptive=True):
        super().__init__()
        self.func = ODEFunc()
        self.classifier = nn.Linear(2, 2)
        self.adaptive = adaptive

    def forward(self, x):

        t = torch.tensor([0.0, 1.0])

        if HAS_ODEINT:
            x = odeint(self.func, x, t, method="dopri5" if self.adaptive else "euler")[-1]
        else:
            # fallback fixed Euler
            for _ in range(10):
                x = x + 0.1 * self.func(0, x)

        return self.classifier(x)


# TRAINING
def train(model, X, y, epochs=20):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time()

    for _ in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

    elapsed = time.time() - start_time
    return elapsed


# MEMORY USAGE (CPU proxy)
def get_param_memory(model):
    return sum(p.numel() for p in model.parameters()) * 4 / 1024  # KB approx


# EVAL
def accuracy(model, X, y):
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        return (pred == y).float().mean().item()


if __name__ == "__main__":

    X, y = make_spiral()

    models = {
        "Euler ResNet": EulerNet(),
        "RK2 ResNet": RK2Net(),
        "Neural ODE (adaptive)": NeuralODE(adaptive=True),
        "Neural ODE (fixed)": NeuralODE(adaptive=False),
    }

    results = []

    for name, model in models.items():

        print(f"\n=== {name} ===")

        mem = get_param_memory(model)
        t = train(model, X, y)

        acc = accuracy(model, X, y)

        print("accuracy:", acc)
        print("time (s):", t)
        print("param memory (KB):", mem)

        results.append({
            "model": name,
            "accuracy": acc,
            "time": t,
            "memory_kb": mem
        })
        
    # PLOT RESULTS
    names = [r["model"] for r in results]
    accs = [r["accuracy"] for r in results]
    times = [r["time"] for r in results]

    plt.figure()
    plt.bar(names, accs)
    plt.xticks(rotation=45)
    plt.title("Accuracy comparison")
    plt.tight_layout()
    plt.savefig("exp2_accuracy.png")

    plt.figure()
    plt.bar(names, times)
    plt.xticks(rotation=45)
    plt.title("Training time comparison")
    plt.tight_layout()
    plt.savefig("exp2_time.png")

    print("\nSaved plots: exp2_accuracy.png, exp2_time.png")
