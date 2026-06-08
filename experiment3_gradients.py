import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

try:
    from torchdiffeq import odeint
    HAS_ODEINT = True
except:
    HAS_ODEINT = False


# LINEAR VECTOR FIELD (consistent with theory)
# x' = A x
class ODEFunc(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = nn.Parameter(A.clone())

    def forward(self, t, x):
        return x @ self.A.T


# LOSS (terminal)
def loss_fn(x):
    return (x ** 2).sum()


# FORWARD SOLVE (Neural ODE)
def solve_forward(func, x0, T=1.0, n_steps=20):
    t = torch.linspace(0, T, n_steps)

    if HAS_ODEINT:
        xT = odeint(func, x0, t, method="dopri5")[-1]
    else:
        # fixed Euler fallback
        x = x0.clone()
        h = T / n_steps
        for _ in range(n_steps):
            x = x + h * func(0, x)
        xT = x

    return xT


# DISCRETE BACKPROP (Euler ResNet view)
def discrete_backprop(A, x0, h, N):
    A = A.clone().requires_grad_(True)
    x = x0.clone()

    for _ in range(N):
        x = x + h * (x @ A.T)

    loss = loss_fn(x)
    loss.backward()

    return A.grad.detach()


# CONTINUOUS ADJOINT (Neural ODE style)
def continuous_adjoint(A, x0, T=1.0, n_steps=50):
    func = ODEFunc(A)
    xT = solve_forward(func, x0, T, n_steps)

    # adjoint variable
    aT = 2 * xT  # dL/dx for L = ||x||^2

    # backward integration (reverse time)
    t = torch.linspace(T, 0, n_steps)

    a = aT.clone()
    x = xT.clone()

    grads = torch.zeros_like(A)

    h = T / n_steps

    for _ in range(n_steps):
        dx = x @ func.A.T
        da = -(a @ func.A)  # linearized adjoint

        grads += (a.T @ x).detach() * h  # simplified accumulation

        x = x - h * dx
        a = a - h * da

    return grads


# RECONSTRUCTION ERROR (stability indicator)
def reconstruction_error(A, x0, h, N):
    x = x0.clone()
    traj = []

    for _ in range(N):
        x = x + h * (x @ A.T)
        traj.append(x.clone())

    xT = traj[-1].clone()

    # backward reconstruction
    x_rec = xT.clone()
    M = torch.eye(2) + h * A
    M_inv = torch.inverse(M)

    for _ in range(N):
        x_rec = x_rec @ M_inv.T

    return torch.norm(x_rec - x0).item()


# EXPERIMENT GRID
alphas = [0.5, 1, 2, 4, 8]
hs = [0.01, 0.05, 0.1, 0.2]


# MAIN LOOP
results = []

x0 = torch.tensor([[1.0, 0.5]], dtype=torch.float32)

A0 = torch.tensor([
    [0.0, -1.0],
    [1.0,  0.0]
], dtype=torch.float32)


for alpha in alphas:
    A = alpha * A0

    for h in hs:

        N = int(1.0 / h)

        # --- discrete backprop ---
        g_backprop = discrete_backprop(A, x0, h, N)

        # --- continuous adjoint ---
        g_adj = continuous_adjoint(A, x0, T=1.0, n_steps=50)

        # --- mismatch ---
        grad_error = torch.norm(g_backprop - g_adj).item()

        # --- reconstruction stability ---
        rec_error = reconstruction_error(A, x0, h, N)

        results.append({
            "alpha": alpha,
            "h": h,
            "grad_error": grad_error,
            "reconstruction_error": rec_error
        })

        print(
            f"α={alpha:>3} h={h:.3f} "
            f"grad_err={grad_error:.6f} "
            f"rec_err={rec_error:.6f}"
        )


# ANALYSIS PLOTS
import pandas as pd

df = pd.DataFrame(results)


# ---- gradient error vs h ----
plt.figure(figsize=(6,4))
for alpha in alphas:
    d = df[df["alpha"] == alpha]
    plt.plot(d["h"], d["grad_error"], marker="o", label=f"α={alpha}")

plt.xlabel("step size h")
plt.ylabel("gradient error (backprop vs adjoint)")
plt.title("Gradient inconsistency vs discretization")
plt.legend()
plt.grid()
plt.savefig("exp3_grad_error_vs_h.png")


# ---- reconstruction error ----
plt.figure(figsize=(6,4))
for alpha in alphas:
    d = df[df["alpha"] == alpha]
    plt.plot(d["h"], d["reconstruction_error"], marker="o", label=f"α={alpha}")

plt.xlabel("step size h")
plt.ylabel("reconstruction error")
plt.title("Backward stability of dynamics")
plt.legend()
plt.grid()
plt.savefig("exp3_reconstruction_vs_h.png")


print("\nSaved:")
print("exp3_grad_error_vs_h.png")
print("exp3_reconstruction_vs_h.png")
