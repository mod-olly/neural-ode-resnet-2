import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from torchdiffeq import odeint
    HAS_ODE = True
except:
    HAS_ODE = False


# LINEAR SYSTEM (theoretical backbone)
# x' = A x
def f(x, A):
    return x @ A.T


# EULER STEP
def euler_step(x, A, h):
    return x + h * f(x, A)


# RK2 STEP (Heun)
def rk2_step(x, A, h):
    k1 = f(x, A)
    k2 = f(x + h * k1, A)
    return x + 0.5 * h * (k1 + k2)


# NEURAL ODE SOLVER (continuous model)
class ODEFunc(torch.nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = torch.nn.Parameter(A.clone())

    def forward(self, t, x):
        return x @ self.A.T


def neural_ode(x0, A, T=1.0, steps=20):
    func = ODEFunc(A)
    t = torch.linspace(0, T, steps)

    if HAS_ODE:
        xT = odeint(func, x0, t, method="dopri5")[-1]
    else:
        # fallback fixed Euler
        x = x0.clone()
        h = T / steps
        for _ in range(steps):
            x = x + h * func(0, x)
        xT = x

    return xT


# FORWARD DYNAMICS
def rollout(x0, A, h, N, method):
    x = x0.clone()

    for _ in range(N):
        if method == "euler":
            x = euler_step(x, A, h)
        elif method == "rk2":
            x = rk2_step(x, A, h)

    return x


# RECONSTRUCTION STABILITY (inverse proxy)
def reconstruction_error(A, x0, h, N, method):
    x = x0.clone()
    traj = []

    for _ in range(N):
        if method == "euler":
            x = euler_step(x, A, h)
        else:
            x = rk2_step(x, A, h)
        traj.append(x.clone())

    xT = traj[-1].clone()

    # linear inverse approximation
    if method == "euler":
        M = torch.eye(2) + h * A
    else:
        M = torch.eye(2) + h * A + 0.5 * h**2 * (A @ A)

    M_inv = torch.inverse(M)

    x_rec = xT.clone()

    for _ in range(N):
        x_rec = x_rec @ M_inv.T

    return torch.norm(x_rec - x0).item()


# STABILITY METRIC (key quantity)
def stability_score(final_x):
    return torch.norm(final_x).item()


# GRID
alphas = [0.5, 1, 2, 4, 8]
hs = [0.01, 0.05, 0.1, 0.2]
methods = ["euler", "rk2", "neural_ode"]


# EXPERIMENT
results = []

x0 = torch.tensor([[1.0, 0.5]], dtype=torch.float32)

A0 = torch.tensor([
    [0.0, -1.0],
    [1.0,  0.0]
], dtype=torch.float32)


for method in methods:

    for alpha in alphas:

        A = alpha * A0

        for h in hs:

            N = int(1.0 / h)

            # forward
            if method == "neural_ode":
                x_final = neural_ode(x0, A)
                rec_err = np.nan  # not defined for continuous solver
            else:
                x_final = rollout(x0, A, h, N, method)
                rec_err = reconstruction_error(A, x0, h, N, method)

            results.append({
                "method": method,
                "alpha": alpha,
                "h": h,
                "stability": stability_score(x_final),
                "reconstruction_error": rec_err
            })

            print(
                f"{method:12} α={alpha:>3} h={h:.3f} "
                f"stab={stability_score(x_final):.4f} "
                f"rec={rec_err}"
            )


# DATAFRAME
df = pd.DataFrame(results)
df.to_csv("ablation_results.csv", sep=";", index=False)


# PLOTS
# ---- reconstruction error ----
plt.figure(figsize=(7,5))

for method in methods[:-1]:
    for alpha in alphas:
        d = df[(df["method"] == method) & (df["alpha"] == alpha)]
        plt.plot(d["h"], d["reconstruction_error"],
                 marker="o",
                 label=f"{method}, α={alpha}")

plt.xlabel("step size h")
plt.ylabel("reconstruction error")
plt.title("Inverse stability vs discretization")
plt.legend()
plt.grid()
plt.savefig("ablation_reconstruction.png")


# ---- stability (phase transition) ----
plt.figure(figsize=(7,5))

for method in methods:
    for h in hs:
        d = df[(df["method"] == method) & (df["h"] == h)]
        plt.plot(d["alpha"], d["stability"],
                 marker="o",
                 label=f"{method}, h={h}")

plt.xlabel("alpha (Lipschitz scaling)")
plt.ylabel("trajectory norm (stability)")
plt.title("Phase transition: stability regime")
plt.legend()
plt.grid()
plt.savefig("ablation_stability.png")


print("\nSaved:")
print("ablation_results.csv")
print("ablation_reconstruction.png")
print("ablation_stability.png")
