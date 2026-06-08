import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Настройки эксперимента
T = 10.0

alphas = [1, 2, 4, 8]

steps = [
    0.01,
    0.05,
    0.10,
    0.20
]

x0 = np.array([1.0, 0.0])


# Точное решение
# x' = A x
# A =
# [ 0 -a ]
# [ a  0 ]
def exact_solution(alpha, t):

    c = np.cos(alpha * t)
    s = np.sin(alpha * t)

    R = np.array([
        [c, -s],
        [s,  c]
    ])

    return R @ x0


# Euler
def euler_forward(A, h, N):

    x = x0.copy()

    trajectory = [x.copy()]

    for _ in range(N):

        x = x + h * (A @ x)

        trajectory.append(x.copy())

    return np.array(trajectory)


# Heun (RK2)
def heun_forward(A, h, N):

    x = x0.copy()

    trajectory = [x.copy()]

    for _ in range(N):

        k1 = A @ x

        k2 = A @ (x + h * k1)

        x = x + 0.5 * h * (k1 + k2)

        trajectory.append(x.copy())

    return np.array(trajectory)


# Обратный проход Euler
def euler_backward(A, h, trajectory):

    M = np.eye(2) + h * A

    M_inv = np.linalg.inv(M)

    x = trajectory[-1].copy()

    recovered = [x.copy()]

    for _ in range(len(trajectory)-1):

        x = M_inv @ x

        recovered.append(x.copy())

    recovered.reverse()

    return np.array(recovered)


# Обратный проход Heun
def heun_backward(A, h, trajectory):

    M = (
        np.eye(2)
        + h * A
        + 0.5 * h**2 * (A @ A)
    )

    M_inv = np.linalg.inv(M)

    x = trajectory[-1].copy()

    recovered = [x.copy()]

    for _ in range(len(trajectory)-1):

        x = M_inv @ x

        recovered.append(x.copy())

    recovered.reverse()

    return np.array(recovered)


# Ошибка траектории
def trajectory_error(traj1, traj2):

    return np.max(
        np.linalg.norm(
            traj1 - traj2,
            axis=1
        )
    )


# Основной цикл
results = []

for alpha in alphas:

    A = np.array([
        [0.0, -alpha],
        [alpha, 0.0]
    ])

    for h in steps:

        N = int(T / h)

        times = np.linspace(
            0,
            T,
            N + 1
        )

        exact = np.array([
            exact_solution(alpha, t)
            for t in times
        ])

        # Euler
        euler_traj = euler_forward(
            A,
            h,
            N
        )

        euler_rec = euler_backward(
            A,
            h,
            euler_traj
        )

        euler_forward_error = (
            trajectory_error(
                exact,
                euler_traj
            )
        )

        euler_reconstruction_error = (
            trajectory_error(
                euler_traj,
                euler_rec
            )
        )

        results.append({
            "alpha": alpha,
            "h": h,
            "method": "Euler",
            "forward_error":
                euler_forward_error,
            "reconstruction_error":
                euler_reconstruction_error
        })

        # Heun
        heun_traj = heun_forward(
            A,
            h,
            N
        )

        heun_rec = heun_backward(
            A,
            h,
            heun_traj
        )

        heun_forward_error = (
            trajectory_error(
                exact,
                heun_traj
            )
        )

        heun_reconstruction_error = (
            trajectory_error(
                heun_traj,
                heun_rec
            )
        )

        results.append({
            "alpha": alpha,
            "h": h,
            "method": "Heun",
            "forward_error":
                heun_forward_error,
            "reconstruction_error":
                heun_reconstruction_error
        })

        print(
            f"alpha={alpha:2d} "
            f"h={h:.2f} "
            f"done"
        )

        # Рисунки только для h=0.1
        if abs(h - 0.1) < 1e-12:

            plt.figure(figsize=(6, 6))

            plt.plot(
                euler_traj[:,0],
                euler_traj[:,1],
                label="Euler"
            )

            plt.plot(
                heun_traj[:,0],
                heun_traj[:,1],
                label="Heun"
            )

            plt.plot(
                exact[:,0],
                exact[:,1],
                "k--",
                linewidth=3,
                label="Exact"
            )

            plt.axis("equal")

            plt.legend()

            plt.title(
                f"Forward trajectories, alpha={alpha}"
            )

            plt.savefig(
                f"forward_alpha_{alpha}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            # -----------------------------

            euler_err = np.linalg.norm(
                euler_traj - euler_rec,
                axis=1
            )

            heun_err = np.linalg.norm(
                heun_traj - heun_rec,
                axis=1
            )

            plt.figure(figsize=(7, 4))

            plt.semilogy(
                times,
                euler_err,
                label="Euler"
            )

            plt.semilogy(
                times,
                heun_err,
                label="Heun"
            )

            plt.xlabel("t")

            plt.ylabel(
                "Reconstruction error"
            )

            plt.legend()

            plt.title(
                f"Reconstruction error, alpha={alpha}"
            )

            plt.savefig(
                f"reconstruction_alpha_{alpha}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()


# Сохранение таблицы
df = pd.DataFrame(results)

df.to_excel(
    "experiment1_results.xlsx",
    index=False
)

print(
    "\nSaved: experiment1_results.xlsx"
)

print("\nSummary:")

print(
    df.groupby(
        ["method"]
    )[
        [
            "forward_error",
            "reconstruction_error"
        ]
    ].mean()
)
