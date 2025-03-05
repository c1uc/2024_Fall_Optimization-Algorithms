import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
import scipy.io as sio
import time

mat = sio.loadmat('./473500_wk.mat')['W']
n, p = mat.shape
mat = torch.from_numpy(mat).float()

def objective(x, a): # -sum(log(W*x))
    prod = torch.matmul(a, x)
    return -torch.sum(torch.log(prod))

def solve_linear_subproblem(grad, p):
    model = gp.Model("linear_subproblem")
    model.setParam('OutputFlag', 0)

    x = model.addMVar((p, ), lb=0.0, name="x")
    ones = np.ones(p)

    model.addConstr(x @ ones == 1, "c0") # x sum to 1

    model.setObjective(grad @ x, GRB.MINIMIZE)

    model.optimize()

    x_opt = np.array(x.X)
    return x_opt


def frank_wolfe(a, p, start_time, max_iter=20000):
    x = torch.ones(p, dtype=torch.float32) / p  # Uniform initialization
    records = []

    records.append((time.time() - start_time,objective(x, a).item()))

    for k in range(max_iter):
        x_ = x.clone().detach().requires_grad_(True)
        obj = objective(x_, a)
        obj.backward()
        grad = x_.grad

        s = solve_linear_subproblem(grad.numpy(), p)

        gamma = 2 / (k + 2) # k from 0
        x_new = (1 - gamma) * x + gamma * torch.tensor(s, dtype=torch.float32)

        x = x_new
        records.append((time.time() - start_time, objective(x, a).item()))

    return x, records


if __name__ == "__main__":
    start = time.time()
    x, records = frank_wolfe(mat, p, start)

    fx = objective(x, mat)

    x = [_[0] for _ in records]
    y = [_[1] - fx for _ in records]

    import matplotlib.pyplot as plt

    plt.plot(x, y)
    plt.xlabel("Time (s)")
    plt.ylabel("$f(x)-f(x^*)$")
    plt.yscale("log")
    plt.title("Sub-optimality gap")
    plt.savefig("frank_wolfe.png")
    plt.show()


