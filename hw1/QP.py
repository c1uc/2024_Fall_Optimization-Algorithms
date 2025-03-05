import gurobipy as gp
import numpy as np

m = gp.Model("qp")

p = np.array([0.12, 0.1, 0.07, 0.03])
ones = np.ones(4)
Sigma = np.array(
    [
        [0.2, -0.03, 0, 0],
        [-0.03, 0.1, -0.02, 0],
        [0, -0.02, 0.05, 0],
        [0, 0, 0, 0.01]
    ]
)
mu_values = [0, 0.1, 1.0, 2.0, 5.0, 10.0]

x = m.addMVar((4, ), name="x", lb=0.0)

m.addConstr(x @ ones == 1, "c0")

res = []
for mu in mu_values:
    obj = -p @ x + mu * (x @ Sigma @ x)
    m.setObjective(obj)

    m.optimize()

    res.append(f"Mu: {mu}, Obj: {m.ObjVal:g}, x: {x.X}")

with open("QP.txt", "w+") as f:
    f.write("\n".join(res))