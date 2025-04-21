from pyomo.environ import *
import numpy as np

def solve_qp(dx, Z, V):
    """
    Solves a QP problem using Pyomo with vector-style variable indexing.

    Minimize: 0.5*x[0]^2 + x[1]^2 + x[0]*x[1] + 3*x[0]
    Subject to: x[0] + x[1] >= 1, x[i] >= 0

    Returns:
        np.ndarray: [x[0], x[1], objective_value]
    """

    # get the kappa
    kappa = np.log(Z[1:] / Z[:-1]) / dx
    b = kappa*1; b[b < 0] = 0
    a = -kappa + b; a[a < 0] = 0
    return b.tolist(), a.tolist()
    model = ConcreteModel()

    # Define index set and variables
    model.I = RangeSet(0, 1)
    model.x = Var(model.I, domain=NonNegativeReals)

    # Objective
    model.obj = Objective(
        expr=0.5 * model.x[0]**2 + model.x[1]**2 + model.x[0] * model.x[1] + 3 * model.x[0],
        sense=minimize
    )

    # Constraint: x[0] + x[1] >= 1
    model.constraint = Constraint(expr=model.x[0] + model.x[1] >= 1)

    # Solve
    solver = SolverFactory('ipopt')
    solver.solve(model, tee=False)

    # Extract result as array
    x_vals = np.array([value(model.x[i]) for i in model.I])
    obj_val = value(model.obj)

    return np.append(x_vals, obj_val)
