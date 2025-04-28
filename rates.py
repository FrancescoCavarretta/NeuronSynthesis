from pyomo.environ import *
import numpy as np


def mk_objective(model, kappa, Z, V):
    terms = []
    for i in range(kappa.size):
        
        # square difference
        A = 2 / kappa[i] * (Z[i+1] - Z[i]) * Z[i+1] / Z[i]
        B = V[i] * np.power(Z[i+1] / Z[i], 2) - (Z[i+1] - Z[i]) * Z[i+1] / Z[i]
        Q = A * A
        c = 2 * A * (B - V[i+1])

        terms.append(model.b[i] ** 2 * Q + model.b[i] * c)
        
    return sum(terms)


def solve_qp(dx, Z, V, n_bif=None, max_iter=10000, kappa_Penalty=5.0):
    """
    Solves a QP problem using Pyomo with vector-style variable indexing.

    Minimize: 0.5*x[0]^2 + x[1]^2 + x[0]*x[1] + 3*x[0]
    Subject to: x[0] + x[1] >= 1, x[i] >= 0

    Returns:
        np.ndarray: [x[0], x[1], objective_value]
    """

    # get the kappa
    kappa = np.log(Z[1:] / Z[:-1]) / dx

    model = ConcreteModel()

    # Define index set and variables
    model.b = Var(range(kappa.size), domain=NonNegativeReals)

    # Objective
    model.obj = Objective(
        expr=mk_objective(model, kappa, Z, V),
        sense=minimize
    )

    # Constraint: 
    model.constraints = ConstraintList()
    for i in range(kappa.size):
        model.constraints.add(model.b[i] >= kappa[i])

    # if we have number of bifurcations as contraings
    if n_bif:        
        
        # define 2 slack variables for eventual constraints of mean and variance of bifurcations
        model.s = Var(range(2), domain=Reals)
        
        # add the slack variables to the objective function
        model.obj.expr += kappa_Penalty * (abs(model.s[0]) ** 2 + abs(model.s[1]) ** 2 )

        # constraint the average number of bifurcations
        f = (Z[1:] - Z[:-1]) / kappa
        model.constraints.add(sum(f[i] * model.b[i] for i in range(kappa.size)) + model.s[0] == n_bif[0])
        #model.constraints.add(sum(f[i] * model.b[i] for i in range(kappa.size)) + model.s[0] == n_bif[0])
        
        # constrain the variance for the number of bifurcations
        f2 = - Z[:-1] * (np.power(Z[1:] / Z[:-1], 2) - 2 * kappa * dx * Z[1:] / Z[:-1] - 1) / kappa ** 2 + V[:-1] * (np.power(Z[1:] / Z[:-1], 2) - 2 * (Z[1:] / Z[:-1]) + 1) / kappa ** 2
        f3 = 2 * Z[:-1] / kappa * (np.power(Z[1:] / Z[:-1], 2) - 2 * kappa * dx * Z[1:] / Z[:-1] - 1) / kappa ** 2
        var_terms = [ ]
        for i in range(kappa.size):
            var_terms.append(f2[i] * model.b[i] ** 2 + f3[i] * model.b[i] ** 3)

        for i in range(kappa.size - 1):
            term_all_Exps = 0
            for j in range(i + 1, kappa.size):
                term_all_Exps += np.prod([np.exp( kappa[k] * dx) for k in range(i + 1, j + 1)])
            var_terms.append(2 * term_all_Exps * (f2[i] * model.b[i] ** 2 + f3[i] * model.b[i] ** 3))
        
##        #model.constraints.add(sum(var_terms) + model.s[1] == n_bif[1])
        model.constraints.add(sum(var_terms) + model.s[1]  == n_bif[1]) 

        
    # Solve
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = max_iter
    solver.solve(model, tee=False)

    # Extract bifurcation rates as array
    b = np.array([value(model.b[i]) for i in model.b])
    b[b < 0] = 0.

    # calculate annihilation rates
    a = - kappa + b
    a[a < 0] = 0.

    
    return b, a
