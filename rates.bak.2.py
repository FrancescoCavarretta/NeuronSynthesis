from pyomo.environ import *
import numpy as np
from functools import reduce 
import operator

def prod(iterable):
    return reduce(operator.mul, iterable, 1)


taylor_exp = lambda model, n: [(model.s[i] ** i) / np.math.factorial(i) for i in range(n + 1)]


def mk_objective(model, h, kappa, Z, V):
    mean_obj = sum(((Z[i + 1]) ** 2 * ((1 + (h * model.s[i]) + (1 / 2) * (h * model.s[i]) ** 2 + (1 / 6) * (h * model.s[i]) ** 3  + (1 / 24) * (h * model.s[i]) ** 4  + (1 / 120) * (h * model.s[i]) ** 5) - 1) ** 2) ** 4 for i in range(kappa.size))
    var_obj = sum( ((2 * model.b[i] / (kappa[i] + model.s[i]) - 1) * (1 + (h * model.s[i]) + (1 / 2) * (h * model.s[i]) ** 2 + (1 / 6) * (h * model.s[i]) ** 3  + (1 / 24) * (h * model.s[i]) ** 4  + (1 / 120) * (h * model.s[i]) ** 5) * (Z[i + 1] / Z[i]) * (Z[i + 1] * (1 + (h * model.s[i]) + (1 / 2) * (h * model.s[i]) ** 2 + (1 / 6) * (h * model.s[i]) ** 3  + (1 / 24) * (h * model.s[i]) ** 4  + (1 / 120) * (h * model.s[i]) ** 5) - Z[i]) + \
                   V[i] * (Z[i + 1] / Z[i]) ** 2 * (1 + (h * model.s[i]) + (1 / 2) * (h * model.s[i]) ** 2 + (1 / 6) * (h * model.s[i]) ** 3  + (1 / 24) * (h * model.s[i]) ** 4  + (1 / 120) * (h * model.s[i]) ** 5) ** 2 - V[i + 1]) ** 2 for i in range(kappa.size))
    return mean_obj, var_obj



def solve_qp(step_size, dx, Z, V, n_bif=None, max_iter=10000, kappa_Penalty_Mean=5.0, kappa_Penalty_Var=5.0, kappa_Penalty_Slack_0=5.0, kappa_Penalty_Slack_1=5.0):
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
    model.s = Var(range(kappa.size), domain=Reals)
    
    mean_obj, var_obj = mk_objective(model, dx, kappa, Z, V)
    # Objective
    model.obj = Objective(
        expr=kappa_Penalty_Mean * mean_obj + kappa_Penalty_Var * var_obj + kappa_Penalty_Slack_0 * sum(model.s[i] ** 4 for i in range(kappa.size)),
        sense=minimize
    )

    # Constraint: 
    model.constraints = ConstraintList()
    for i in range(kappa.size):
        model.constraints.add(model.b[i] - model.s[i] >= kappa[i])
        model.constraints.add(model.b[i] * step_size <= 1)
        model.constraints.add((-kappa[i] - model.s[i] + model.b[i]) * step_size <= 1)
        model.constraints.add((-kappa[i] - model.s[i] + 2 * model.b[i]) * step_size <= 1)

    # if we have number of bifurcations as contraings
    if n_bif:        
        
        # define 2 slack variables for eventual constraints of mean and variance of bifurcations
        model.q = Var(range(2), domain=Reals)

        # constraint the average number of bifurcations
        if n_bif[0]:
            # add the slack variables to the objective function
            model.obj.expr += kappa_Penalty_Slack_0 * abs(model.q[0]) ** 2
            #f =
            model.constraints.add(sum((Z[i + 1] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 ) - Z[i]) / (kappa[i] + model.s[i]) * model.b[i] for i in range(kappa.size)) + model.q[0] == n_bif[0])
        
        # constrain the variance for the number of bifurcations
        if n_bif[1]:
            model.obj.expr += kappa_Penalty_Slack_1 * abs(model.q[1]) ** 2
            
            var_terms = [ ]
            for i in range(kappa.size):
                var_terms.append(model.b[i] ** 3 * 2 * Z[i] * ((Z[i + 1] / Z[i] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 )) ** 2 - \
                                                               (kappa[i] + model.s[i]) * dx * (Z[i + 1] / Z[i] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 )) - 1) / ((kappa[i] + model.s[i]) ** 3) + \
                                 model.b[i] ** 2 * (Z[i] * ((Z[i + 1] / Z[i] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 )) ** 2 - \
                                                            (kappa[i] + model.s[i]) * dx * (Z[i + 1] / Z[i] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 )) - 1) / ((kappa[i] + model.s[i]) ** 2) + \
                                                    V[i] * ((Z[i + 1] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 )- Z[i]) / (Z[i] * (kappa[i] + model.s[i]))) ** 2) + \
                                 model.b[i] * (Z[i + 1] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 )- Z[i]) / (kappa[i] + model.s[i]))

            covar_terms = []
            for i in range(kappa.size - 1):
                term = 0
                for j in range(i + 1, kappa.size):
                    term += model.b[j] * (Z[j + 1] * (1 + dx * model.s[j]  + 1 / 2 * dx ** 2 * model.s[j] ** 2 + 1 / 6 * dx ** 3 * model.s[j] ** 3 + 1 / 24 * dx ** 4 * model.s[j] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[j] ** 5) - Z[j]) / Z[j] / (kappa[j] + model.s[j]) * \
                            Z[j + 1] / Z[i] * prod((1 + dx * model.s[j]  + 1 / 2 * dx ** 2 * model.s[j] ** 2 + 1 / 6 * dx ** 3 * model.s[j] ** 3 + 1 / 24 * dx ** 4 * model.s[j] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[j] ** 5) for k in range(i + 1, j + 1))
                covar_terms.append(2 * (model.b[i] ** 3 * 2 * Z[i] * ((Z[i + 1] / Z[i] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 )) ** 2 - \
                                                               (kappa[i] + model.s[i]) * dx * (Z[i + 1] / Z[i] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 )) - 1) / ((kappa[i] + model.s[i]) ** 3) + \
                                 model.b[i] ** 2 * (Z[i] * ((Z[i + 1] / Z[i] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 )) ** 2 - \
                                                            (kappa[i] + model.s[i]) * dx * (Z[i + 1] / Z[i] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 )) - 1) / ((kappa[i] + model.s[i]) ** 2) + \
                                                    V[i] * ((Z[i + 1] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 )- Z[i]) / (Z[i] * (kappa[i] + model.s[i]))) ** 2) + \
                                 model.b[i] * (Z[i + 1] * (1 + dx * model.s[i]  + 1 / 2 * dx ** 2 * model.s[i] ** 2 + 1 / 6 * dx ** 3 * model.s[i] ** 3 + 1 / 24 * dx ** 4 * model.s[i] ** 4 + 1 / (24 * 5) * dx ** 5 * model.s[i] ** 5 + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 )- Z[i]) / (kappa[i] + model.s[i])) * model.b[i] * (Z[i + 1] * (   + 1 / (24 * 5 * 6) * dx ** 6 * model.s[i] ** 6 ) - Z[i]) / Z[i] / (kappa[i] + model.s[i]) * term )
                
            model.constraints.add(sum(var_terms) + sum(covar_terms) + model.q[1]  == n_bif[1]) 

        
    # Solve
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = max_iter
    solver.solve(model, tee=False)

##    # Extract bifurcation rates as array
    kappa_sol = np.array([value(model.s[i]) + kappa[i] for i in model.s])
    s = np.array([value(model.s[i]) for i in model.s])
    b = np.array([value(model.b[i]) for i in model.b])

##    b[b < 0] = 0.
##
##    # calculate annihilation rates
##    a = - kappa_sol + b
##    a[a < 0] = 0.

    #b# = kappa_sol[kappa_sol > 0]
    #a = -kappa_sol[kappa_sol < 0]

##    import matplotlib.pyplot as plt
##    Z1 = Z[:-1] * np.exp(kappa_sol * dx)
##    print(b.shape, s.shape, kappa.shape)
##    V1 = (2 * b / (kappa + s) - 1) * (1 + dx * s + dx ** 2 * np.power(s, 2)) * Z[1:] / Z[:-1] * (Z[1:] * (1 + dx * s + dx ** 2 * np.power(s, 2)) - Z[:-1]) + V[:-1] * np.power(Z[1:] / Z[:-1], 2) * np.power(1 + dx * s + dx ** 2 * np.power(s, 2), 2) 
##    plt.errorbar(np.arange(0, Z.size), Z, yerr=V, label='exp')
##    plt.errorbar(np.arange(0, Z1.size) + 1 + 0.1, Z1, yerr=V1, label='sim', alpha=0.5)
##    plt.legend()
##    plt.show()

    a = -kappa_sol + b
    a[a < 0] = 0
    b[b < 0] = 0
##    print(b)
##    print(a)
    return b, a
