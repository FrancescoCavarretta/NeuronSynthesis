from pyomo.environ import *
import numpy as np


def mk_objective(model, kappa, Z, V):
    terms = []
    
    # B & A
    tmp = Z[1:] * (Z[1:] - Z[:-1]) / Z[:-1]
    
    A = 2 * tmp / kappa
    B = tmp + V[:-1] * np.power(Z[1:] / Z[:-1],  2)
        
    return sum(model.b[i] ** 2 * (A[i] ** 2) + model.b[i] * (2 * A[i] * (B[i] - V[i + 1])) for i in range(kappa.size))

def getA_term(model, kappa, Z, V, m, i):
    return model.b[m] * 2 * Z[i] ** 2 * (Z[m + 1] - Z[m]) / (Z[m + 1] * Z[m]) / kappa[m]

def getB_term(kappa, Z, V, m):
    return (Z[m + 1] - Z[m]) / (Z[m + 1] * Z[m])
    
def mk_objective(model, kappa, Z, V):
    terms = []
    for i in range(1, kappa.size + 1):
        B = (V[0] - sum(getB_term(kappa, Z, V, m) for m in range(0, i))) * (Z[i] ** 2)
        terms += [2 * getA_term(model, kappa, Z, V, m, i) * (B - V[i]) for m in range(0, i) ]              
        terms += [getA_term(model, kappa, Z, V, m, i) * getA_term(model, kappa, Z, V, n, i) for m in range(0, i) for n in range(0, i)]
    return sum(terms)

def solve_qp(step_size, dx, Z, V, n_bif=None, max_iter=10000, kappa_Penalty_Var=1.0):
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

    # define 1 slack variables for eventual constraints of variance of bifurcations
    model.s = Var(domain=Reals)
            
    # Constraint: 
    model.constraints = ConstraintList()
    for i in range(kappa.size):
        model.constraints.add(model.b[i] >= kappa[i])
        model.constraints.add((2 * model.b[i] - kappa[i]) * step_size <= 1)

    # if we have number of bifurcations as contraings
    if n_bif:        
        # constraint the average number of bifurcations
        if n_bif[0]:
            f = (Z[1:] - Z[:-1]) / kappa
            model.constraints.add(sum(f[i] * model.b[i] for i in range(kappa.size)) == n_bif[0])
        
        # constrain the variance for the number of bifurcations
        if n_bif[1]:            
            f1 = (Z[1:] - Z[:-1]) / kappa
            f2 = - Z[:-1] * (np.power(Z[1:] / Z[:-1], 2) - 2 * kappa * dx * Z[1:] / Z[:-1] - 1) / kappa ** 2 + V[:-1] * np.power((Z[1:] - Z[:-1]) / Z[:-1] / kappa, 2)
            f3 = 2 * Z[:-1] * (np.power(Z[1:] / Z[:-1], 2) - 2 * kappa * dx * Z[1:] / Z[:-1] - 1) / kappa ** 3
            
            var_terms = [ ]
            for i in range(kappa.size):
                var_terms.append(f1[i] * model.b[i] + f2[i] * model.b[i] ** 2 + f3[i] * model.b[i] ** 3)

            for i in range(1, kappa.size):
                for j in range(i + 1, kappa.size + 1):
                    term1 = model.b[i - 1] * (Z[i] - Z[i - 1])/  Z[i - 1] / kappa[i - 1] 
                    term2 = model.b[j - 1] * (Z[j] - Z[j - 1]) / Z[j - 1] / kappa[j - 1]
                    for k in range(1, i + 1):
                        if k == 1:
                            Vprev = V[0]
                        term3 = (2 * model.b[k - 1] - kappa[k - 1]) / kappa[k - 1] * Z[k] * (Z[k] - Z[k - 1]) / Z[k - 1] + Vprev * np.power(Z[k] / Z[k - 1], 2)
                        var_terms.append(2 * term1 * term2 * Z[j - 1] / Z[i - 1] * term3)
                        Vprev = term3
                
            model.constraints.add(sum(var_terms) + model.s == n_bif[1]) 

    # Objective
    model.obj = Objective(
        expr=mk_objective(model, kappa, Z, V) + kappa_Penalty_Var * model.s ** 2,
        sense=minimize
    )

    
    # Solve
    solver = SolverFactory('ipopt')
    #solver.options['max_iter'] = max_iter
    solver.solve(model, tee=False)
    print("Objective value:", value(model.obj))
    print(value(model.s))
    
    # Extract bifurcation rates as array
    b = np.array([value(model.b[i]) for i in model.b])
    b[b < 0] = 0.

    # calculate annihilation rates
    a = - kappa + b
    a[a < 0] = 0.
    return b, a
