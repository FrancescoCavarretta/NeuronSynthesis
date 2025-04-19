import numpy as np

def mean_Z(x, E_z0, beta, alpha):
    """
    Compute the mean of Z(x) using kappa = beta - alpha.

    Parameters:
        x (float or array): Position(s)
        E_z0 (float): Expected value of z0
        beta (float): Growth rate parameter
        alpha (float): Decay rate parameter

    Returns:
        float or np.ndarray: E[Z(x)]
    """
    kappa = beta - alpha
    return E_z0 * np.exp(kappa * x)

def var_Z(x, E_z0, Var_z0, beta, alpha):
    """
    Compute the variance of Z(x) using kappa = beta - alpha.

    Parameters:
        x (float or array): Position(s)
        E_z0 (float): Expected value of z0
        Var_z0 (float): Variance of z0
        beta (float): Growth rate parameter
        alpha (float): Decay rate parameter

    Returns:
        float or np.ndarray: Var[Z(x)]
    """
    kappa = beta - alpha

    if abs(kappa) > 1e-5:
        term1 = E_z0 * (2 * beta - kappa) * np.exp(kappa * x) * (np.exp(kappa * x) - 1) / kappa
        term2 = Var_z0 * np.exp(2 * kappa * x)
        return term1 + term2
    else:
        return 2 * E_z0 * beta * x + Var_z0


def mean_B(x, E_z0, beta, alpha):
    """
    Compute the mean number of bifurcations B(x).

    Parameters:
        x (float or array): Position(s)
        E_z0 (float): Expected initial number of branches z0
        beta (float): Growth rate
        alpha (float): Pruning or death rate

    Returns:
        float or np.ndarray: E[B(x)]
    """
    kappa = beta - alpha
    if abs(kappa) > 1e-5:
        return beta * E_z0 * (np.exp(kappa * x) - 1) / kappa
    else:
        return beta * E_z0 * x


def var_B(x, E_z0, Var_z0, beta, alpha):
    """
    Compute the variance of the number of bifurcations B(x).

    Parameters:
        x (float or array): Position(s)
        E_z0 (float): Expected initial number of branches z0
        Var_z0 (float): Variance of z0
        beta (float): Growth rate
        alpha (float): Pruning or death rate

    Returns:
        float or np.ndarray: Var[B(x)]
    """
    kappa = beta - alpha
    if kappa != 0:
        term1 = (beta / kappa) * (np.exp(kappa * x) - 1)
        term2 = (beta**2 * (beta + alpha) / kappa**3) * (np.exp(2 * kappa * x) - 2 * kappa * x * np.exp(kappa * x) - 1)
        return E_z0 * (term1 + term2)
    else:
        return beta**2 * x * ((E_z0**2 + 2 * Var_z0) * x + 2 * beta * E_z0 * x**2)
