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

    if kappa != 0:
        exp_kx = np.exp(kappa * x)
        term1 = E_z0 * (2 * beta - kappa) * exp_kx * (exp_kx - 1) / kappa
        term2 = Var_z0 * np.exp(2 * kappa * x)
        return term1 + term2
    else:
        return 2 * E_z0 * beta * x + Var_z0


def cov_Z(x, E_z0, Var_z0, beta, alpha):
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

    return Var_z0 * np.exp(kappa * x)

    
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

    if abs(kappa) > 1e-5:
        exp_kappa = np.exp(kappa * x)
        exp_2kappa = np.exp(2 * kappa * x)

        #term1 = beta * E_z0 * (exp_kappa - 1) / kappa
        term2 = beta**2 * (2 * beta - kappa) / kappa * E_z0 * (exp_2kappa - 2 * kappa * x * exp_kappa - 1) / kappa**2
        term3 = beta**2 * Var_z0 * (exp_2kappa - 2 * exp_kappa + 1) / kappa**2

        return term2 + term3 #term1 * 0 + term2 + term3
    else:
        return ( beta * x ) ** 2 * (2 / 3 * beta * E_z0 * x + Var_z0)
