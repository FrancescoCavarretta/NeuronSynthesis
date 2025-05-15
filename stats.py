from scipy.stats import chi2

from scipy.stats import norm, f

from scipy import stats

import numpy as np


def two_samples_tests(mean1, var1, n1, mean2, var2, n2):
    """
    Compares the means and variances of two samples.

    Parameters:
    - mean1, var1, n1: mean, variance, and size of sample 1
    - mean2, var2, n2: mean, variance, and size of sample 2
    - alpha: significance level (default is 0.05)

    Returns:
    - Dictionary with Z-test and F-test results
    """

    # --- Z-Test for Means ---
    se = ((var1 / n1) + (var2 / n2)) ** 0.5
    z_score = (mean1 - mean2) / se
    p_value_z = 2 * (1 - norm.cdf(abs(z_score)))

    # --- F-Test for Variances ---
    if var1 > var2:
        f_stat = var1 / var2
        df1, df2 = n1 - 1, n2 - 1
    else:
        f_stat = var2 / var1
        df1, df2 = n2 - 1, n1 - 1

    p_value_f = 2 * min(f.cdf(f_stat, df1, df2), 1 - f.cdf(f_stat, df1, df2))

    return {
        "Z Test": {
            "statistic": z_score,
            "p_value": p_value_z
        },
        "F Test": {
            "statistic": f_stat,
            "p_value": p_value_f
        }
    }


def one_sample_tests(obs_mean, obs_var, n, mean_h0=None, var_h0=None, alternative='two-sided'):
    """
    Perform one-sample t-test (for mean) and chi-square test (for variance).
    
    Parameters:
    - data: list or 1D array of numerical values.
    - mean_h0: hypothesized mean under H0 (optional).
    - var_h0: hypothesized variance under H0 (optional).
    - alternative: 'two-sided', 'less', or 'greater'.
    
    Returns:
    - Dictionary of test results with observed values, test statistics, and p-values.
    """
    results = {}

    if mean_h0 is not None:
        # Calculate t-statistic
        se = obs_var / np.sqrt(n)
        t_stat = (obs_mean - mean_h0) / se

        # Degrees of freedom
        df = n - 1

        # Two-tailed p-value
        p_val = 2 * stats.t.sf(np.abs(t_stat), df)
        
        results['T Test'] = {
            'statistic': t_stat,
            'p_value': p_val
        }

    if var_h0 is not None:
        chi2_stat = (n - 1) * obs_var / var_h0

        if alternative == 'two-sided':
            p_val = 2 * min(
                stats.chi2.cdf(chi2_stat, df=n-1),
                1 - stats.chi2.cdf(chi2_stat, df=n-1)
            )
        elif alternative == 'greater':
            p_val = 1 - stats.chi2.cdf(chi2_stat, df=n-1)
        elif alternative == 'less':
            p_val = stats.chi2.cdf(chi2_stat, df=n-1)
        else:
            raise ValueError("Alternative must be 'two-sided', 'less', or 'greater'.")

        results['Chi2 Test'] = {
            'statistic': chi2_stat,
            'p_value': p_val
        }

    if not results:
        raise ValueError("Provide at least one of mean_h0 or var_h0.")

    return results
