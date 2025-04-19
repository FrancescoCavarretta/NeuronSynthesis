import random
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import model

def sign(x):
    """ return the sign of the variable """
    return int(x / abs(x))


import numpy as np
from scipy import stats
import numpy as np


def one_sample_tests(data, mean_h0=None, var_h0=None, alternative='two-sided'):
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
    data = np.asarray(data)
    n = len(data)
    results = {}

    if mean_h0 is not None:
        t_stat, p_val = stats.ttest_1samp(data, popmean=mean_h0, alternative=alternative)
        obs_mean = np.mean(data)
        results['mean'] = {
            'observed': obs_mean,
            'test_statistic': t_stat,
            'p_value': p_val
        }

    if var_h0 is not None:
        sample_var = np.var(data, ddof=1)
        chi2_stat = (n - 1) * sample_var / var_h0

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

        results['variance'] = {
            'observed': sample_var,
            'test_statistic': chi2_stat,
            'p_value': p_val
        }

    if not results:
        raise ValueError("Provide at least one of mean_h0 or var_h0.")

    return results


def bootstrap_test_1sample(data, hypothesized_value, 
                           statistic='mean', 
                           n_iterations=10000, 
                           resample_size=None,
                           seed=None):
    """
    One-sample bootstrap hypothesis test with customizable resample size.

    Parameters:
    - data: Array or list of numeric values.
    - hypothesized_value: The null hypothesis value for the statistic.
    - statistic: 'mean', 'median', 'var', or a custom function.
    - n_iterations: Number of bootstrap iterations.
    - resample_size: Size of each resample (default: same as len(data)).
    - seed: Random seed for reproducibility.

    Returns:
    - obs_stat: Observed statistic.
    - p_value: Two-tailed bootstrap p-value.
    """
    if seed is not None:
        np.random.seed(seed)

    data = np.array(data)
    n = len(data)
    if resample_size is None:
        resample_size = n
    else:
        assert resample_size < n

    # Define statistic function
    if statistic == 'mean':
        stat_func = np.mean
    elif statistic == 'median':
        stat_func = np.median
    elif statistic == 'var':
        stat_func = lambda x: np.var(x, ddof=1)
    elif callable(statistic):
        stat_func = statistic
    else:
        raise ValueError("Statistic must be 'mean', 'median', 'var', or a callable function.")

    # Observed statistic
    obs_stat = stat_func(data)

    # Bootstrap sampling
    boot_stats = []
    for _ in range(n_iterations):
        boot_stats.append(stat_func(np.random.choice(data, size=resample_size, replace=True)))

    boot_stats = np.array(boot_stats)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(boot_stats - hypothesized_value) >= np.abs(obs_stat - hypothesized_value))

    return obs_stat, p_value


def random_walk_1d(steps: int, step_size: float=0.5, prob_fugal: float = 0.5, seed: int = 0, rate_bifurcation: float = 0, rate_annihilation: float = 0, bin_size=None):
    """
    Perform a 1D random walk and return a Sholl Plot

    Parameters:
        steps (int): Number of steps to simulate the walk.
        step_size (flat): Size of each step (default is 0.5)
        prob_fugal (float): Probability of stepping to the right (default is 0.5 for a symmetric random walk).
        seed (int): Random seed for reproducibility.
        rate_bifurcation (float): Bifurcation rate (default is 0, i.e., no bifurcation).
        rate_annihilation (float): Annihilation rate (default is 0, i.e., no annihilation).
        bin_size (float): Spatial interval size of the returned Sholl Plot (default is None, i.e., bin size and step size are the same)

    Returns:
        (x_visit, y_visit): Two lists showing the visit count for each spatial location
        num_bifurcations: number of bifurcations
    """

    prob_annihilation = rate_annihilation * step_size
    prob_bifurcation = rate_bifurcation * step_size

    # check correctness of values
    assert 0 <= prob_fugal <= 1
    assert 0 <= prob_bifurcation <= 1
    assert 0 <= prob_annihilation <= 1
    assert 0 <= prob_bifurcation + prob_annihilation <= 1
    assert step_size > 0
    
    
    np.random.seed(seed)
    visit_counts = defaultdict(float)   # this vector count the visit as a function of the distance from soma
    
    walkers = [0]
    visit_counts[0] += 1
    num_bifurcations = 0
    
    for _ in range(steps):
        if len(walkers) == 0:
            break
        
        new_walkers = []
        for i in range(len(walkers)):
            X = np.random.rand()

            # choose between elongation, annihilation, bifurcation
            if X < prob_annihilation:
                n_branches = 0
            elif X < (prob_annihilation + prob_bifurcation):
                n_branches = 2
                num_bifurcations += 1 # count the bifurcations
            else:
                n_branches = 1

            # walk
            position = walkers[i]
            for _ in range(n_branches):
                # pick up a step forward or backward
                # if it start on the origin, the probability is the same
                # otherwise it is biased toward the somatofugal direction
                # if the particle is on negative coordinates the sign need to be inverted
                if position == 0:
                    step = np.random.choice([-step_size, step_size], p=[0.5] * 2)
                else:
                    step = np.random.choice([-step_size, step_size], p=[1 - prob_fugal, prob_fugal]) * sign(position)

                new_position = position + step
                
                visit_counts[abs(new_position)] += 1
                new_walkers.append(new_position)
                
        walkers = new_walkers

    # make a visit count at different intervals than step size
    # yielding a Sholl Plot
    visit_counts = np.array(sorted(visit_counts.items())).T
    if bin_size and bin_size > 0:
        max_distance = steps * step_size
        xp = np.arange(0, max_distance + bin_size, bin_size)
        yp = np.interp(xp, visit_counts[0, :], visit_counts[1, :])
        yp[xp > max_distance] = 0
        return (xp.tolist(), yp.tolist()), num_bifurcations
    else:
        return (visit_counts[0, :].tolist(), visit_counts[1, :].tolist()), num_bifurcations


def run_multiple_trials(n_trials: int, steps: int, step_size: float=0.5, prob_fugal: float = 0.5, rate_bifurcation: float = 0, rate_annihilation: float = 0, bin_size: float=None, base_seed: int = 42):
    """
    Run multiple trials of the random walk and store each path.

    Parameters:
        n_trials (int): Number of independent random walk trials.
        steps (int): Number of steps in each walk.
        prob_fugal (float): Probability of moving right.
        base_seed (int): Seed for reproducibility.

    Returns:
        list of lists: Each inner list is a position trace of one walk.
    """
    all_walks = []
    num_bifurcations = []
    for trial in range(n_trials):
        seed = base_seed + trial  # unique seed per trial
        sp, _num_bifurcations = random_walk_1d(steps, step_size, prob_fugal, seed, rate_bifurcation, rate_annihilation, bin_size)
        all_walks.append(sp)
        num_bifurcations.append(_num_bifurcations)
    return all_walks, num_bifurcations



# Example usage:
if __name__ == "__main__":
    maximum_distance = 150.0
    prob_fugal = 1  # Probability of moving right
    rate_bifurcation = 0.03
    rate_annihilation = 0.01
    n_trials = 200
    step_size = 0.5
    steps = int(round(maximum_distance / step_size))  # Number of steps in the random walk
    all_walks, num_bifurcations = run_multiple_trials(n_trials, steps, step_size=step_size,
                                                      prob_fugal=prob_fugal, rate_bifurcation=rate_bifurcation, rate_annihilation=rate_annihilation, bin_size=50, base_seed=1000)

    distance = np.arange(0, int(maximum_distance / 50) + 1) * 50
    all_visits = np.array([ yp + ([0.] * (distance.size - len(yp))) for _, yp in all_walks ])
    visits_mean_sim = np.mean(all_visits, axis=0)
    visits_std_sim = np.std(all_visits, axis=0)
    
    bif_mean_sim = np.mean(num_bifurcations)
    bif_std_sim = np.std(num_bifurcations)

    visits_mean_hyp = np.array([ model.mean_Z(x, 1, rate_bifurcation, rate_annihilation) for x in distance ])
    visits_std_hyp = np.array([ np.sqrt(model.var_Z(x, 1, 0, rate_bifurcation, rate_annihilation)) for x in distance ])

    bif_mean_hyp = model.mean_B(steps * step_size, 1, rate_bifurcation, rate_annihilation)
    bif_std_hyp = np.sqrt(model.var_B(steps * step_size, 1, 0, rate_bifurcation, rate_annihilation))

    # print bifurcations
    print('# sim. bifurcations mean and std:\t%.1f\t%.1f' % (bif_mean_sim, bif_std_sim))
    print('# modeled bifurcations mean and std:\t%.1f\t%.1f' % (bif_mean_hyp, bif_std_hyp))

    print('comparison of bifurcations (sim vs hyp):')
    print(one_sample_tests(num_bifurcations, mean_h0=bif_mean_hyp, var_h0=bif_std_hyp ** 2))
    
    print('using boostrap')
    print('\tcomparison of bifurcation mean (sim vs hyp):\tstatistic=%.1f,\tp=%.3f' % bootstrap_test_1sample(num_bifurcations, bif_mean_hyp, resample_size=150, seed=48))
    
    print('\tcomparison of bifurcation var (sim vs hyp):\tstatistic=%.1f,\tp=%.3f' % bootstrap_test_1sample(num_bifurcations, bif_std_hyp ** 2, statistic='var', resample_size=150, seed=84))
    print()

    # compare sholl plots using bootstrap
    for i, x in enumerate(distance):
        print('distance from soma %d' % x)
        print(one_sample_tests(all_visits[:, i], mean_h0=visits_mean_hyp[i], var_h0=visits_std_hyp[i] ** 2))
        print('\tcomparison of bifurcation mean (sim vs hyp):\tstatistic=%.1f,\tp=%.3f' % bootstrap_test_1sample(all_visits[:, i], visits_mean_hyp[i], resample_size=150, seed=48 + i))
        
        print('\tcomparison of bifurcation var (sim vs hyp):\tstatistic=%.1f,\tp=%.3f' % bootstrap_test_1sample(all_visits[:, i], visits_std_hyp[i] ** 2, statistic='var', resample_size=150, seed=84 + i))
        print('\n')
        
    # Plotting the result
    plt.plot(distance, visits_mean_sim, color='blue')
    plt.scatter(distance, visits_mean_sim, color='blue')
    plt.fill_between(distance, visits_mean_sim - visits_std_sim, visits_mean_sim + visits_std_sim, alpha=0.1, color='blue')
    
    plt.plot(distance, visits_mean_hyp, color='darkgray')
    plt.scatter(distance, visits_mean_hyp, color='darkgray')
    plt.fill_between(distance, visits_mean_hyp - visits_std_hyp, visits_mean_hyp + visits_std_hyp, alpha=0.1, color='darkgray')
    
    #plt.errorbar(distance, model_mean, yerr=model_std, fmt='o', capsize=5, color='black')
    
    plt.xlabel('Distance')
    plt.ylabel('Visits')
    plt.title('1D Random Walk')
    plt.grid(True)
    plt.show()
