import random
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import model
from scipy import stats
import rates

import pandas as pd

from scipy.stats import chi2



def read_distance_csv_no_header(filepath):
    """
    Reads a CSV file without a header, assuming columns are:
    distance, mean, SD (in that order).

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns ['distance', 'avg', 'SD'].
    """
    df = pd.read_csv(filepath, header=None, names=['distance', 'avg', 'SD'])
    print(df.shape)
    return df.astype(float)


def sign(x):
    """ return the sign of the variable """
    return int(x / abs(x))


def positive_normal_sample(mean, std):
    while True:
        sample = np.random.normal(loc=mean, scale=std)
        if sample > 0:
            return sample


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


def random_walk_1d(bin_size, rate_bifurcation: list, rate_annihilation: list, step_size: float=0.5, prob_fugal: float = 0.5, seed: int = 0, init_num = 1):
    """
    Perform a 1D random walk and return a Sholl Plot

    Parameters:
        bin_size (float): Spatial interval size of the returned Sholl Plot.
        rate_bifurcation (list): Bifurcation rates.
        rate_annihilation (list): Annihilation rates.
        steps (int): Number of steps to simulate the walk.
        step_size (flat): Size of each step (default is 0.5)
        prob_fugal (float): Probability of stepping to the right (default is 0.5 for a symmetric random walk).
        seed (int): Random seed for reproducibility.
        init_num (int): Initial number of dendrites (default is 1, i.e.); it can be also a tuple (mean, SD)

    Returns:
        (x_visit, y_visit): Two lists showing the visit count for each spatial location
        num_bifurcations: number of bifurcations
    """

    prob_annihilation = np.array(rate_annihilation) * step_size
    prob_bifurcation = np.array(rate_bifurcation) * step_size
   
    

    # check correctness of values
    assert 0 <= prob_fugal <= 1
    assert prob_bifurcation.size == prob_annihilation.size
    assert ((0 <= prob_bifurcation) & (prob_bifurcation <= 1)).all()
    assert ((0 <= prob_annihilation) & (prob_annihilation <= 1)).all()
    assert ((0 <= prob_bifurcation + prob_annihilation) & (prob_bifurcation + prob_annihilation <= 1)).all()
    assert step_size > 0
    assert bin_size > step_size
    
    
    np.random.seed(seed)
    visit_counts = defaultdict(float)   # this vector count the visit as a function of the distance from soma

    # if the init_num is a tuple, extract a random number
    if type(init_num) in [tuple, list]:
        init_num = int(round(positive_normal_sample(*init_num)))
        
    walkers = [0] * init_num
    visit_counts[0] += init_num
    num_bifurcations = 0
    
    for _ in range(int(round(bin_size * prob_bifurcation.size / step_size))):
        if len(walkers) == 0:
            break
        
        new_walkers = []
        for i in range(len(walkers)):
            # walk
            position = walkers[i]

            # bin_index
            bin_index = int(abs(position) / bin_size)
            if bin_index >= prob_annihilation.size:
                _prob_annihilation = 1
                _prob_bifurcation = 0
            else:
                _prob_annihilation = prob_annihilation[bin_index]
                _prob_bifurcation = prob_bifurcation[bin_index]


            # generate a random number
            X = np.random.rand()

            # choose between elongation, annihilation, bifurcation
            if X < _prob_annihilation:
                n_branches = 0
            elif X < (_prob_annihilation + _prob_bifurcation):
                n_branches = 2
                num_bifurcations += 1 # count the bifurcations
            else:
                n_branches = 1


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


def run_multiple_trials(n_trials: int, bin_size: float, rate_bifurcation: list, rate_annihilation: list, step_size: float=0.5, prob_fugal: float = 0.5, base_seed: int = 42, init_num = 1):
    """
    Run multiple trials of the random walk and store each path.

    Parameters:
        n_trials (int): Number of independent random walk trials.
        prob_fugal (float): Probability of moving right.
        base_seed (int): Seed for reproducibility.

    Returns:
        list of lists: Each inner list is a position trace of one walk.
    """
    all_walks = []
    num_bifurcations = []
    for trial in range(n_trials):
        seed = base_seed + trial  # unique seed per trial
        sp, _num_bifurcations = random_walk_1d(bin_size, rate_bifurcation, rate_annihilation, step_size, prob_fugal, seed, init_num)
        all_walks.append(sp)
        num_bifurcations.append(_num_bifurcations)
    return all_walks, num_bifurcations



# Example usage:
if __name__ == "__main__":
    
    import sys

    filepath = {'SP':'sp_apical_sholl_plot.txt', 'SL':'sl_apical_sholl_plot.txt'}[sys.argv[-1]]
    exp_data = read_distance_csv_no_header(filepath)

    # assess equal size of intervals
    dx = np.unique(np.diff(exp_data.distance))
    assert dx.size == 1
    dx = dx[0]

    rate_bifurcation, rate_annihilation = rates.solve_qp(dx, exp_data.avg.to_numpy(), exp_data.SD.to_numpy())

    maximum_distance = len(rate_annihilation) * 50.0
    prob_fugal = 1  # Probability of moving right
    ##    print('rate_bifurcation:', rate_bifurcation, '\trate_annihilation', rate_annihilation)
    n_trials = 300
    step_size = 0.5
    init_num = exp_data.loc[0, ['avg', 'SD']].tolist()
    ##    print(init_num)
    ##    resample_size = 10
    steps = int(round(maximum_distance / step_size))  # Number of steps in the random walk
    all_walks, num_bifurcations = run_multiple_trials(n_trials, 50, rate_bifurcation, rate_annihilation, step_size=step_size,
                                                      prob_fugal=prob_fugal, base_seed=1000, init_num=init_num)

    distance = np.arange(0, int(maximum_distance / 50) + 1) * 50
    all_visits = np.array([ yp + ([0.] * (distance.size - len(yp))) for _, yp in all_walks ])
    visits_mean_sim = np.mean(all_visits, axis=0)
    visits_std_sim = np.std(all_visits, axis=0)
    
    bif_mean_sim = np.mean(num_bifurcations)
    bif_std_sim = np.std(num_bifurcations)

    #print(distance), print(all_visits)
    if type(init_num) == list:
        visits_mean_hyp = [init_num[0]]
        visits_std_hyp = [init_num[1]]
    else:
        visits_mean_hyp = [init_num]
        visits_std_hyp = [0]
        
    for i, x in enumerate(distance):
        if i < 1:
            continue            
        visits_mean_hyp.append(model.mean_Z(dx, visits_mean_hyp[-1], rate_bifurcation[i-1], rate_annihilation[i-1]) )
        visits_std_hyp.append(np.sqrt(model.var_Z(dx, visits_mean_hyp[-1], visits_std_hyp[-1] ** 2, rate_bifurcation[i-1], rate_annihilation[i-1])))
        print(visits_mean_hyp[-2], exp_data.loc[i-1, 'avg'], exp_data.loc[i-1, 'SD'])
        print(visits_mean_hyp[-1], exp_data.loc[i, 'avg'], exp_data.loc[i, 'SD'])
        print()
    visits_mean_hyp = np.array(visits_mean_hyp)
    visits_std_hyp = np.array(visits_std_hyp)

    bif_mean_hyp = model.mean_B(steps * step_size, init_num[0], rate_bifurcation[0], rate_annihilation[0])
    bif_std_hyp = np.sqrt(model.var_B(steps * step_size, init_num[0], init_num[1], rate_bifurcation[0], rate_annihilation[0]))

    # print bifurcations
    print('# sim. bifurcations mean and std:\t%.1f\t%.1f' % (bif_mean_sim, bif_std_sim))
    print('# modeled bifurcations mean and std:\t%.1f\t%.1f' % (bif_mean_hyp, bif_std_hyp))

    print('comparison of bifurcations (sim vs hyp):')
    print(one_sample_tests(num_bifurcations, mean_h0=bif_mean_hyp, var_h0=bif_std_hyp ** 2))
    
    #print('using boostrap')
    #print('\tcomparison of bifurcation mean (sim vs hyp):\tstatistic=%.1f,\tp=%.3f' % bootstrap_test_1sample(num_bifurcations, bif_mean_hyp, resample_size=resample_size, seed=48))
    
    #print('\tcomparison of bifurcation var (sim vs hyp):\tstatistic=%.1f,\tp=%.3f' % bootstrap_test_1sample(num_bifurcations, bif_std_hyp ** 2, statistic='var', resample_size=resample_size, seed=84))
    print()

    # compare sholl plots using bootstrap
    for i, x in enumerate(distance):
        
        print('distance from soma %d' % x)
        
        print(one_sample_tests(all_visits[:, i], mean_h0=visits_mean_hyp[i], var_h0=visits_std_hyp[i] ** 2))
        #print('\tcomparison of bifurcation mean (sim vs hyp):\tstatistic=%.1f,\tp=%.3f' % bootstrap_test_1sample(all_visits[:, i], visits_mean_hyp[i], resample_size=resample_size, seed=48 + i))
        
        #print('\tcomparison of bifurcation var (sim vs hyp):\tstatistic=%.1f,\tp=%.3f' % bootstrap_test_1sample(all_visits[:, i], visits_std_hyp[i] ** 2, statistic='var', resample_size=resample_size, seed=84 + i))
        print('\n')
        
    # Plotting the result
    plt.plot(distance, visits_mean_sim, color='blue')
    plt.scatter(distance, visits_mean_sim, color='blue')
    plt.fill_between(distance, visits_mean_sim - visits_std_sim, visits_mean_sim + visits_std_sim, alpha=0.1, color='blue')

    plt.plot(distance, visits_mean_hyp, color='darkgray')
    plt.scatter(distance, visits_mean_hyp, color='darkgray')
    plt.fill_between(distance, visits_mean_hyp - visits_std_hyp, visits_mean_hyp + visits_std_hyp, alpha=0.1, color='darkgray')

    plt.errorbar(exp_data.distance, exp_data.avg, yerr=exp_data.SD, fmt='o', capsize=5, color='black')

    plt.xlabel('Distance')
    plt.ylabel('Visits')
    plt.title('1D Random Walk')
    plt.grid(True)
    plt.show()
