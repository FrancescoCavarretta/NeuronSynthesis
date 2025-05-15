import numpy as np
from collections.abc import Iterable
from collections import defaultdict


def sign(x):
    """ return the sign of the variable """
    return int(x / abs(x))


def positive_normal_sample(mean, std):
    while True:
        sample = np.random.normal(loc=mean, scale=std)
        if sample > 0:
            return sample
        

def random_walk_1d(bin_size, rate_bifurcation, rate_annihilation, max_distance: float=None, step_size: float=0.5, prob_fugal: float = 1, seed: int = 1400, init_num = 1, bin_size_interp: float = None):
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
    # check correctness of values
    assert 0 <= prob_fugal <= 1
    assert step_size > 0

    # check whether it works on sholl plot or single rates
    sholl_plot_flag = isinstance(rate_bifurcation, Iterable) and isinstance(rate_annihilation, Iterable) and (bin_size is not None) and (prob_bifurcation.size == prob_annihilation.size)
    single_rate_flag = not (isinstance(rate_bifurcation, Iterable) or isinstance(rate_annihilation, Iterable)) and (max_distance is not None)    

    if single_rate_flag:
        n = int(round(max_distance / step_size))
    elif sholl_plot_flag:
        n = int(round(bin_size / step_size))
    else:
        raise Exception()

    # bifurcations and annihilation probabilities
    prob_bifurcation = np.concatenate((np.repeat(rate_bifurcation * step_size, n), [0])) 
    prob_annihilation = np.concatenate((np.repeat(rate_annihilation * step_size, n), [1]))

    nsteps = prob_bifurcation.size
    
    np.random.seed(seed)

    # if the init_num is a tuple, extract a random number
    if type(init_num) in [tuple, list]:
        init_num = int(round(positive_normal_sample(*init_num)))
        
    
    d_walkers = np.zeros(init_num, dtype=int)
    visit_counts = np.zeros(nsteps, dtype=int)
    visit_counts[0] += init_num
    num_bifurcations = np.zeros(nsteps, dtype=int)

    
    for i in range(1, nsteps):
        if d_walkers.size == 0:
            num_bifurcations[i:] = num_bifurcations[i - 1]
            break
        

        X = np.random.rand(d_walkers.size) # random number select to select one out of annihilation, branching, elongation          
        d_walkers[d_walkers >= prob_annihilation.size] = prob_annihilation.size - 1 # if distance exceed the size of random walks, annihilate
        pb = prob_bifurcation[d_walkers]
        pa = prob_annihilation[d_walkers] 
        idx_bif = (X >= pa) & (X < (pa + pb)) # these will branch
        idx_eln = X >= (pa + pb)            # these will elongate
        d_walkers = np.concatenate((d_walkers[idx_eln], d_walkers[idx_bif], d_walkers[idx_bif])) # new walkers
        moves = np.random.choice([-1, 1], p=[1 - prob_fugal, prob_fugal], size=d_walkers.size) # moves
        moves[d_walkers == 0] = 1
        d_walkers += moves
        num_bifurcations[i] = num_bifurcations[i - 1] + np.sum(idx_bif) # update number of bifurcations

        # update visits
        d, counts = np.unique(d_walkers, return_counts=True)

        visit_counts[d] += counts

    # make a visit count at different intervals than step size
    # yielding a Sholl Plot
    if bin_size_interp is None:
        inc = 1
        bin_size_interp = step_size
    else:
        inc = int(round(bin_size_interp / step_size))
        
    yp = visit_counts[::inc]
    zp = num_bifurcations[::inc]

    xp = np.arange(0, yp.size) * bin_size_interp

    return np.concatenate((xp.reshape(-1, 1), yp.reshape(-1, 1)), axis=1), np.concatenate((xp.reshape(-1, 1), zp.reshape(-1, 1)), axis=1)


def run_multiple_trials(bin_size: float, rate_bifurcation, rate_annihilation, max_distance: float=None, n_trials: int = 100, step_size: float=0.5, prob_fugal: float = 1, base_seed: int = 42, init_num = 1, bin_size_interp: float = None):
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
        sp, _num_bifurcations = random_walk_1d(bin_size, rate_bifurcation, rate_annihilation, max_distance=max_distance, step_size=step_size, prob_fugal=prob_fugal, seed=seed, init_num=init_num, bin_size_interp=bin_size_interp)
        all_walks.append(sp)
        num_bifurcations.append(_num_bifurcations)
    return all_walks, num_bifurcations


