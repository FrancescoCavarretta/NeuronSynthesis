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
    if not isinstance(rate_bifurcation, Iterable):
        rate_bifurcation = [rate_bifurcation]
        
    if not isinstance(rate_annihilation, Iterable):
        rate_annihilation = [rate_annihilation]
        
    prob_annihilation = np.array(rate_annihilation) * step_size
    prob_bifurcation = np.array(rate_bifurcation) * step_size

    # check correctness of values
    assert 0 <= prob_fugal <= 1
    assert prob_bifurcation.size == prob_annihilation.size
    assert ((0 <= prob_bifurcation) & (prob_bifurcation <= 1)).all()
    assert ((0 <= prob_annihilation) & (prob_annihilation <= 1)).all()
    assert ((0 <= prob_bifurcation + prob_annihilation) & (prob_bifurcation + prob_annihilation <= 1)).all()
    assert step_size > 0        
    assert not bin_size or bin_size and bin_size >= step_size
    
    
    np.random.seed(seed)
    visit_counts = defaultdict(float)   # this vector count the visit as a function of the distance from soma

    # if the init_num is a tuple, extract a random number
    if type(init_num) in [tuple, list]:
        init_num = int(round(positive_normal_sample(*init_num)))
        
    walkers = [0] * init_num
    visit_counts[0] += init_num
    num_bifurcations = [0]

    if max_distance is None:
        max_distance = bin_size * prob_bifurcation.size

    
    for _ in range(int(round(max_distance / step_size))):

        if len(walkers) == 0:
            break
        
        num_bifurcations.append(num_bifurcations[-1])

        new_walkers = []
        for i in range(len(walkers)):
            # walk
            position = walkers[i] 
            
            # bin_index
            if bin_size:
                bin_index = int(abs(position) / bin_size)
                stop_cond = bin_index >= prob_annihilation.size
            else:
                bin_index = 0
                stop_cond = abs(position) >= max_distance
                
            if stop_cond:
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
                num_bifurcations[-1] += 1 # count the bifurcations
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
    if bin_size_interp is None:
        bin_size_interp = bin_size
    if bin_size_interp and bin_size_interp > 0:
        #import matplotlib.pyplot as plt
        #plt.plot(visit_counts[0, :], visit_counts[1, :])
        xp = np.arange(0, max_distance + bin_size_interp, bin_size_interp)
        yp = np.interp(xp, visit_counts[0, :], visit_counts[1, :])
        yp[xp > max_distance] = 0
        xb = np.arange(0, len(num_bifurcations)) * step_size
        zp = np.interp(xp, xb, num_bifurcations)
        return np.concatenate((xp.reshape(-1, 1), yp.reshape(-1, 1)), axis=1), np.concatenate((xp.reshape(-1, 1), zp.reshape(-1, 1)), axis=1)
    else:
        return (visit_counts[0, :].tolist(), visit_counts[1, :].tolist()), num_bifurcations


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


