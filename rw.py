import numpy as np
from collections.abc import Iterable
from collections import defaultdict

def positive_normal_sample(mean, std):
    while True:
        sample = np.random.normal(loc=mean, scale=std)
        if sample > 0:
            return sample


def _get_bin_sz(sp):
    sz = np.diff(sp[:, 0])
    sz_min = sz.min()
    assert (sz_min == sz).all()
    return sz_min


def sign(x):
    s = x / np.abs(x)
    s[x == 0] = 1
    return s.astype(int)

def random_walk_1d(rate_bifurcation, rate_annihilation, max_distance: float=None, max_time: float=None, step_size: float=0.1, prob_fugal: float = 1, seed: int = 1400, init_num = 1, bin_size_interp: float = None, store_moves=False):
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
    sholl_plot_flag = isinstance(rate_bifurcation, Iterable) and isinstance(rate_annihilation, Iterable) and (rate_bifurcation.size == rate_annihilation.size)
    single_rate_flag = not (isinstance(rate_bifurcation, Iterable) or isinstance(rate_annihilation, Iterable)) and (max_distance is not None or max_time is not None)    

    if max_distance is None:
        if max_time:
            if prob_fugal == 0.5:
                max_distance = max_time * step_size
            else:
                max_distance = max_time * step_size * (2 * prob_fugal - 1)

    if single_rate_flag:
        n = int(round(max_distance / step_size))
    elif sholl_plot_flag:
        # check bin sizes and assign one
        bin_size_bif = _get_bin_sz(rate_bifurcation)
        bin_size_ann = _get_bin_sz(rate_annihilation)
        assert bin_size_bif == bin_size_ann
        bin_size = bin_size_bif
        del bin_size_bif, bin_size_ann
        n = int(round(bin_size / step_size))
        rate_bifurcation, rate_annihilation = rate_bifurcation[:, 1], rate_annihilation[:, 1]
    else:
        raise Exception()
    
    # bifurcations and annihilation probabilities
    prob_bifurcation = np.concatenate((np.repeat(rate_bifurcation * step_size, n), [0])) 
    prob_annihilation = np.concatenate((np.repeat(rate_annihilation * step_size, n), [1]))    

    # check probabilities
    assert ((0 <= prob_bifurcation) & (prob_bifurcation <= 1)).all() and ((0 <= prob_annihilation) & (prob_annihilation <= 1)).all() and \
           ((0 <= prob_bifurcation + prob_annihilation) & (prob_bifurcation + prob_annihilation <= 1)).all()
    
    # set the random seed
    np.random.seed(seed)

    # if the init_num is a tuple, extract a random number
    if type(init_num) in [tuple, list]:
        init_num = int(round(positive_normal_sample(*init_num)))

    if max_time is None:
        nsteps = int(round(prob_bifurcation.size / (2 * prob_fugal - 1))) # number of steps
    else:
        nsteps = max_time + 1
    
    walkers = np.zeros(init_num, dtype=int)
    visit_counts = np.zeros(nsteps, dtype=int)
    visit_counts[0] = init_num
    num_bifurcations = np.zeros(nsteps, dtype=int)


    # handle branches
    if store_moves:
        branches = defaultdict(list)
        for i in range(init_num):
            branches[i] += [[0, 1]]
        idx = np.array(np.arange(0, init_num))
        
        
    for time in range(1, nsteps):

        if walkers.size == 0:
            break

        # random number select to select one out of annihilation, branching, elongation
        # using the method of anthitetic variables
        if time % 2:
            X = np.random.rand(walkers.size)
        else:
            X = 1 - X[np.concatenate((np.flatnonzero(idx_eln), np.repeat(np.flatnonzero(idx_bif), 2)))]
        
        # walkers which will branch
        idx_bif = (X >= prob_annihilation[np.abs(walkers)]) & (X < prob_bifurcation[np.abs(walkers)] + prob_annihilation[np.abs(walkers)])
        
        # walkers which will elongate
        idx_eln = X >= prob_bifurcation[np.abs(walkers)] + prob_annihilation[np.abs(walkers)]
        
        # create entries for branches
        if store_moves:
            new_idx_bif = len(branches) + np.arange(0, idx_bif.sum() * 2)
            old_idx_bif = np.repeat(idx[idx_bif], 2)
            for orig, dest in zip(old_idx_bif, new_idx_bif):
                branches[dest] += [ [time - 1, branches[orig][-1][1]] ]
            new_idx = np.concatenate((idx[idx_eln], new_idx_bif))

        # generate the new walkers
        walkers = np.concatenate((walkers[idx_eln], np.repeat(walkers[idx_bif], 2)))                
                            

        # generate steps
        # in any other location
        steps = np.random.choice([-1, 1], p=[1 - prob_fugal, prob_fugal], size=walkers.size) * sign(walkers)
        
        # correction of steps if the walker is at the origin
        walkers_at_origin = walkers == 0
        steps[walkers_at_origin] = np.random.choice([-1, 1], p=[0.5, 0.5], size=walkers_at_origin.sum()) 

        # move the walkers
        walkers += steps 

        if store_moves:
            # update indices
            idx = new_idx
            
            # extend
            for i, dest in enumerate(new_idx):
                branches[dest] += [ [time, walkers[i]] ]
                
        # update visits
        d, counts = np.unique(np.abs(walkers), return_counts=True)        
        visit_counts[d] += counts
        
        # count bifurcations
        num_bifurcations[time] = idx_bif.sum()

    # comulate the bifurcations
    num_bifurcations = np.cumsum(num_bifurcations)
    
    # make a visit count at different intervals than step size
    # yielding a Sholl Plot
    if bin_size_interp is None:
        inc = 1
        bin_size_interp = step_size
    else:
        assert bin_size_interp > 0
        inc = int(round(bin_size_interp / step_size))
        
    yp = visit_counts[::inc]
    zp = num_bifurcations[::inc]
    xp = np.arange(0, yp.size) * bin_size_interp

    if store_moves:
        return np.concatenate((xp.reshape(-1, 1), yp.reshape(-1, 1)), axis=1), np.concatenate((xp.reshape(-1, 1), zp.reshape(-1, 1)), axis=1), branches
    else:
        return np.concatenate((xp.reshape(-1, 1), yp.reshape(-1, 1)), axis=1), np.concatenate((xp.reshape(-1, 1), zp.reshape(-1, 1)), axis=1)



def run_multiple_trials(rate_bifurcation, rate_annihilation, max_distance: float=None, max_time: float=None, n_trials: int = 100, step_size: float=0.1, prob_fugal: float = 1, base_seed: int = 42, init_num = 1, bin_size_interp: float = None):
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
    all_bif = []
    for trial in range(n_trials):
        seed = base_seed + trial  # unique seed per trial
        sp, bif = random_walk_1d(rate_bifurcation, rate_annihilation, max_distance=max_distance, max_time=max_time, step_size=step_size, prob_fugal=prob_fugal, seed=seed, init_num=init_num, bin_size_interp=bin_size_interp)
        all_walks.append(sp)
        all_bif.append(bif)
    return np.array(all_walks), np.array(all_bif)




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.0, 3.5))

    plt.rcParams['font.size'] = 10
    plt.rcParams['font.weight'] = 'bold'  # NEW
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'    

    step_size = 0.5
    rb = 0.005
    ra = rb / 5
    sh1 = 0.01
    sh2 = 0.003
    max_time = 1000
    init_num = 20
    prob_fugal1 = 0.5
    prob_fugal2 = 0.75
    
    branches = random_walk_1d(rb, ra, max_time=max_time, prob_fugal = prob_fugal1, step_size = step_size, store_moves=True, init_num = init_num)[-1] # test
    flag = True
    for b in branches.values():
        b = np.array(b)
        plt.plot(b[:, 0], np.abs(b[:, 1]) * step_size, color='gray', label='RW' if flag else None, linewidth=0.25)
        flag = False
        

    branches = random_walk_1d(rb, ra, max_time=max_time, prob_fugal = prob_fugal2, step_size = step_size, store_moves=True, init_num = init_num)[-1] # test
    flag = True
    for b in branches.values():
        b = np.array(b)
        plt.plot(b[:, 0], np.abs(b[:, 1]) * step_size, color='black', label='BRW' if flag else None, linewidth=0.5)
        flag = False
        
    branches = random_walk_1d(rb + sh1, ra + sh1, max_time=max_time, prob_fugal = prob_fugal2, step_size = step_size, store_moves=True, init_num = init_num)[-1] # test
    flag = True
    for b in branches.values():
        b = np.array(b)
        plt.plot(b[:, 0], np.abs(b[:, 1]) * step_size, color='red', label=r'BRW, sh. $\beta$ and $\alpha$' if flag else None, alpha=0.8, linewidth=0.25)
        flag = False

    branches = random_walk_1d(rb + sh2, ra, max_time=max_time, prob_fugal = prob_fugal2, step_size = step_size, store_moves=True, init_num = init_num)[-1] # test
    flag = True
    for b in branches.values():
        b = np.array(b)
        plt.plot(b[:, 0], np.abs(b[:, 1]) * step_size, color='blue', label=r'BRW, inc.  $\beta$' if flag else None, alpha=0.5, linewidth=0.25)
        flag = False
        
    plt.xlabel('t')
    plt.ylabel('x')
    
    plt.ylim([0, 250])
    plt.xlim([0, max_time])
    plt.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig('fig_1_a.png', dpi=300)
    plt.show()
    
    
    
