import random
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def sign(x):
    return int(x / abs(x))

def random_walk_1d(steps: int, step_size: float=0.5, prob_right: float = 0.5, seed: int = 42, rate_bifurcation: float = 0, rate_annihilation: float = 0, bin_size=None):
    """
    Perform a 1D random walk.

    Parameters:
        steps (int): Number of steps to simulate the walk.
        prob_right (float): Probability of stepping to the right (default is 0.5 for a symmetric random walk).
        seed (int): Random seed for reproducibility.

    Returns:
        list: A list of positions at each step.
    """

    prob_annihilation = rate_annihilation * step_size
    prob_bifurcation = rate_bifurcation * step_size

    # check correctness of values
    assert 0 <= prob_right <= 1
    assert 0 <= prob_bifurcation <= 1
    assert 0 <= prob_annihilation <= 1
    assert 0 <= prob_bifurcation + prob_annihilation <= 1
    assert step_size > 0
    
    
    random.seed(seed)
    visit_counts = defaultdict(float)
    
    walkers = [0]
    visit_counts[0] += 1
    
    for _ in range(steps):
        if len(walkers) == 0:
            break
        
        new_walkers = []
        for i in range(len(walkers)):
            X = random.random()

            # choose between elongation, annihilation, bifurcation
            if X < prob_annihilation:
                n_branches = 0
            elif X < (prob_annihilation + prob_bifurcation):
                n_branches = 2
            else:
                n_branches = 1

            # walk
            position = walkers[i]
            for _ in range(n_branches):
                if position == 0:
                    step_sign = 1 if random.random() < 0.5 else -1
                else:
                    step_sign = (1 if random.random() < prob_right else -1) * sign(position)
                step = step_size * step_sign
                new_position = position + step
                
                visit_counts[abs(new_position)] += 1
                new_walkers.append(new_position)
                
        walkers = new_walkers

    # make a visit count at different intervals than step size
    visit_counts = np.array(sorted(visit_counts.items())).T
    if bin_size and bin_size > 0:
        max_distance = steps * step_size
        xp = np.arange(0, max_distance + bin_size, bin_size)
        yp = np.interp(xp, visit_counts[0, :], visit_counts[1, :])
        yp[xp > max_distance] = 0
        return xp.tolist(), yp.tolist()
    else:
        return visit_counts[0, :].tolist(), visit_counts[1, :].tolist()


def run_multiple_trials(n_trials: int, steps: int, step_size: float=0.5, prob_right: float = 0.5, rate_bifurcation: float = 0, rate_annihilation: float = 0, bin_size: float=None, base_seed: int = 42):
    """
    Run multiple trials of the random walk and store each path.

    Parameters:
        n_trials (int): Number of independent random walk trials.
        steps (int): Number of steps in each walk.
        prob_right (float): Probability of moving right.
        base_seed (int): Seed for reproducibility.

    Returns:
        list of lists: Each inner list is a position trace of one walk.
    """
    all_walks = []
    for trial in range(n_trials):
        seed = base_seed + trial  # unique seed per trial
        sp = random_walk_1d(steps, step_size, prob_right, seed, rate_bifurcation, rate_annihilation, bin_size)
        all_walks.append(sp)
    return all_walks



# Example usage:
if __name__ == "__main__":
    steps = 350  # Number of steps in the random walk
    prob_right = 1  # Probability of moving right
    rate_bifurcation = 0.05
    rate_annihilation = 0.01
    n_trials = 100
    all_walks = run_multiple_trials(100, steps, prob_right=prob_right, rate_bifurcation=rate_bifurcation, rate_annihilation=rate_annihilation, bin_size=50)

    all_visits = np.array([ yp + ([0.] * (14 - len(yp))) for _, yp in all_walks ])
    distance = np.arange(0, 14) * 50
    visits_mean = np.mean(all_visits, axis=0)
    visits_std = np.std(all_visits, axis=0)
    
    # Plotting the result
    plt.plot(distance, visits_mean, color='blue')
    plt.scatter(distance, visits_mean, color='blue')
    plt.fill_between(distance, visits_mean - visits_std, visits_mean + visits_std, alpha=0.1, color='blue')
    
    plt.xlabel('Distance')
    plt.ylabel('Visits')
    plt.title('1D Random Walk')
    plt.grid(True)
    plt.show()
