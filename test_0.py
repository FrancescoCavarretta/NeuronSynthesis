import random
import matplotlib.pyplot as plt
import numpy as np
import model

import pandas as pd

import pickle

from stats import *
from rw import *
from concurrent.futures import ThreadPoolExecutor


if __name__ == "__main__":



    
    data = {}
    param_list = [ (b, l, m0, m0 * s0) for b in np.arange(0.1, 0.205, 0.005) for l in np.arange(-0.10, 0.105, 0.005) for m0 in [30] for s0 in [0.5] if (-l + b) >= 0]
    n_trials = 100
    max_distance = 50.5

    def wrapper(args):
        b, l, m0, s0 = args
        a = - l + b
        return args, run_multiple_trials(b, a, max_distance=max_distance, init_num=(m0, s0), bin_size_interp=5, n_trials=n_trials)
        

    for j, key in enumerate(param_list):
            b, l, m0, s0 = key
            a = - l + b
            
            print(j, ' out of ', len(param_list))
            data[key] = run_multiple_trials(b, a, max_distance=max_distance, init_num=(m0, s0), bin_size_interp=5, step_size=0.5, n_trials=n_trials)
        
            

    with open('sim_data_0.1.pkl', 'wb') as f:
        pickle.dump(data, f)

    print('done!')
