from test_1 import *

# Example usage:
if __name__ == "__main__":

    

    import sys
    cell_type = sys.argv[sys.argv.index('--cell_type') + 1]    # get cell type from command line

    # number of bifurcations
    data_bif = {
        'SP':{'Mean':18.8, 'SD':4.3},
        'SL':{'Mean':18.0, 'SD':9.9},
        }[cell_type]
    
    # get number of neurons from Bathellier et al
    n_exp_neuron = {
        'SP':12,
        'SL':3
        }[cell_type]
    
    try:
        n_trials = int(sys.argv[sys.argv.index('--n_trials') + 1])
    except ValueError:
        n_trials = 100

    try:
        step_size = float(sys.argv[sys.argv.index('--step_size')+1])
    except ValueError:
        step_size = 0.5

    try:
        prob_fugal = float(sys.argv[sys.argv.index('--prob_fugal')+1])
    except ValueError:
        prob_fugal = 1       

    try:
        alpha = float(sys.argv[sys.argv.index('--alpha')+1])
    except ValueError:
        alpha = 0.01
        
    assert cell_type == 'SL' or cell_type == 'SP'   # check cell type is correct
    
    filepath = {'SP':'sp_apical_sholl_plot.txt', 'SL':'sl_apical_sholl_plot.txt'}[cell_type] # file containing sholl plots
    
    # load the sholl plots
    exp_data = read_distance_csv_no_header(filepath)
    exp_data.drop(exp_data[(exp_data.Mean == 0) & (exp_data.SD == 0)].index, inplace=True)


    # assess equal size of intervals
    dx = np.unique(np.diff(exp_data.index))
    assert dx.size == 1
    dx = dx[0]

    
    # initial number of dendrites
    init_num = exp_data.loc[0, ['Mean', 'SD']].tolist()

    # estimate bifurcation and annihilation rates
    sim_data = []
    num_bifurcations = []
    for bif_factor in [0.5, 1, 2, 3]:
        rate_bifurcation, rate_annihilation = rates.solve_qp(step_size, dx, exp_data.Mean.to_numpy(), exp_data.SD.to_numpy(), n_bif= (data_bif['Mean'] * bif_factor, None))

        # run multiple trials
        xp = np.arange(0, rate_bifurcation.size) * dx
        all_walks, _num_bifurcations = run_multiple_trials(np.array([xp, rate_bifurcation]).T, np.array([xp, rate_annihilation]).T,
                                                          prob_fugal=prob_fugal, init_num=init_num, bin_size_interp=dx, n_trials=n_trials, step_size=step_size, base_seed=0)
        # get the number of bifurcations
        _num_bifurcations = [ tmp[-1, 1] for tmp in _num_bifurcations ]
        num_bifurcations.append(_num_bifurcations)
        print(np.mean(_num_bifurcations))
        # calculate the sholl plots for simulation data
        tmp = np.array([walk[:, 1].reshape(-1) for walk in all_walks]).T
        sim_data.append(rowwise_mean_sd(tmp, dx))
        print(sim_data[-1])
##
##    # calculate theoretical sholl plots
##    theor_data = expected_mean_sd(exp_data.loc[0, 'Mean'], exp_data.loc[0, 'SD'] ** 2, rate_bifurcation, rate_annihilation, dx)
##
##    # perform tests between sholl plots
##    if sim_data.size == exp_data.size:
##        r = compare_dataframes(sim_data, exp_data, n_trials, n_exp_neuron)
####        print('\nsim vs exp')
####        print(r, '\n\n')
##        significant_bins1 = r[r.SD < alpha].index
##        significant_bins2 = r[r.Mean < alpha].index
##        
####        r = compare_dataframes(sim_data, theor_data, n_trials, None)
####        print('\nsim vs theor')
####        print(r, '\n\n')
####
####        r = compare_dataframes(exp_data, theor_data, n_exp_neuron, None)
####        print('\nexp vs theor')
####        print(r, '\n\n')
####        
####        print('Exp\n', exp_data, '\n')
####        print('Sim\n', sim_data, '\n')
####        print('Th.\n', theor_data, '\n')
##    
##    # compare mean and variances
##
##        
##    bif_mean_exp = data_bif['Mean']
##    bif_std_exp = data_bif['SD']
##    bif_mean_sim = np.mean(num_bifurcations)
##    bif_std_sim = np.std(num_bifurcations)
##
##    bif_mean_theor, bif_std_theor = total_bifurcation_mean_std(dx, rate_bifurcation, rate_annihilation, theor_data['Mean'].to_numpy(), np.power(theor_data['SD'].to_numpy(), 2))
##
##    print(bif_mean_exp, bif_std_exp)
##    print(bif_mean_theor, bif_std_theor) 
##    # compare number of bifurcations    
##    print(f'Exp:\t{bif_mean_exp:.1f}+/-{bif_std_exp:.1f}\nSim:\t{bif_mean_sim:.1f}+/-{bif_std_sim:.1f}\nTheor.:\t{bif_mean_theor:.1f}+/-{bif_std_theor:.1f}')
##
##    r = one_sample_tests(bif_mean_exp, bif_std_exp ** 2, n_exp_neuron, mean_h0=bif_mean_theor, var_h0=bif_std_theor ** 2)
##    print('exp vs th', r)
##                          
##    r = one_sample_tests(bif_mean_sim, bif_std_sim ** 2, n_trials, mean_h0=bif_mean_theor, var_h0=bif_std_theor ** 2)
##    print('sim vs th', r)
##
##    r = two_samples_tests(bif_mean_sim, bif_std_sim ** 2, n_trials, bif_mean_exp, bif_std_exp ** 2, n_exp_neuron)
##    print('sim vs exp', r)
    
        
    # Plotting the result
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
##    plot_histogram(num_bifurcations, exp_mean=bif_mean_exp, exp_std=bif_std_exp)
##    plt.savefig(f'hist_{cell_type}.png', dpi=300)
##    plt.show()
    
    plot_three_curves_with_errorbars(sim_data[0].index, sim_data[0].Mean, sim_data[0].SD, sim_data[1].Mean, sim_data[1].SD, y3=sim_data[2].Mean, err3=sim_data[2].SD,
                                     color1='black', color2='darkred', color3='red', label1='Sim.', label2='Sim. X2', label3='Sim. X3')
    plt.savefig(f'sholl_plots_three_{cell_type}.png', dpi=300)
    plt.show()


    plot_custom_boxplots(num_bifurcations, ['black', 'darkred', 'red'], ['Sim.', 'Sim. X2', 'Sim. X3'])
    plt.show()
