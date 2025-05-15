import random
import matplotlib.pyplot as plt
import numpy as np
import model
import rates

import pandas as pd



from stats import *
from rw import *



    

def read_distance_csv_no_header(filepath):
    """
    Reads a CSV file without a header, assuming columns are:
    distance, mean, SD (in that order).

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns ['distance', 'Mean', 'SD'].
    """
    df = pd.read_csv(filepath)
    return df.astype(float).set_index('Distance').sort_index()


def expected_mean_sd(Mean0, Var0, rate_bifurcation, rate_annihilation, bin_size):
    # Apply row labels if provided
    assert len(rate_bifurcation) == len(rate_annihilation)
    r = {'Distance':np.arange(0, len(rate_bifurcation) + 1) * bin_size,
         'Mean':[Mean0],
         'Var':[Var0]}

    for i in range(len(rate_bifurcation)):
        Ez0 = r['Mean'][-1]
        Varz0 = r['Var'][-1]
        Ez1 = model.mean_Z(bin_size, Ez0, rate_bifurcation[i], rate_annihilation[i])
        Varz1 = model.var_Z(bin_size, Ez0, Varz0, rate_bifurcation[i], rate_annihilation[i])
        r['Mean'].append(Ez1)
        r['Var'].append(Varz1)

    r = pd.DataFrame(r)
    r['SD'] = np.sqrt(r['Var'])
    r = r.drop('Var', axis=1).set_index('Distance')
    return r


def rowwise_mean_sd(data, bin_size, index_names=None):
    """
    Calculates the mean and standard deviation for each row of a 2D array or DataFrame.

    Parameters:
        data (np.ndarray or pd.DataFrame): Input 2D numerical data.
        index_names (list of str, optional): Row names for the result. If None, uses default or DataFrame indices.

    Returns:
        pd.DataFrame: DataFrame with 'Mean' and 'Standard Deviation' for each row.
    """
    # Convert to DataFrame if needed
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Input must be a NumPy array or Pandas DataFrame.")

    # Apply row labels if provided
    if index_names is not None:
        if len(index_names) != df.shape[0]:
            raise ValueError("Length of index_names must match number of rows.")
        df.index = index_names

        
    result = pd.DataFrame({
        'Distance':np.arange(df.shape[0]) * bin_size,
        'Mean': df.mean(axis=1),
        'SD': df.std(axis=1)
    }).set_index('Distance')

    return result


def plot_histogram(data, bins=10, title='', xlabel='number of branch points', ylabel='frequency',
                   color='black', edgecolor=None, exp_mean=None, exp_std=None):
    """
    Plots a histogram from a NumPy array using matplotlib and shows:
    - Mean ± std of the data
    - Optional second error bar (exp_mean ± exp_std) above it

    Parameters:
        data (np.ndarray): Input numerical data array.
        bins (int or sequence): Number of bins or bin edges.
        title (str): Title of the histogram.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        color (str): Fill color of the bars and error bars.
        edgecolor (str or None): Color of the bar edges.
        exp_mean (float or None): Optional second mean for comparison.
        exp_std (float or None): Optional second standard deviation.
    """
    mean = np.mean(data)
    std = np.std(data)

    plt.figure(figsize=(3.5, 4.8/6.4*4))
    counts, bins_edges, _ = plt.hist(data, bins=np.linspace(0,60,bins), color=color, edgecolor=edgecolor)

    # Positioning error bars
    y_max = max(counts)
    y1 = y_max + y_max * 0.05       # position for main mean±SD
    y2 = y1 + y_max * 0.15          # position for exp_mean±exp_std

    # First error bar: sample mean ± std
    plt.errorbar(mean, y1, xerr=std, fmt='o', color=color, capsize=0,
                 label=f'Sim')

    # Second error bar: expected/reference mean ± std
    if exp_mean is not None and exp_std is not None:
        plt.errorbar(exp_mean, y2, xerr=exp_std, fmt='o', color='lightgray', mfc='white', capsize=0,
                     label=f'Exp')

    # Formatting
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0,60])
    plt.ylim([0,50]) #(top=y2 + y_max * 0.15)  # ensure both error bars are visible
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.show()

    
def plot_two_curves_with_errorbars(x, y1, err1, y2, err2, 
                                    label1='Sim', label2='Exp', 
                                    color1='black', color2='lightgray', 
                                    xlabel='distance from soma($\mu$m)', ylabel='intersections', title='',
                                    significant_bins1=[], significant_bins2=[], marker_height1=0.1, marker_height2=10):
    
    return plot_three_curves_with_errorbars(x, y1, err1, y2, err2, y3=None, err3=None,
                                    label1=label1, label2=label2, 
                                    color1=color1, color2=color2, 
                                    xlabel=xlabel, ylabel=ylabel, title=title,
                                    significant_bins1=significant_bins1, significant_bins2=significant_bins2, marker_height1=marker_height1, marker_height2=marker_height2)
                                     
    
def plot_three_curves_with_errorbars(x, y1, err1, y2, err2, y3=None, err3=None,
                                    label1='Sim', label2='Exp', label3='Test',
                                    color1='black', color2='lightgray', color3='red',
                                    xlabel='distance from soma($\mu$m)', ylabel='intersections', title='',
                                    significant_bins1=[], significant_bins2=[], marker_height1=0.1, marker_height2=10):
    """
    Plots two curves with error bars and asterisk markers for statistically significant bins.

    Parameters:
        x (array-like): Shared x-axis values.
        y1, y2 (array-like): Mean values for each curve.
        err1, err2 (array-like): Corresponding error (std or SEM) for each curve.
        label1, label2 (str): Labels for the curves.
        color1, color2 (str): Colors for the curves.
        xlabel, ylabel, title (str): Axis and plot labels.
        significant_bins (list of bool or indices): Marks "*" on x positions where True or present.
        marker_height (float): Additional vertical space above the highest error bar at a bin.
    """
    plt.figure(figsize=(7, 4.8/6.4*4))

    pad = 700 * 0.0075 #(x[-1] - x[0]) * 0.01

    # Plot error bars
    if y3 is not None and err3 is not None:
        plt.errorbar(x - pad, y3, yerr=err3, label=label3, fmt='-o', color=color3, capsize=0)
        plt.errorbar(x, y2, yerr=err2, label=label2, fmt='-o', color=color2, capsize=0)
        plt.errorbar(x + pad, y1, yerr=err1, label=label1, fmt='-o', color=color1, capsize=0)
    else:
        plt.errorbar(x + pad, y2, yerr=err2, label=label2, fmt='-o', color=color2, capsize=0, mfc='white')
        plt.errorbar(x - pad, y1, yerr=err1, label=label1, fmt='-o', color=color1, capsize=0)
        
    # Handle significance markers
    for xi in x:
            # Calculate the higher point between the two ± error
            top1 = y1[xi] + err1[xi]
            top2 = y2[xi] + err2[xi]
            ymax = max(top1, top2)
            if xi in significant_bins1:
                plt.text(xi + pad, ymax + marker_height1, "*", ha='center', va='bottom', fontsize=14, color='black')
            if xi in significant_bins2:
                plt.text(xi - pad, ymax + marker_height2, "#", ha='center', va='bottom', fontsize=14, color='black')

                
    # Format plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlim([-25, 725])
    plt.ylim([0, 25])
    #plt.show()
    



def compare_dataframes(df1, df2, n1, n2):
    p_values = {'Distance':list(), 'Mean':list(), 'SD':list()}
    assert (df1.index == df2.index).all()
    for d in df1.index:
        p_values['Distance'].append(d)
        if n2:
            r = two_samples_tests(df1.loc[d, 'Mean'], df1.loc[d, 'SD'] ** 2, n1, df2.loc[d, 'Mean'], df2.loc[d, 'SD'] ** 2, n2)
            p_values['Mean'].append(r['Z Test']['p_value'])
            p_values['SD'].append(r['F Test']['p_value'])
        else:
            r = one_sample_tests(df1.loc[d, 'Mean'], df1.loc[d, 'SD'] ** 2, n1, df2.loc[d, 'Mean'], df2.loc[d, 'SD'] ** 2)
            p_values['Mean'].append(r['T Test']['p_value'])
            p_values['SD'].append(r['Chi2 Test']['p_value'])
    return pd.DataFrame(p_values).set_index('Distance')


def total_bifurcation_mean_std(dx, rate_bifurcation, rate_annihilation, mean_z, var_z):
    kappa = rate_bifurcation - rate_annihilation

    bif_mean_theor = 0
    for i in range(len(rate_bifurcation)):
        bif_mean_theor += model.mean_B(dx, mean_z[i], rate_bifurcation[i], rate_annihilation[i])

    
    bif_var_theor = 0
    for i in range(len(rate_bifurcation)):
        bif_var_theor += model.var_B(dx, mean_z[i], var_z[i], rate_bifurcation[i], rate_annihilation[i])

    for i in range(1, len(rate_bifurcation)):           
        for j in range(i + 1, len(rate_bifurcation) + 1):
            term1 = rate_bifurcation[i - 1] * (np.exp(kappa[i - 1] * dx) - 1) / kappa[i - 1] * np.exp(-kappa[i - 1] * dx) if i >= 1 else 1
            term2 = rate_bifurcation[j - 1] * (np.exp(kappa[j - 1] * dx) - 1) / kappa[j - 1] * np.exp(np.sum(kappa[np.arange(i, j - 1)]) * dx)
            bif_var_theor += 2 * term1 * term2 * var_z[i]

    bif_std_theor = np.sqrt(bif_var_theor)

    return bif_mean_theor, bif_std_theor

def plot_custom_boxplots(data, box_colors, labels):
    """
    Plots boxplots with no background, custom colors for borders/whiskers/outliers,
    x-axis labels, and a legend.

    Parameters:
    - data: list of arrays or lists, each representing a dataset.
    - box_colors: list of colors (str or tuple), one for each boxplot.
    - labels: list of str, x-axis labels and legend labels.
    - outlier_kwargs: dict of keyword arguments for outlier styling (optional).
    """

    if not (len(data) == len(box_colors) == len(labels)):
        raise ValueError("data, box_colors, and labels must have the same length.")

    # Default outlier style if not provided
    outlier_kwargs = {'marker': 'o', 'markersize': 6, 'alpha': 0.7}

    fig, ax = plt.subplots()

    # Remove background
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')

    # Create boxplot
    box = ax.boxplot(data, patch_artist=True)

    for i, color in enumerate(box_colors):
        # Box
        box['boxes'][i].set(color=color, linewidth=2, facecolor='none')

        # Whiskers
        box['whiskers'][2*i].set(color=color, linewidth=2)
        box['whiskers'][2*i+1].set(color=color, linewidth=2)

        # Caps
        box['caps'][2*i].set(color=color, linewidth=2)
        box['caps'][2*i+1].set(color=color, linewidth=2)

        # Median
        box['medians'][i].set(color=color, linewidth=2)

        # Fliers (outliers) — match box color
        box['fliers'][i].set(marker=outlier_kwargs.get('marker', 'o'),
                             markersize=outlier_kwargs.get('markersize', 6),
                             alpha=outlier_kwargs.get('alpha', 0.7),
                             markerfacecolor=color,
                             markeredgecolor=color,
                             linestyle='none')

    # Set x-axis labels
    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels)

    # Create legend
##    legend_handles = [Patch(edgecolor=color, facecolor='none', label=label, linewidth=2) 
##                      for color, label in zip(box_colors, labels)]
##    ax.legend(handles=legend_handles, loc='upper right', frameon=False)

    # Optional: clean up spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()


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
        n_trials = 50

    try:
        bif_factor = float(sys.argv[sys.argv.index('--n-bif-factor')+1])
    except ValueError:
        bif_factor = 1

    if '--n-bif-mean' in sys.argv:
        n_bif = (data_bif['Mean'] * bif_factor, None)
    elif '--n-bif-var' in sys.argv:
        n_bif = (None, (data_bif['SD'] * bif_factor) ** 2)
    elif '--n-bif-both' in sys.argv:
        n_bif = (data_bif['Mean'] * bif_factor, (data_bif['SD'] * bif_factor) ** 2)
    else:
        n_bif = (None, None)

    try:
        step_size = float(sys.argv[sys.argv.index('--step_size')+1])
    except ValueError:
        step_size = 5

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

    # estimate bifurcation and annihilation rates
    rate_bifurcation, rate_annihilation = rates.solve_qp(step_size, dx, exp_data.Mean.to_numpy(), exp_data.SD.to_numpy(), n_bif=n_bif, kappa_Penalty_Var=1.0)

    print(rate_bifurcation)
    print(rate_annihilation)
    
    # initial number of dendrites
    init_num = exp_data.loc[0, ['Mean', 'SD']].tolist()

    # run multiple trials
    xp = np.arange(0, rate_bifurcation.size) * dx
    all_walks, num_bifurcations = run_multiple_trials(np.array([xp, rate_bifurcation]).T, np.array([xp, rate_annihilation]).T,
                                                      prob_fugal=prob_fugal, init_num=init_num, bin_size_interp=dx, n_trials=n_trials, step_size=step_size, base_seed=0)
    # get the number of bifurcations
    num_bifurcations = [ tmp[-1, 1] for tmp in num_bifurcations ]

    # calculate the sholl plots for simulation data
    tmp = np.array([walk[:, 1].reshape(-1) for walk in all_walks]).T
    sim_data = rowwise_mean_sd(tmp, dx)

    # calculate theoretical sholl plots
    theor_data = expected_mean_sd(exp_data.loc[0, 'Mean'], exp_data.loc[0, 'SD'] ** 2, rate_bifurcation, rate_annihilation, dx)

    # perform tests between sholl plots
    if sim_data.size == exp_data.size:
        r = compare_dataframes(sim_data, exp_data, n_trials, n_exp_neuron)
##        print('\nsim vs exp')
##        print(r, '\n\n')
        significant_bins1 = r[r.SD < alpha].index
        significant_bins2 = r[r.Mean < alpha].index
        
##        r = compare_dataframes(sim_data, theor_data, n_trials, None)
##        print('\nsim vs theor')
##        print(r, '\n\n')
##
##        r = compare_dataframes(exp_data, theor_data, n_exp_neuron, None)
##        print('\nexp vs theor')
##        print(r, '\n\n')
##        
##        print('Exp\n', exp_data, '\n')
##        print('Sim\n', sim_data, '\n')
##        print('Th.\n', theor_data, '\n')
    
    # compare mean and variances

        
    bif_mean_exp = data_bif['Mean']
    bif_std_exp = data_bif['SD']
    bif_mean_sim = np.mean(num_bifurcations)
    bif_std_sim = np.std(num_bifurcations)

    bif_mean_theor, bif_std_theor = total_bifurcation_mean_std(dx, rate_bifurcation, rate_annihilation, theor_data['Mean'].to_numpy(), np.power(theor_data['SD'].to_numpy(), 2))

    print(bif_mean_exp, bif_std_exp)
    print(bif_mean_theor, bif_std_theor) 
    # compare number of bifurcations    
    print(f'Exp:\t{bif_mean_exp:.1f}+/-{bif_std_exp:.1f}\nSim:\t{bif_mean_sim:.1f}+/-{bif_std_sim:.1f}\nTheor.:\t{bif_mean_theor:.1f}+/-{bif_std_theor:.1f}')

    r = one_sample_tests(bif_mean_exp, bif_std_exp ** 2, n_exp_neuron, mean_h0=bif_mean_theor, var_h0=bif_std_theor ** 2)
    print('exp vs th', r)
                          
    r = one_sample_tests(bif_mean_sim, bif_std_sim ** 2, n_trials, mean_h0=bif_mean_theor, var_h0=bif_std_theor ** 2)
    print('sim vs th', r)

    r = two_samples_tests(bif_mean_sim, bif_std_sim ** 2, n_trials, bif_mean_exp, bif_std_exp ** 2, n_exp_neuron)
    print('sim vs exp', r)
    
        
    # Plotting the result
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    plot_histogram(num_bifurcations, exp_mean=bif_mean_exp, exp_std=bif_std_exp)
    plt.savefig(f'hist_{cell_type}.png', dpi=300)
    plt.show()
    
    plot_two_curves_with_errorbars(sim_data.index, sim_data.Mean, sim_data.SD, exp_data.Mean, exp_data.SD, significant_bins1=significant_bins1, significant_bins2=significant_bins2)
    plt.savefig(f'sholl_plots_{cell_type}.png', dpi=300)
    plt.show()
