from rw import *


def custom_boxplot_from_arrays(sample1, sample2, sample3, sample4):
    """
    Boxplot for three samples with customized styling:
    - Sample 1: gray box, black border and features
    - Sample 2: transparent box, gray border and features, thick edge
    - Sample 3: red box, black border and features
    """
    data = [sample1, sample2, sample3, sample4]
    edge_colors = ['gray', 'black', 'red', 'blue']
    face_colors = ['none', 'none', 'none', 'none']
    linewidths = [1, 2, 1, 1]


    fig = plt.figure(figsize=(5, 3.5))

    bp = plt.boxplot(
        data,
        patch_artist=True,
        widths=0.6
    )

    # Loop through each box
    for i in range(4):
        # Box
        bp['boxes'][i].set_facecolor(face_colors[i])
        bp['boxes'][i].set_edgecolor(edge_colors[i])
        bp['boxes'][i].set_linewidth(linewidths[i])

        # Whiskers (2 per box)
        bp['whiskers'][2*i].set_color(edge_colors[i])
        bp['whiskers'][2*i+1].set_color(edge_colors[i])
        bp['whiskers'][2*i].set_linewidth(linewidths[i])
        bp['whiskers'][2*i+1].set_linewidth(linewidths[i])

        # Caps (2 per box)
        bp['caps'][2*i].set_color(edge_colors[i])
        bp['caps'][2*i+1].set_color(edge_colors[i])
        bp['caps'][2*i].set_linewidth(linewidths[i])
        bp['caps'][2*i+1].set_linewidth(linewidths[i])

        # Medians
        bp['medians'][i].set_color(edge_colors[i])
        bp['medians'][i].set_linewidth(linewidths[i])

        # Fliers
        bp['fliers'][i].set_marker('o')
        bp['fliers'][i].set_markerfacecolor(edge_colors[i])
        bp['fliers'][i].set_markeredgecolor=edge_colors[i]
        bp['fliers'][i].set_markersize(5)

    # X-axis labels
    plt.xticks([1, 2, 3, 4], ['RW', 'BRW', r'BRW, sh. $\beta$ and $\alpha$', r'BRW, inc. $\beta$'], rotation=15)
    plt.ylabel('branch point count')
    plt.tight_layout()
    plt.ylim([0, 1500])
    plt.yticks([0, 500, 1000, 1500])
    #plt.legend(loc='upper left')
    #plt.show()

    return fig


def plot_mean_sd_with_points(data, category_labels=None, point_jitter=0.1, colors=None, significance=[], h=[], xsh=[]):
    """
    Plot data points with mean ± SD error bars for multiple categories.

    Parameters:
    - data: list of arrays/lists, one array per category (length = number of categories)
    - category_labels: list of strings for x-axis ticks (default: 1,2,...)
    - point_jitter: float, max horizontal jitter to spread data points for visibility
    """

    num_categories = len(data)
    if category_labels is None:
        category_labels = [f'Cat {i+1}' for i in range(num_categories)]

    fig = plt.figure(figsize=(5, 3.5))

    means = [np.mean(d) for d in data]
    sds = [np.std(d, ddof=1) for d in data]  # sample standard deviation

    # Plot individual points with jitter
    for i, d in enumerate(data):
        x = np.random.normal(loc=i+1, scale=point_jitter, size=len(d))
        if colors:
            color = colors[i]
        else:
            color = 'gray'
            
        plt.scatter(x, d, alpha=0.05, label=None, color=color)

        # Plot mean ± SD error bars
        plt.errorbar([i+1], [means[i]], yerr=[sds[i]], fmt='o', 
                    color=color, ecolor=color, elinewidth=2, capsize=5, label='Mean ± SD')

    for k, (i, j, text) in enumerate(significance):
        y0 = means[i-1] + sds[i-1]
        y1 = means[j-1] + sds[j-1]
        y2 = max([y0, y1])
        plt.plot([i + xsh[k][0], i + xsh[k][0], j + xsh[k][1], j + xsh[k][1]], [y0+h[k][0], y2+h[k][1], y2+h[k][1], y1+h[k][0]], lw=1, color='black')
        plt.text((i + j) * 0.5, y2 + h[k][1] - 10, text, ha='center', va='bottom', color='black')
    
    plt.xticks(range(1, num_categories+1), labels=category_labels, rotation=10)
    #ax.set_xlabel('Category')
    plt.ylabel('Branch point count')
    #plt.legend(loc='upper left')

    plt.tight_layout()
    return fig


def plot(walks, color='black', alpha=1, label=None, linewidth=1):
    x = all_walks[0][:, 0] 
    y = [w[:, 1] for w in all_walks]
    m = np.mean(y, axis=0)
    s = np.std(y, axis=0)
    plt.plot(x, m, color=color, label=label, linewidth=linewidth)
    plt.fill_between(x, m - s, m + s, color=color, alpha=alpha)

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
    sh2 = 0.0025
    max_time = 1000
    init_num = (20, 2)
    prob_fugal1 = 0.5
    prob_fugal2 = 0.75
    
    all_walks, all_bif1 = run_multiple_trials(rb, ra, max_time=max_time, prob_fugal = prob_fugal1, step_size = step_size, init_num = init_num, base_seed=200)
    plot(all_walks, color='gray', alpha=.5, label='RW')

    all_walks, all_bif2 = run_multiple_trials(rb, ra, max_time=max_time, prob_fugal = prob_fugal2, step_size = step_size, init_num = init_num, base_seed=200)
    plot(all_walks, color='black', alpha=.5, label='BRW', linewidth=2)

    all_walks, all_bif3 = run_multiple_trials(rb + sh1, ra + sh1, max_time=max_time, prob_fugal = prob_fugal2, step_size = step_size, init_num = init_num, base_seed=200)
    plot(all_walks, color='red', alpha=.25, label=r'BRW, sh. $\beta$ and $\alpha$')

    all_walks, all_bif4 = run_multiple_trials(rb + sh2, ra, max_time=max_time, prob_fugal = prob_fugal2, step_size = step_size, init_num = init_num, base_seed=200)
    plot(all_walks, color='blue', alpha=.25, label=r'BRW, inc. $\beta$')
    plt.xlim([0, 250])
    plt.ylim([0, 1200])
    plt.yticks([0, 200, 400, 600, 800, 1000, 1200])
    plt.xlabel('x')
    plt.ylabel('visits')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    fig.savefig('fig_1_b.png', dpi=300)
    plt.show()
    


    for tmp in [[ aux[-1, 1] for aux in all_bif1 ], [ aux[-1, 1] for aux in all_bif2 ], [ aux[-1, 1] for aux in all_bif3 ], [ aux[-1, 1] for aux in all_bif4 ]]:

        print(round(np.mean(tmp), 1), round(np.std(tmp), 1))
        
        
##    # plot bifurcations
##    fig = custom_boxplot_from_arrays([ aux[-1, 1] for aux in all_bif1 ], [ aux[-1, 1] for aux in all_bif2 ], [ aux[-1, 1] for aux in all_bif3 ], [ aux[-1, 1] for aux in all_bif4 ])
##    
##    plt.show()

    tmp = [[ aux[-1, 1] for aux in all_bif1 ], [ aux[-1, 1] for aux in all_bif2 ], [ aux[-1, 1] for aux in all_bif3 ], [ aux[-1, 1] for aux in all_bif4 ]]
    
    fig = plot_mean_sd_with_points(tmp, ['RW', 'BRW', r'BRW, sh. $\beta$ and $\alpha$', r'BRW, inc. $\beta$'], colors=['gray', 'black', 'red', 'blue'],
                             significance=[(2, 3, '*'), (2, 4, '*')], h=[(10, 50), (10, 175)],
                             xsh=[(-0.015, 0.0), (0.015, 0.0)])
    fig.savefig('fig_1_c.png', dpi=300)
    plt.show()
    
    from stats import two_samples_tests


    print(two_samples_tests(np.mean(tmp[0]), np.var(tmp[0]), 100, np.mean(tmp[1]), np.var(tmp[1]), 100))
    print(two_samples_tests(np.mean(tmp[0]), np.var(tmp[0]), 100, np.mean(tmp[2]), np.var(tmp[2]), 100))
    print(two_samples_tests(np.mean(tmp[0]), np.var(tmp[0]), 100, np.mean(tmp[3]), np.var(tmp[3]), 100))
    print(two_samples_tests(np.mean(tmp[1]), np.var(tmp[1]), 100, np.mean(tmp[2]), np.var(tmp[2]), 100))
    print(two_samples_tests(np.mean(tmp[1]), np.var(tmp[1]), 100, np.mean(tmp[3]), np.var(tmp[3]), 100))
    print(two_samples_tests(np.mean(tmp[2]), np.var(tmp[2]), 100, np.mean(tmp[3]), np.var(tmp[3]), 100))
