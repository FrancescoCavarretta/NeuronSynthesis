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
    #plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1))
    #plt.show()

    return fig

def plot(walks, color='black', alpha=1, label=None, linewidth=1):
    x = all_walks[0][:, 0] 
    y = [w[:, 1] for w in all_walks]
    m = np.mean(y, axis=0)
    s = np.std(y, axis=0)
    plt.plot(x, m, color=color, alpha=alpha, label=label, linewidth=linewidth)
    plt.fill_between(x, m - s, m + s, color=color, alpha=alpha / 2)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.0, 3.5))

    plt.rcParams['font.size'] = 10
    plt.rcParams['font.weight'] = 'bold'  # NEW
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'  


    all_walks, all_bif1 = run_multiple_trials(0.05, 0.005, max_distance=50, prob_fugal = 0.5, init_num = (30, 3))
    plot(all_walks, color='gray', label='RW')

    all_walks, all_bif2 = run_multiple_trials(0.05, 0.005, max_distance=50, prob_fugal = 0.75, init_num = (30, 3))
    plot(all_walks, color='black', label='BRW', linewidth=2)

    all_walks, all_bif3 = run_multiple_trials(0.05 + 0.075, 0.005 + 0.075, max_distance=50, prob_fugal = 0.75, init_num = (30, 3))
    plot(all_walks, color='red', alpha=.5, label=r'BRW, sh. $\beta$ and $\alpha$')

    all_walks, all_bif4 = run_multiple_trials(0.05 + 0.025, 0.005, max_distance=50, prob_fugal = 0.75, init_num = (30, 3))
    plot(all_walks, color='blue', alpha=.5, label=r'BRW, inc. $\beta$')
    plt.xlim([0, 45])
    plt.ylim([0, 1650])
    plt.xlabel('x')
    plt.ylabel('visits')
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1))
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    fig.savefig('fig_1_b.png', dpi=300)
    plt.show()
    


    for tmp in [[ aux[-1, 1] for aux in all_bif1 ], [ aux[-1, 1] for aux in all_bif2 ], [ aux[-1, 1] for aux in all_bif3 ], [ aux[-1, 1] for aux in all_bif4 ]]:

        print(round(np.mean(tmp), 1), round(np.std(tmp), 1))
        
        
    # plot bifurcations
    fig = custom_boxplot_from_arrays([ aux[-1, 1] for aux in all_bif1 ], [ aux[-1, 1] for aux in all_bif2 ], [ aux[-1, 1] for aux in all_bif3 ], [ aux[-1, 1] for aux in all_bif4 ])
    fig.savefig('fig_1_c.png', dpi=300)
    plt.show()
