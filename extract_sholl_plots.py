import neurom as nm
import matplotlib.pyplot as plt
from neurom.core.types import NeuriteType
import numpy as np

from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

def process_morphology(m, dx=50.0):
    collections = defaultdict(int)
    segments = []

    tot = 0.0
    bw = 0.0
    bp = 0
    for s in nm.iter_sections(m):
        if s.type == 4:
            if s.parent is None:
                print('first point', s.points[0])
                collections[0] += 1
                
            if len(s.children) == 2:
                bp += 1
            elif len(s.children) > 2 or len(s.children) == 1:
                print(len(s.children))
                raise Exception()
                  
            for i in range(1, len(s.points)):
                slen = abs(np.linalg.norm(s.points[i - 1][:3]) - np.linalg.norm(s.points[i][:3]))
                tot += slen
                
                i0 = int(np.linalg.norm(s.points[i - 1][:3]) / dx)
                i1 = int(np.linalg.norm(s.points[i][:3]) / dx)
                segments.append(np.linalg.norm(s.points[i][:3] - s.points[i - 1][:3]))
                if i1 != i0:
                    if i0 > i1:
                        print('Warning there is a segment that move backward: ', slen)
                        bw += slen
                        collections[i0 * dx] += 1
                    elif i0 < i1:
                        if i1 - i0 > 1:
                            print('Warning there is a segment that skip spatial intervals')
                        else:
                            # count intersection
                            collections[i1 * dx] += 1
    l = bw / tot
    print('segments (min/max):', min(segments), '/', max(segments), ' l=', l)
    print()
    return collections, l, bp


def process_morphologies(dirname, xmax, filename, dx=50.0):
    xp = np.arange(0, xmax, dx)
    yp_all = []
    l_all = []
    bp_all = []
    for p in nm.load_morphologies(dirname):
        collections, l, bp = process_morphology(p)
        tmp = np.array(sorted(collections.items()))
        yp = np.interp(xp, tmp[:, 0], tmp[:, 1])
        yp[xp > tmp[-1, 0]] = 0
        yp_all.append(yp)
        plt.plot(xp, yp, color='gray')
        l_all.append(l)
        bp_all.append(bp)
    plt.errorbar(xp, np.mean(yp_all, axis=0), yerr=np.std(yp_all, axis=0), color='black')
    plt.show()
    pd.DataFrame({'Distance':xp, 'Mean':np.mean(yp_all, axis=0), 'SD':np.std(yp_all, axis=0)}).set_index('Distance').to_csv(filename)
    print('l +/- SE:', np.mean(l_all), np.std(l_all) / np.sqrt(len(l_all)))
    print('branch points +/- SD:', np.mean(bp_all), np.std(bp_all))

if __name__ == '__main__':
    print('\nSP')
    process_morphologies('morphologies/SP', 800, 'sp_apical_sholl_plot.txt')
    print('\nSL')
    process_morphologies('morphologies/SL', 500, 'sl_apical_sholl_plot.txt')
