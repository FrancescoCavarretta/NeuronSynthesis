import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import model

l = np.arange(-0.1, 0.11, 0.01)
b = np.arange(0.1, 0.21, 0.01)
z = np.array([model.mean_Z(50, 30, _b, -_l + _b) for _b in b for _l in l])


l = np.tile(l, 11)
b = np.repeat(b, 21)

def plot_3d_triangulated_mesh(x, y, z, xlabel='X', ylabel='Y', zlabel='Z', title='3D Triangulated Mesh'):
    """
    Plots a 3D triangulated surface from 1D x, y, z arrays.

    Parameters:
    - x, y, z: 1D arrays of the same length
    - xlabel, ylabel, zlabel: Axis labels
    - title: Title of the plot
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, z must have the same shape")

    # Combine x and y to form 2D points for triangulation
    points2D = np.vstack((x, y)).T
    tri = Delaunay(points2D)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap='viridis', edgecolor='k', alpha=0.9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=30, azim=45, roll=15)
    plt.tight_layout()
    plt.show()



plot_3d_triangulated_mesh(l, b, z)
