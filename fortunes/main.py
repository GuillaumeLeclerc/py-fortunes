import numpy as np
from structures import Context, Arc, BreakPointNode, EventQueue, walk_tree
from scipy.spatial import voronoi_plot_2d, Voronoi
import matplotlib.pyplot as plt
import argparse

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--no-random", action='store_true',
                    help="Chose 30 points at random instead of manual input")
args = parser.parse_args()


def gather_points_manually():
    ax = plt.gca()
    ax.set_aspect('equal')

    points = plt.ginput(-1)
    plt.close()
    return np.array(points)

def gather_points_random():
    return np.random.uniform(0, 1, size=(30, 2))


if __name__ == '__main__':
    if not args.no_random:
        points = gather_points_random()
    else:
        points = gather_points_manually()

    context = Context(points)
    queue = EventQueue(context)
    plt.ion()
    context.plot()
    while len(queue) > 0:
        queue.handle_next()
        queue.plot()
        print(context.beach_line)
    solution = context.finalize()
    plt.close()
    voronoi_plot_2d(solution)

    plt.gca().set_aspect('equal')
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    plt.ginput()
