from collections import defaultdict, namedtuple
from numba import njit, jitclass
from numba import types
import numba

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--no-random", action='store_true',
                    help="Chose 30 points at random instead of manual input")
args = parser.parse_args()

Line = namedtuple('line', ['slope', 'intercept'])


spec = [
    ('points', numba.float32[:, :]),
    ('next_point', numba.int32),
]

@jitclass(spec)
class Queue(object):
    def __init__(self, points):
        self.points = points
        self.next_point = 0

    @property
    def length(self):
        points_left = len(self.points) - self.next_point

        # TODO intersection queue
        intersections_left = 0

        return points_left + intersections_left

    @property
    def is_next_point(self):
        #TODO check if an intersection is closer
        return True

    def pop_point(self):
        current_point = self.points[self.next_point]
        self.next_point += 1
        return current_point


def gather_points_manually():
    f = plt.figure()
    points = plt.ginput(-1)
    plt.close(f)
    return np.array(points)

def gather_points_random():
    return np.random.uniform(-1, 1, size=(30, 2))


def dist(points):
    if points.shape[0] == 1:
        return np.inf
    return np.sqrt(((points[0] - points[1])**2).sum())

def sort_points_numpy(points):
    return points[np.lexsort(points.T)]


@njit
def solve(queue):
    while queue.length > 0:
        if queue.is_next_point:
            point = queue.pop_point()
            print(point)

if not args.no_random:
    points = gather_points_random()
else:
    points = gather_points_manually()

print(points.shape)
points = np.array([[5, 1], [0, 1], [-1, 1] ,[0, 0], [5, 0],[-1, 0]])
points = sort_points_numpy(points)
queue = Queue(points.astype('float32'))

solve(queue)

