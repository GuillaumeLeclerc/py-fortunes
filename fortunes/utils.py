import numpy as np

def jit(enabled=False):
    if enabled:
        try:
            from numba import njit
            return njit
        except:
            pass
    return lambda x: x

X_COORD = 0
Y_COORD = 1

@jit()
def x(point):
    return point[X_COORD]

@jit()
def y(point):
    return point[Y_COORD]


def circle_from_points(a, b, c):

    center = np.zeros(2, dtype='float64')
    radius = -1

    A = x(b) - x(a)
    B = y(b) - y(a)
    C = x(c) - x(a)
    D = y(c) - y(a)
    E = A * (x(a) + x(b)) + B * (y(a) + y(b))
    F = C * (x(a) + x(c)) + D * (y(a) + y(c))
    G = 2 * (A * (y(c) - y(b)) - B * (x(c) - x(b)))

    if G != 0:
        center[X_COORD] = ((D * E) - (B * F)) / G
        center[Y_COORD] = ((A * F) - (C * E)) / G

        radius = ((x(a) - x(center))**2 + (y(a) - y(center))**2)**0.5
    return center, radius



# Compute the coeficient of the parabola given the site and the sweep line
def compute_coefs(p, l):
    a, b = x(p), y(p)
    F = 1/(2 * (b - l))
    assert F <= 0
    return np.array((F, 2 * - a * F, F * (a ** 2 + b ** 2 - l**2)))


def arc_intersection(p1, p2, line):
    left_intersection = y(p1) < y(p2)
    a, b, c = compute_coefs(p2, line) - compute_coefs(p1, line)
    if a == 0:
        return -c / b
    delta = np.sqrt(b**2 - 4 * a * c)
    if delta <= 0:
        delta = 0
    solutions = ((-b + delta)/(2*a), (-b - delta)/(2 * a))
    if left_intersection:
        return min(solutions)
    else:
        return max(solutions)
