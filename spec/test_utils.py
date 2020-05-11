import math
from hypothesis import given, example, settings
from hypothesis.strategies import text, floats
from hypothesis.extra.numpy import arrays
import numpy as np

from fortunes.utils import circle_from_points, compute_coefs, arc_intersection, y, x

COL_ERROR = 1e-12
RADIUS_ERROR = 1e-5

def distance(p1, p2):
    return np.sqrt(((p1 - p2)**2).sum())

def collinear(p0, p1, p2):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1)

FLOAT_RANGE = floats(np.float32(-10**2), np.float32(10**2), allow_nan=False, width=32)

@settings(max_examples=1000)
@given(point=arrays(np.float64,  (2,), elements=FLOAT_RANGE),
       xes=arrays(np.float64,  (50,), elements=FLOAT_RANGE),
       line_offset=floats(np.float32(RADIUS_ERROR), 1000, width=32)
       )
def test_compute_coefs(point, xes, line_offset):
    line = y(point) + line_offset
    a, b, c = compute_coefs(point, line)
    yes = xes**2*a + xes*b + c
    for xx, yy in zip(xes, yes):
        distance_to_line = line - yy
        distance_to_site = np.sqrt((xx - x(point)) ** 2 + (yy - y(point))**2)
        assert math.isclose(distance_to_line, distance_to_site, rel_tol=1e-2)


@settings(max_examples=1000)
@given(points=arrays(np.float64,  (2,2), elements=FLOAT_RANGE),
       line_offset=floats(np.float32(RADIUS_ERROR), 1000, width=32)
       )
def test_arc_intersection(points, line_offset):
    if np.abs(x(points[0]) - x(points[1])) < 1e-5:
        return
    if np.linalg.norm(points[0] - points[1]) < 1e-4:
        return
    line = max(y(points[0]), y(points[1])) + line_offset
    a, b, c = compute_coefs(points[0], line)
    d, e, f = compute_coefs(points[1], line)
    predicted_intersection_x = arc_intersection(points[0], points[1], line)
    y_1 = a*predicted_intersection_x**2 + b * predicted_intersection_x + c
    y_2 = d*predicted_intersection_x**2 + e * predicted_intersection_x + f
    assert math.isclose(y_1, y_2, rel_tol=1e-2)


@given(arrays(np.float64, (3, 2), elements=floats(-10**5, 10**5, allow_nan=False, width=32)))
def test_circle_from_points(points):
    p1, p2, p3 = points
    center, radius = circle_from_points(p1, p2, p3)
    colinearity = collinear(p1, p2, p3)
    if radius >= 0:
        d1 = distance(center, p1)
        d2 = distance(center, p2)
        d3 = distance(center, p3)
        assert math.isclose(d1, radius, rel_tol=RADIUS_ERROR)
        assert math.isclose(d2, radius, rel_tol=RADIUS_ERROR)
        assert math.isclose(d3, radius, rel_tol=RADIUS_ERROR)
    else:
        assert colinearity < 1e-15



