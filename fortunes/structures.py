from collections import namedtuple, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from utils import arc_intersection, x, y, circle_from_points, compute_coefs


VoronoiSolution = namedtuple("VoronoiSolution", ["points", "vertices",
                                                 "ridge_points", "ridge_vertices",
                                                 "regions"])

class Voronoi():

    def __init__(self, points):
        self.ridge_points = defaultdict(lambda: [])
        self.vertices = []

    def add_edge(self, x, y):
        if y < x:
            x, y = y, x
        self.ridge_points[(x, y)]

    def fill_edge_endpoint(self, x, y, vertex_id):
        if y < x:
            x, y = y, x
        self.ridge_points[(x, y)].append(vertex_id)


    def add_vertex(self, x):
        self.vertices.append(x)
        return len(self.vertices) - 1

    def build(self, context):
        points = context.points
        vertices = np.array(self.vertices)
        ridge_points = []
        ridge_vertices = []
        for k, v in self.ridge_points.items():
            while len(v) < 2:
                v = [-1] + v
            ridge_vertices.append(v)
            ridge_points.append(k)
        ridge_points = np.array(ridge_points)
        ridge_vertices = np.array(ridge_vertices)

        regions = [[] for p in points]
        vertices_in_regions = [set() for p in points]

        for i, (a, b) in enumerate(ridge_points):
            for vertex in ridge_vertices[i]:
                if vertex != -1:
                    vertices_in_regions[a].add(vertex)
                    vertices_in_regions[b].add(vertex)


        return VoronoiSolution(points, vertices, ridge_points, ridge_vertices,
                               regions)


class Context():
    def __init__(self, points):
        self.beach_line = BeachLine()
        self.points = points[np.lexsort(points.T)]
        self.line = y(self.points[0])
        self.voronoi = Voronoi(self.points)

    def plot(self):
        self.beach_line.plot(self)
        plt.scatter(self.points[:, 0], self.points[:, 1])
        vertices = np.array(self.voronoi.vertices)
        if len(vertices) > 0:
            plt.scatter(vertices[:, 0], vertices[:, 1])
        plt.gca().axhline(self.line)

    def finalize(self):
        return self.voronoi.build(self)
class BreakPoint():

    def __init__(self, arc_left, arc_right):
        self.left_pid = arc_left.pid
        self.right_pid = arc_right.pid

    def compute_intersection(self, context):
        p1 = context.points[self.left_pid]
        p2 = context.points[self.right_pid]
        return arc_intersection(p1, p2, context.line)

    def __repr__(self):
        return f"[{self.left_pid}/{self.right_pid}]"

    def __str__(self):
        return repr(self)

class CircleEvent:
    def __init__(self, left, middle, right, context):
        self.left = left
        self.middle = middle
        self.right = right
        if self.middle.arc.die_event is not None:
            self.middle.arc.die_event.abort_event()
        self.center = np.zeros(2, dtype='float64')
        a, b, c = (context.points[self.left.arc.pid],
                   context.points[self.middle.arc.pid],
                   context.points[self.right.arc.pid])
        self.center, self.radius = circle_from_points(a, b, c)

        wrong_turn = ((x(b)-x(a))*(y(c)-y(a)) - (x(c)-x(a))*(y(b)-y(a)) <= 0)

        self.aborted = self.radius < 0 or wrong_turn
        if not self.aborted:
            # self.middle.arc.die_event.abort_event()
            #
            self.middle.arc.die_event = self

            print("----------AFTER", self.middle.arc, id(self.middle.arc), self.middle.arc.die_event)

    def __gt__(self, other):
        return x(self.center) > x(other.center)

    @property
    def top_y(self):
        return y(self.center) + self.radius

    def __repr__(self):
        return f"CircleEvent({self.left.arc.pid}, {self.middle.arc.pid}, {self.right.arc.pid})"

    def __str__(self):
        return repr(self)

    def plot(self, context):
        if not self.aborted:
            c = plt.Circle((self.center), self.radius, fill=False)
            plt.gca().add_artist(c)


    def handle(self, context):
        if self.aborted:
            print("### I have been aborted")
            return []
        context.beach_line.remove_arc_node(self.middle)
        a = self.left.check_for_adjacent_circle_events(context)
        b = self.right.check_for_adjacent_circle_events(context)

        vertex_id = context.voronoi.add_vertex(self.center)
        context.voronoi.fill_edge_endpoint(self.left.arc.pid, self.middle.arc.pid, vertex_id)
        context.voronoi.fill_edge_endpoint(self.left.arc.pid, self.right.arc.pid, vertex_id)
        context.voronoi.fill_edge_endpoint(self.middle.arc.pid, self.right.arc.pid, vertex_id)

        total = a + b
        return total

    def to_tuple(self, context):
        return (self.top_y, self)


class BeachLine():
    
    def __init__(self):
        self.left = None

    # Linear search
    def find_arc(self, x, context):
        previous_arc = self.left
        while True:
            if previous_arc.right_node is None:
                break
            breakpoint = previous_arc.right_node
            x_intersection = breakpoint.compute_intersection(context)
            if x_intersection >= x:
                return previous_arc
            else:
                previous_arc = breakpoint.right_node
        return previous_arc

    @property
    def end_node(self):
        cur = self.left
        while cur.right_node is not None:
            cur = cur.right_node
        return cur

    def insert_arc(self, arc, context):
        x_arc = x(context.points[arc.pid])
        new_node = BreakPointNode(arc=arc)
        if self.left is None:
            self.left = new_node
        else:
            previous_arc_node = self.find_arc(x_arc, context)
            left_arc_node, right_arc_node = previous_arc_node.split()
            if previous_arc_node is self.left:
                self.left = left_arc_node
            br_left = BreakPointNode(left=left_arc_node, right=new_node)
            br_right = BreakPointNode(left=new_node, right=right_arc_node)

            left_arc_node.right_node = br_left
            left_arc_node.left_node = previous_arc_node.left_node

            new_node.left_node = br_left
            new_node.right_node = br_right

            right_arc_node.left_node = br_right
            right_arc_node.right_node = previous_arc_node.right_node

            try:
                previous_arc_node.left_node.right_node = left_arc_node
            except:
                pass
            try:
                previous_arc_node.right_node.left_node = right_arc_node
            except:
                pass

        return new_node

    def basic_removal(self, node):
        if node.left_node is not None:
            node.left_node.right_node = node.right_node

        if node.right_node is not None:
            node.right_node.left_node = node.left_node

    def remove_arc_node(self, arc_node):
        print("# removing", arc_node)
        arc_node.arc.abort_event()
        print("~~~~event aborted", arc_node.arc.die_event)
        arc_node.left_node.left_node.arc.abort_event()
        arc_node.right_node.right_node.arc.abort_event()

        left_node = arc_node.left_node
        right_node = arc_node.right_node

        self.basic_removal(arc_node)
        if left_node.arc is None and right_node.arc is None:  # two break
            self.basic_removal(left_node)


    def plot(self, context):
        cur = self.left
        while cur is not None:
            cur.plot(context)
            cur = cur.right_node

    def __repr__(self):
        cur = self.left
        result = ""
        while cur is not None:
            result += repr(cur)
            cur = cur.right_node
        if result != self.inv_rep():
            print(result, self.inv_rep())
            assert False
        return result

    def inv_rep(self):
        cur = self.end_node
        result = ""
        while cur is not None:
            result = repr(cur) + result
            cur = cur.left_node
        return result

    def __str__(self):
        return repr(self)


class BreakPointNode():
    def __init__(self, arc=None, left=None, right=None):
        self.arc = arc
        self.left_node = left
        self.right_node = right

    def compute_intersection(self, context):
        assert self.arc is None
        p1 = context.points[self.left_node.arc.pid]
        p2 = context.points[self.right_node.arc.pid]
        return arc_intersection(p1, p2, context.line + 1e-10)

    def plot(self, context):
        if self.is_arc_node:
            x_min = -1
            x_max = 2
            if self.left_node is not None:
                try:
                    x_min = self.left_node.compute_intersection(context)
                except:
                    pass
            if self.right_node is not None:
                try:
                    x_max = self.right_node.compute_intersection(context)
                except:
                    pass
            self.arc.plot(context, x_min, x_max)


    @property
    def base_rep(self):
        if self.arc is not None:
            return repr(self.arc)
        else:
            return ""
            return f"[{self.left_node.arc.pid}/{self.right_node.arc.pid}]"

    def __repr__(self):
        return self.base_rep

    def split(self):
        assert self.is_arc_node
        left, right = self.arc.split()
        return BreakPointNode(arc=left), BreakPointNode(arc=right)


    def __str__(self):
        return self.base_rep

    @property
    def is_arc_node(self):
        return self.arc is not None

    def check_for_adjacent_circle_events(self, context):
        # Making sure are are at a arc node
        assert self.is_arc_node
        events = []
        try:
            left = self.left_node.left_node
            left2 = left.left_node.left_node
            event = CircleEvent(left2, left, self, context)
            events.append(event)
        except Exception as e:
            pass
        try:
            right = self.right_node.right_node
            right2 = right.right_node.right_node
            event = CircleEvent(self, right, right2, context)
            events.append(event)
        except Exception as e:
            pass

        to_add = [e for e in events if not e.aborted]
        print(to_add)
        return to_add


class Arc():
    def __init__(self, pid):
        self.pid = pid
        self.die_event = None

    def __repr__(self):
        return f"({self.pid})"

    def __str__(self):
        return repr(self)
    
    def plot(self, context, x_min, x_max):
        xes = np.linspace(x_min, x_max, 100)
        point = context.points[self.pid]
        try:
            if y(point) == context.line:
                plt.axvline(x(point))
            else:
                a, b, c = compute_coefs(point, context.line)
                yes = xes**2 * a + xes * b + c
                plt.plot(xes, yes, alpha=0.5)
        except:
            pass


    def split(self):
        self.abort_event()
        return Arc(self.pid), Arc(self.pid)

    def abort_event(self):
        if self.die_event is not None:
            print("aborring", self.die_event, "on", self, id(self))
            self.die_event.aborted = True
        self.die_event = None


class SiteEvent:
    def __init__(self, pid):
        self.pid = pid

    def to_tuple(self, context):
        return (y(context.points[self.pid]), self)

    def handle(self, context):
        new_events = []
        new_arc = Arc(self.pid)

        new_node = context.beach_line.insert_arc(new_arc, context)
        new_events = new_node.check_for_adjacent_circle_events(context)

        return new_events

    def __gt__(self, other):
        return True

    def __repr__(self):
        return f"SiteEvent({self.pid})"

    def __str__(self):
        return repr(self)

class EventQueue:

    def __init__(self, context):
        self.context = context
        self.events = []
        for pid in range(len(context.points)):
            event = SiteEvent(pid)
            heappush(self.events, event.to_tuple(context))

    def plot(self):
        plt.cla()
        ax = plt.gca()
        plt.xlim(-2, 3)
        plt.ylim(-2, 3)
        self.context.plot()
        for _, ev in self.events:
            if isinstance(ev, CircleEvent):
                ev.plot(self.context)

        ax.set_aspect('equal')
        print("## after", self.context.beach_line)
        plt.waitforbuttonpress(timeout=-1)

    def handle_next(self):
        y, event = heappop(self.events)
        print("# handling", event)
        self.context.line = y
        new_events = event.handle(self.context)
        for new_event in new_events:
            tuple_to_add = new_event.to_tuple(self.context)
            # print("Adding to the queu", tuple_to_add)
            heappush(self.events, new_event.to_tuple(self.context))

    def __len__(self):
        return len(self.events)

    def __repr__(self):
        return repr(list(self.events))

    def __str__(self):
        return repr(self)



def walk_tree_recursive(node, state):
    if node is None or node in state:
        return
    state[node] = [node.left_node, node.right_node, node.parent]
    if None in state[node]:
        state[node].remove(None)
    for n in state[node]:
        walk_tree_recursive(n, state)

def walk_tree(root):
    from graphviz import Digraph
    dot = Digraph()
    state = {}
    walk_tree_recursive(root, state)
    for k in state.keys():
        dot.node(str(id(k)), str(k))
    for k, n in state.items():
        for i in n:
            dot.edge(str(id(k)), str(id(i)))
    return dot
