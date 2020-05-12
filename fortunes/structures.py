from collections import namedtuple
from utils import arc_intersection, x, y

Context = namedtuple("Context", ["points", "line"])

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


class BreakPointNode():
    def __init__(self, breakpoint=None, arc=None, left=None, right=None):
        self.breakpoint = breakpoint
        self.arc = arc
        self.left_node = left
        self.right_node = right

    def find_arc_node(self, x, context):
        if self.breakpoint is None:
            return self
        else:
            x_intersection = self.breakpoint.compute_intersection(context)
            children = self.left_node if x < x_intersection else self.right_node
            return children.find_arc_node(x, context)

    @property
    def base_rep(self):
        if self.breakpoint is not None:
            return repr(self.breakpoint)
        else:
            return repr(self.arc)

    def __repr__(self):
        return self.base_rep

    def full_rep(self):
        result = self.base_rep
        if self.breakpoint is not None:
            if self.left_node is not None:
                result = repr(self.left_node) + result
            if self.right_node is not None:
                result = result + repr(self.right_node)
        return result

    def __str__(self):
        return self.base_rep

    def insert_arc(self, arc, context):
        x_arc = x(context.points[arc.pid])
        arc_node = self.find_arc_node(x_arc, context)
        left_leaf = arc_node.left_node
        right_leaf = arc_node.right_node
        intersecting_arc = arc_node.arc
        left_br = BreakPoint(intersecting_arc, arc)
        right_br_node = BreakPointNode(breakpoint=BreakPoint(arc, intersecting_arc))
        arc_node.arc = None
        arc_node.breakpoint = left_br
        arc_node.left_node = BreakPointNode(arc=intersecting_arc, left=left_leaf)
        arc_node.right_node = right_br_node
        right_br_node.left_node = BreakPointNode(arc=arc,left=arc_node.left_node)
        arc_node.left_node.right_node = right_br_node.left_node
        right_br_node.right_node = BreakPointNode(arc=intersecting_arc,
                                                  left=right_br_node.left_node,
                                                  right=right_leaf)
        right_br_node.left_node.right_node = right_br_node.right_node


class Arc():
    def __init__(self, pid):
        self.pid = pid

    def __repr__(self):
        return f"({self.pid})"

    def __str__(self):
        return repr(self)

def walk_tree_recursive(node, state):
    if node in state:
        return
    state[node] = set([node.left_node, node.right_node])
    if None in state[node]:
        state[node].remove(None)
    print(state[node])
    for n in state[node]:
        walk_tree_recursive(n, state)

def walk_tree(root):
    state = {}
    walk_tree_recursive(root, state)
    return state
