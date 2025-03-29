import pybullet as p
import random
import math

class RRT_Node:
    def __init__(self, conf):  # conf is (x, y, theta)
        self.conf = conf
        self.parent = None
        self.children = []

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)

def sample_conf(goal_conf, maze_bounds, goal_bias=0.05):
    """
    Sample a random configuration within maze boundaries.
    maze_bounds = (min_x, max_x, min_y, max_y)
    """
    if random.random() < goal_bias:
        return goal_conf
    else:
        min_x, max_x, min_y, max_y = maze_bounds
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        theta = random.uniform(-math.pi, math.pi)
        return (x, y, theta)

def find_nearest(rand_node, node_list):
    nearest_node = None
    min_distance = float("inf")
    for node in node_list:
        dist = math.dist((node.conf[0], node.conf[1]), (rand_node.conf[0], rand_node.conf[1]))
        if dist < min_distance:
            min_distance = dist
            nearest_node = node
    return nearest_node

def check_collision(pos, obstacles):
    ray_start = [pos[0], pos[1], 1.0]
    ray_end = [pos[0], pos[1], 0.0]
    ray = p.rayTest(ray_start, ray_end)
    hit_object = ray[0][0]
    return hit_object in obstacles

def steer_to(rand_conf, nearest_node, step_size=0.5):
    from_x, from_y, from_theta = nearest_node.conf
    to_x, to_y, to_theta = rand_conf
    dist = math.dist((from_x, from_y), (to_x, to_y))

    if dist < step_size:
        new_conf = rand_conf
    else:
        alpha = step_size / dist
        new_x = from_x + alpha * (to_x - from_x)
        new_y = from_y + alpha * (to_y - from_y)
        new_theta = from_theta + alpha * (to_theta - from_theta)
        new_conf = (new_x, new_y, new_theta)

    return RRT_Node(new_conf)

def extract_path(goal_node):
    path = []
    current = goal_node
    while current:
        path.append(current.conf)
        current = current.parent
    path.reverse()
    return path

def run_rrt(start_conf, goal_conf, obstacles, maze_bounds, max_iter=1000, goal_threshold=0.6):
    """
    Run RRT planner to find a path from start_conf to goal_conf.
    """
    start_node = RRT_Node(start_conf)
    tree = [start_node]

    for _ in range(max_iter):
        rand_conf = sample_conf(goal_conf, maze_bounds)
        rand_node = RRT_Node(rand_conf)
        nearest_node = find_nearest(rand_node, tree)
        new_node = steer_to(rand_conf, nearest_node)

        if check_collision(new_node.conf, obstacles):
            continue

        new_node.set_parent(nearest_node)
        nearest_node.add_child(new_node)
        tree.append(new_node)

        dist_to_goal = math.dist((new_node.conf[0], new_node.conf[1]), (goal_conf[0], goal_conf[1]))
        if dist_to_goal < goal_threshold:
            print("[RRT] Goal reached!")
            return extract_path(new_node)

    print("[RRT] Failed to find a path.")
    return None
