import pybullet as p
import random
import math
import numpy as np

class RRT_Node:
    def __init__(self, conf):  # conf is (x, y, theta)
        self.conf = conf
        self.parent = None
        self.children = []
        self.cost = float("inf")  # cost from start to this node

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
        # Only consider (x,y) distance for simplicity.
        dist = math.dist((node.conf[0], node.conf[1]),
                         (rand_node.conf[0], rand_node.conf[1]))
        if dist < min_distance:
            min_distance = dist
            nearest_node = node
    return nearest_node

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

def check_collision_path(p1, p2, obstacles, robot_dims, resolution=0.2):
    """
    Checks collision along the line from p1 to p2 by sampling multiple intermediate configurations.
    """
    dist = math.dist((p1[0], p1[1]), (p2[0], p2[1]))
    steps = int(dist / resolution)
    for i in range(steps + 1):
        alpha = i / steps if steps > 0 else 0
        x = p1[0] + alpha * (p2[0] - p1[0])
        y = p1[1] + alpha * (p2[1] - p1[1])
        theta = p1[2] + alpha * (p2[2] - p1[2])
        if is_box_collision((x, y, theta), obstacles, robot_dims):
            return True
    return False

def is_box_collision(conf, obstacles, robot_dims):
    """
    Checks whether the robot, modeled as a box with given dimensions,
    placed at conf = (x, y, theta), collides with any obstacle.
    
    The robot's box is defined with:
      robot_dims = (length, width, height)
    """
    x, y, theta = conf
    length, width, _ = robot_dims  # height is not used in a 2D check.
    half_length = length / 2.0
    half_width  = width  / 2.0
    # Define the four corners of the robot in its local frame.
    corners = [
        [ half_length,  half_width],
        [ half_length, -half_width],
        [-half_length, -half_width],
        [-half_length,  half_width]
    ]
    # Rotation matrix to transform local corners to world frame.
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta,  cos_theta]])
    corners_world = [R.dot(np.array(corner)) + np.array([x, y]) for corner in corners]
    corners_world = np.array(corners_world)
    # Compute axis-aligned bounding box (AABB) of the robot.
    min_xy = np.min(corners_world, axis=0)
    max_xy = np.max(corners_world, axis=0)
    # For each obstacle, obtain its AABB and check for overlap in x and y.
    for obs in obstacles:
        obs_min, obs_max = p.getAABB(obs)
        if (max_xy[0] >= obs_min[0] and min_xy[0] <= obs_max[0] and
            max_xy[1] >= obs_min[1] and min_xy[1] <= obs_max[1]):
            return True
    return False

def extract_path(goal_node):
    path = []
    current = goal_node
    while current:
        path.append(current.conf)
        current = current.parent
    path.reverse()
    return path

def run_rrt_star(start_conf, goal_conf, obstacles, maze_bounds, robot_dims,
                 max_iter=20000, goal_threshold=0.6, step_size=0.5):
    """
    Run RRT* planner to find a path from start_conf to goal_conf.

    Parameters:
      start_conf, goal_conf: tuples (x, y, theta)
      obstacles: list of PyBullet body ids to check collisions against.
      maze_bounds: (min_x, max_x, min_y, max_y)
      robot_dims: (length, width, height) of the robot's box.
      goal_threshold: distance threshold (in x,y) to consider the goal reached.
      step_size: maximum step distance for new nodes.
    """
    start_node = RRT_Node(start_conf)
    start_node.cost = 0.0
    tree = [start_node]
    
    for _ in range(max_iter):
        # Sample a new configuration
        rand_conf = sample_conf(goal_conf, maze_bounds)
        rand_node = RRT_Node(rand_conf)
        
        # Find nearest node in the tree
        nearest_node = find_nearest(rand_node, tree)
        new_node = steer_to(rand_conf, nearest_node, step_size)
        
        # Skip if the path from nearest to new node is in collision.
        if check_collision_path(nearest_node.conf, new_node.conf, obstacles, robot_dims):
            continue
        
        # Initialize new_node's cost and set nearest_node as parent.
        new_node.cost = nearest_node.cost + math.dist(nearest_node.conf[:2], new_node.conf[:2])
        new_node.set_parent(nearest_node)
        
        # Define a neighbor radius (could be made adaptive; here we use a constant)
        neighbor_radius = step_size * 2.0
        neighbors = [node for node in tree 
                     if math.dist(node.conf[:2], new_node.conf[:2]) < neighbor_radius]
        
        # Choose the best parent from neighbors (if collision-free)
        for neighbor in neighbors:
            if not check_collision_path(neighbor.conf, new_node.conf, obstacles, robot_dims):
                cost = neighbor.cost + math.dist(neighbor.conf[:2], new_node.conf[:2])
                if cost < new_node.cost:
                    new_node.cost = cost
                    new_node.set_parent(neighbor)
        
        # Add new_node to the tree.
        tree.append(new_node)
        
        # Rewire the tree: try to improve the cost to neighbors through new_node.
        for neighbor in neighbors:
            if neighbor == new_node.parent:
                continue
            if not check_collision_path(new_node.conf, neighbor.conf, obstacles, robot_dims):
                cost_through_new = new_node.cost + math.dist(new_node.conf[:2], neighbor.conf[:2])
                if cost_through_new < neighbor.cost:
                    neighbor.set_parent(new_node)
                    neighbor.cost = cost_through_new
        
        # Check if new_node is near the goal.
        if math.dist(new_node.conf[:2], goal_conf[:2]) < goal_threshold:
            print("Goal reached!")
            return extract_path(new_node)
            
    return None

