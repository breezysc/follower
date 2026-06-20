import math

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_direction(leader_x, leader_y, center_x, center_y):
    dx = leader_x - center_x
    dy = leader_y - center_y
    return dx, dy

def apply_dead_zone(dx, dy, dead_zone):
    if abs(dx) < dead_zone:
        dx = 0
    if abs(dy) < dead_zone:
        dy = 0
    return dx, dy