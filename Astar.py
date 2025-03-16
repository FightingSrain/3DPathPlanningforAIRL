"""
A_star 2D
@author: huiming zhou
"""

import os
import sys
import math
import heapq
import numpy as np
from config import config
from map_3D import generate_map, render_map


class Env:
    def __init__(self, config, map):
        self.x_range = config.vis_map_size[0]  # size of background
        self.y_range = config.vis_map_size[1]
        self.z_range = config.vis_map_size[2]
        self.map = map
        self.motions = [(1, 0, 0),
                        (-1, 0, 0),
                        (0, 1, 0),
                        (0, -1, 0),
                        (0, 0, 1),
                        (0, 0, -1),
                        (1, 1, 0),
                        (-1, -1, 0),
                        (-1, 1, 0),
                        (1, -1, 0)]
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        z = self.z_range
        obs = set()
        for i in range(5, x + 6):
            for j in range(5, y + 6):
                for k in range(5, z + 6):
                    if self.map[i, j, k] == 1:
                        obs.add((i, j, k))

        return obs


class AStar:
    """AStar set the cost + heuristics as the priority
    """

    def __init__(self, s_start, s_goal, map, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.map = map
        self.Env = Env(config, self.map)  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles
        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))
        # print(">>>>>>>>")
        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break
            F1 = 0
            for s_n in self.get_neighbor(s, self.obs, config):
                new_cost = self.g[s] + self.cost(s, s_n, F1)
                F1 += 1
                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))
        # paths = []
        # for poi in reversed(self.extract_path(self.PARENT)):
        #     paths.append(list(poi))
        # return paths, self.CLOSED
        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break
            F1 = 0
            for s_n in self.get_neighbor(s, self.obs, config):
                new_cost = g[s] + self.cost(s, s_n, F1)
                F1 += 1

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s, obs, config):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """
        neig = []
        flg = 0
        for u in self.u_set:
            temp = []
            if flg >= 6:
                if ((s[0] + u[0], s[1], s[2]) not in obs) or ((s[0], s[1] + u[1], s[2]) not in obs):
                    for i in range(3):
                        if s[i] + u[i] >= config.boundary and s[i] + u[i] < config.map_size[0] - config.boundary:
                            temp.append(s[i] + u[i])
                    if len(temp) == 3:
                        neig.append(tuple(temp))
            else:

                for i in range(3):
                    if s[i] + u[i] >= config.boundary and s[i] + u[i] < config.map_size[0] - config.boundary:
                        temp.append(s[i] + u[i])
                if len(temp) == 3:
                    neig.append(tuple(temp))

            flg += 1

        return neig
        # return [(s[0] + u[0], s[1] + u[1], s[2] + u[2]) for u in self.u_set]

    def cost(self, s_start, s_goal, flg):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal, flg):
            return math.inf
        hypot = np.sqrt(np.square(s_goal[0] - s_start[0]) +
                        np.square(s_goal[1] - s_start[1]) +
                        np.square(s_goal[2] - s_start[2]))
        return hypot

    def distence(self, x1, x2, y1, y2):
        return np.sqrt(np.square(x1 - x2) +
                       np.square(y1 - y2))

    def is_collision(self, s_start, s_end, flg):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        # print(s_start)
        # print(s_end)
        # print("GGGGG")
        # if flg >= 6:
        #     if s_end not in self.obs:


        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True
        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node
        # print(goal)
        # print(s)
        # print("JJJJJJJJJJJJJJ")

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1]) + abs(goal[2] - s[2])
        else:
            hypot = np.sqrt(np.square(goal[0] - s[0]) +
                            np.square(goal[1] - s[1]) +
                            np.square(goal[2] - s[2]))
            return hypot


def main():
    # s_start = (5, 5, 5)
    # s_goal = (29, 29, 29)
    map, start, end = generate_map(config.vis_map_size, config.obs_num,
                                   config.obs_size_min, config.obs_size_max,
                                   is_random=True)
    # print(start)
    s_start = (start[0], start[1], start[2])
    s_goal = (end[0], end[1], end[2])
    print(s_start)
    print(s_goal)
    print("===============")
    astar = AStar(s_start, s_goal, map, "euclidean")
    # plot = plotting.Plotting(s_start, s_goal)

    path, visited = astar.searching()
    # path, visited = astar.searching_repeated_astar(2.5)
    posx, posy, posz = [], [], []
    for i in range(len(path)):
        posx.append(path[i][0] + 0.5 - config.boundary)
        posy.append(path[i][1] + 0.5 - config.boundary)
        posz.append(path[i][2] + 0.5 - config.boundary)
    render_map(map, np.asarray(posx),
               np.asarray(posy),
               np.asarray(posz), start, end)
    print(path)
    print(path[-2])
    # print(visited)
    # plot.animation(path, visited, "A*")  # animation

    # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
    # plot.animation_ara_star(path, visited, "Repeated A*")


# if __name__ == '__main__':
#     main()
