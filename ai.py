from __future__ import print_function
from heapq import heappop, heappush
from collections import deque

ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


class AI:
    def __init__(self, grid, type):
        self.grid = grid
        self.set_type(type)
        self.set_search()

    def set_type(self, type):
        self.final_cost = 0
        self.type = type

    def set_search(self):
        self.final_cost = 0
        self.grid.reset()
        self.finished = False
        self.failed = False
        self.previous = {}
        self.explored = []

        if self.type == "dfs":
            self.frontier = [self.grid.start]
            self.frontier_set = {self.grid.start}
            self.explored_set = set()

        elif self.type == "bfs":
            self.frontier = deque([self.grid.start])
            self.frontier_set = {self.grid.start}
            self.explored_set = set()

        elif self.type == "ucs":
            self.frontier = []
            self.frontier_set = {self.grid.start}
            self.explored_set = set()
            self.best_cost = {self.grid.start: 0}
            heappush(self.frontier, (0, self.grid.start))

        elif self.type == "astar":
            self.frontier = []
            self.frontier_set = {self.grid.start}
            self.explored_set = set()
            self.best_cost = {self.grid.start: 0}
            start_h = self.heuristic(self.grid.start)
            heappush(self.frontier, (start_h, start_h, 0, self.grid.start))

        self.grid.nodes[self.grid.start].color_frontier = True

    def get_result(self):
        total_cost = 0
        current = self.grid.goal
        while not current == self.grid.start:
            if self.type == "bfs":
                total_cost += 1
            else:
                total_cost += self.grid.nodes[current].cost()
            current = self.previous[current]
            self.grid.nodes[current].color_in_path = True
        total_cost += self.grid.nodes[current].cost()
        self.final_cost = total_cost

    def make_step(self):
        if self.type == "dfs":
            self.dfs_step()
        elif self.type == "bfs":
            self.bfs_step()
        elif self.type == "ucs":
            self.ucs_step()
        elif self.type == "astar":
            self.astar_step()

    def get_children(self, current):
        children = []
        for dr, dc in ACTIONS:
            nxt = (current[0] + dr, current[1] + dc)
            if nxt[0] not in range(self.grid.row_range):
                continue
            if nxt[1] not in range(self.grid.col_range):
                continue
            if self.grid.nodes[nxt].puddle:
                continue
            children.append(nxt)
        return children

    def mark_explored(self, node):
        self.frontier_set.discard(node)
        self.grid.nodes[node].color_frontier = False
        self.grid.nodes[node].color_checked = True
        self.explored.append(node)
        self.explored_set.add(node)

    def heuristic(self, node):
        return abs(node[0] - self.grid.goal[0]) + abs(node[1] - self.grid.goal[1])

    # Fixed DFS
    def dfs_step(self):
        while self.frontier and self.frontier[-1] in self.explored_set:
            stale = self.frontier.pop()
            self.frontier_set.discard(stale)
            self.grid.nodes[stale].color_frontier = False

        if not self.frontier:
            self.failed = True
            self.finished = True
            print("no path")
            return

        current = self.frontier.pop()
        self.mark_explored(current)

        if current == self.grid.goal:
            self.finished = True
            return

        for n in self.get_children(current):
            if n not in self.explored_set and n not in self.frontier_set:
                self.previous[n] = current
                self.frontier.append(n)
                self.frontier_set.add(n)
                self.grid.nodes[n].color_frontier = True

    # BFS
    def bfs_step(self):
        while self.frontier and self.frontier[0] in self.explored_set:
            stale = self.frontier.popleft()
            self.frontier_set.discard(stale)
            self.grid.nodes[stale].color_frontier = False

        if not self.frontier:
            self.failed = True
            self.finished = True
            return

        current = self.frontier.popleft()
        self.mark_explored(current)

        if current == self.grid.goal:
            self.finished = True
            return

        for n in self.get_children(current):
            if n not in self.explored_set and n not in self.frontier_set:
                self.previous[n] = current
                self.frontier.append(n)
                self.frontier_set.add(n)
                self.grid.nodes[n].color_frontier = True

    # Uniform Cost Search
    def ucs_step(self):
        while self.frontier:
            current_cost, current = heappop(self.frontier)
            if current in self.explored_set:
                continue
            if current_cost != self.best_cost.get(current, float("inf")):
                continue
            break
        else:
            self.failed = True
            self.finished = True
            return

        self.mark_explored(current)

        if current == self.grid.goal:
            self.finished = True
            return

        for n in self.get_children(current):
            if n in self.explored_set:
                continue

            new_cost = current_cost + self.grid.nodes[n].cost()
            if new_cost < self.best_cost.get(n, float("inf")):
                self.best_cost[n] = new_cost
                self.previous[n] = current
                heappush(self.frontier, (new_cost, n))
                self.frontier_set.add(n)
                self.grid.nodes[n].color_frontier = True

    # A* with Manhattan distance
    def astar_step(self):
        while self.frontier:
            f_cost, h_cost, g_cost, current = heappop(self.frontier)
            if current in self.explored_set:
                continue
            if g_cost != self.best_cost.get(current, float("inf")):
                continue
            break
        else:
            self.failed = True
            self.finished = True
            return

        self.mark_explored(current)

        if current == self.grid.goal:
            self.finished = True
            return

        for n in self.get_children(current):
            if n in self.explored_set:
                continue

            new_g = g_cost + self.grid.nodes[n].cost()
            if new_g < self.best_cost.get(n, float("inf")):
                self.best_cost[n] = new_g
                self.previous[n] = current
                new_h = self.heuristic(n)
                new_f = new_g + new_h
                heappush(self.frontier, (new_f, new_h, new_g, n))
                self.frontier_set.add(n)
                self.grid.nodes[n].color_frontier = True