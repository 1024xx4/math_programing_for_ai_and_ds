import networkx as nx
import numpy as np
from itertools import combinations
import pandas as pd
import random
import matplotlib.pyplot as plt

inf = float('inf')


class Graph:
    class Edge:
        def __init__(self, weight=1, **args):
            self.weight = weight

        def __repr__(self):
            return f"{self.weight}"

    def __init__(self, n):
        self.N = n
        self.edges = [{} for _ in range(n)]

    # 辺を追加
    def add_edge(self, u, v, **args):
        self.edges[u][v] = self.Edge(**args)

    @classmethod
    def from_csv(cls, path):
        nodes = pd.read_csv(path).values
        n = nodes.shape[0]
        print(f"頂点数: {n}")
        weights = cls.weights_from_nodes(nodes)

        g = cls(n)
        g.generate_network(nodes)

        for u in range(n):
            for v in range(n):
                g.add_edge(u, v, weight=weights[u, v])
        return g

    @staticmethod
    def generate_nodes(n):
        nodes = np.random.randint(low=0, high=100, size=(n, 2))
        return nodes

    def generate_network(self, nodes):
        n = len(nodes)
        network = nx.DiGraph()
        network.add_nodes_from(range(n))
        pos = dict(
            enumerate(zip(nodes[:, 0], nodes[:, 1]))
        )
        nx.draw_networkx(network, pos=pos, node_color='c')
        self.network = network
        self.pos = pos
        return network

    @staticmethod
    def weights_from_nodes(nodes):
        return np.linalg.norm(
            nodes[:, None] - nodes[None, :], axis=-1,
        ).astype(np.int64)

    # Random に辺を生成する関数(csv 以外の Pattern も作成したい場合に使用)
    def generate_edges(self):
        random.seed(0)
        for u, v in combinations(range(self.N), 2):
            weight = random.randint(1, 100)
            self.add_edge(u, v, weight=weight)
            self.add_edge(v, u, weight=weight)
        for u in range(self.N):
            self.add_edge(u, u, weight=0)

    # Route の総距離を計算（全検索）
    def calculate_dist(self, route):
        n = self.N
        source = route[0]
        route += [source]
        return sum(
            self.edges[route[i]][route[i + 1]].weight
            for i in range(n)
        )

    def show_path(self, path):
        n = self.N
        network = self.network
        pos = self.pos
        for i in range(n):
            network.add_edge(path[i], path[i + 1])
        nx.draw_networkx(
            network,
            pos=pos,
            node_color='c',
        )
        plt.show()
        self.remove_edges()

    def remove_edges(self):
        network = self.network
        network.remove_edges_from(
            list(network.edges)
        )


class TSPBruteForce(Graph):
    # 全検索Algorithm（DP alogrithm との比較用に、numpy を使わず実装）
    def __call__(self, src=0):
        n = self.N
        stack = [([src], 1 << src)]
        dist = float('inf')
        calc_count = 0
        while stack:
            route, visited = stack.pop()
            if visited == (1 << n) - 1:
                calc_count += 1
                d = self.calculate_dist(route)
                if d >= dist: continue
                dist = d
                res_route = route

            for i in range(n):
                if i == src or visited >> i & 1: continue
                nxt_route = route.copy()
                nxt_route.append(i)
                stack.append((nxt_route, visited | (1 << i)))

        print(f"計算回数: {calc_count}")
        return dist, res_route


class TSPDP(Graph):
    # DP Algorithm
    def __call__(self, src=0):
        n = self.N
        dp = [[(inf, None)] * n for _ in range(1 << n)]
        dp[1][src] = (0, None)
        calc_count = 0
        for s in range(1 << n):
            for v in range(n):
                if s >> v & 1: continue
                t = s | (1 << v)  # tはsにvを追加した集合
                for u in range(n):
                    if ~s >> u & 1: continue
                    d = dp[s][u][0] + self.edges[u][v].weight
                    if d >= dp[t][v][0]:
                        continue
                    dp[t][v] = (d, u)
                    calc_count += 1

        print(f'計算回数: {calc_count}')

        dist = inf
        predecessor = []
        for u in range(1, n):
            s = (1 << n) - 1
            d = dp[s][u][0] + self.edges[u][src].weight
            if d >= dist: continue
            dist = d
            predecessor = [src]
            while True:
                v = u
                predecessor.append(v)
                u = dp[s][v][1]
                if u is None: break
                s &= ~(1 << v)

        return dist, predecessor[::-1]
