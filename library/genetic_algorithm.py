import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# 今回は便宜上始点と終点を 0 に固定します
class GATSP:
    def __init__(self, n=10):
        """
        Graph の頂点数を指定して Instance を生成
        :param n:
        """
        self.N = n

    def generate_nodes(self):
        """
        最初に頂点を生成する
        :return:
        """
        np.random.seed(0)
        self.nodes = np.random.uniform(size=(self.N, 2))
        self._dist = np.linalg.norm(
            self.nodes[:, None] - self.nodes[None, :],
            axis=-1,
        )

    @classmethod
    def from_csv(cls, path):
        nodes = pd.read_csv(path).values
        n = nodes.shape[0]
        tsp = cls(n)
        tsp._dist = np.linalg.norm(
            nodes[:, None] - nodes[None, :],
            axis=-1,
        )
        tsp.nodes = nodes
        return tsp

    def generate_route(self):
        """
        個体、つまり１つの経路を Random に生成する
        :return:
        """
        return np.random.permutation(np.arange(1, self.N))

    @staticmethod
    def routes_from_csv(path):
        routes = pd.read_csv(path).values
        return routes

    def init_routes(self, m=100):
        """
        初期世代の個体群（ｍ個体）を生成。
        各個体を生成する generate_route() を使用している
        :param m:
        :return:
        """
        routes = np.array([self.generate_route() for _ in range(m)])
        return np.pad(routes, pad_width=((0, 0), (1, 0)), constant_values=0)

    def dist(self, routes):
        """
        個体群を渡すと、各個体の経路長を一括で計算
        個体群（経路群）に対して各経路の総距離を一括で計算して返す。
        :param routes:
        :return:
        """
        routes = np.pad(routes, pad_width=((0, 0), (0, 1)), constant_values=0)
        return self._dist[routes[:, :-1], routes[:, 1:]].sum(axis=1)

    def fitness(self, routes):
        """
        個体の適応度を計算する際に使用
        :param routes:
        :return:
        """
        return 1 / self.dist(routes)

    def select_parents(self, routes, m=None):
        """
        交叉のときに親個体を選ぶ際に使用
        :param routes:
        :param m:
        :return:
        """
        if m is None: m = routes.shape[0] // 2
        assert 2 * m <= routes.shape[0]
        f = self.fitness(routes)
        p = f / f.sum()
        pair = np.random.choice(routes.shape[0], (m, 2), replace=True, p=p)
        i = np.argsort(routes, axis=1)
        return routes[pair], i[pair]

    def crossover(self, routes, m=None):
        """
        交叉を行なう
        :param routes:
        :param m:
        :return:
        """
        if m is None: m = routes.shape[0] // 2
        parents, i = self.select_parents(routes, m)
        for j in range(m):  # ペアごとに交叉
            k = np.random.randint(1, self.N - 1)
            parents[j, np.arange(2), i[j, np.arange(2), parents[j, ::-1, k]]], parents[j, :, k] \
                = parents[j, :, k], parents[j, np.arange(2), i[j, np.arange(2), parents[j, ::-1, k]]]
        childs = parents.reshape(-1, self.N)
        return childs

    def mutate(self, routes, p=0.7):
        """
        突然変異を行なう
        :param routes:
        :param p:
        :return:
        """
        m = routes.shape[0]
        bl = np.random.choice((0, 1), m, replace=True, p=(1 - p, p)).astype(bool)  # 突然変異を起こす確率を指定
        k = np.arange(m)[bl]
        i, j = np.random.randint(1, self.N - 1, (m, 2))[bl].T
        routes[k, i], routes[k, j] = routes[k, j], routes[k, i]
        return routes

    def extract_elites(self, routes, elite_cnt):
        """
        選択を行なう
        :param routes:
        :param elite_cnt:
        :return:
        """
        return routes[np.argsort(self.fitness(routes))[-elite_cnt:]]

    def generate_nxt(self, routes, elite_cnt=2):
        """
        次世代の経路群を生成する関数
        cross, mutate, extract_elites を使用して次世代の個体群を生成する
        :param routes:
        :param elite_cnt:
        :return:
        """
        elites = self.extract_elites(routes, elite_cnt)
        childs = self.crossover(routes, m=(routes.shape[0] - elite_cnt) // 2)
        childs = self.mutate(childs)
        return np.vstack([elites, childs])

    def show(self, routes):
        """
        個体群を渡すとその中で最も優秀な（経路長が短い）個体を可視化する
        :param routes:
        :return:
        """
        path = list(routes[np.argsort(self.dist(routes))][0]) + [0]
        plt.figure(figsize=(15, 10))
        g = nx.DiGraph()
        g.add_nodes_from(range(self.N))
        pos = dict(enumerate(zip(self.nodes[:, 0], self.nodes[:, 1])))
        nx.draw_networkx(g, pos=pos, node_color='c')
        for i in range(len(path) - 1):
            g.add_edge(path[i], path[i + 1])
        nx.draw_networkx(g, pos=pos, node_color='c')
        plt.show()
        plt.clf()
