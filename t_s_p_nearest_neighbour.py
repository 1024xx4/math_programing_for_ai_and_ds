from comparison_df_and_full_search import Graph


class TSPNearestNeighbour(Graph):
    def __call__(self, src=0):
        """
        Graph という Class を定義し、巡回すべき倉庫の情報を与えることで、
        それぞれの倉庫間の移動距離を計算し、最短Route を計算する。
        """
        n = self.N
        visited = [False] * n
        visited[0] = True
        dist = 0
        u = src  # １つ前の頂点
        path = [src]
        calc_count = 0
        for _ in range(n - 1):
            cand = []
            for v in range(n):  # 次に訪れる頂点
                calc_count += 1
                if visited[v]: continue
                cand.append((v, dist + self.edges[u][v].weight))

            cand.sort(key=lambda x: x[1])
            u, dist = cand[0]  # 最も近い頂点へ移動
            visited[u] = True
            path.append(u)
        path.append(src)
        print(f"計算回数: {calc_count}")

        return dist + self.edges[u][src].weight, path
