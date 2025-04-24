import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import heapq

# Класс для представления рёбер
class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight

# Система непересекающихся множеств (Union-Find)
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return False
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1
        return True

# Алгоритм Крускала
def kruskal(n, edges):
    mst = []
    total_weight = 0
    dsu = DSU(n)
    for edge in sorted(edges, key=lambda e: e.weight):
        if dsu.union(edge.u, edge.v):
            mst.append(edge)
            total_weight += edge.weight
    return mst, total_weight

# Полный перебор
def brute_force_mst(n, edges):
    min_weight = float('inf')
    best_tree = []
    for combo in combinations(edges, n - 1):
        G = nx.Graph()
        G.add_weighted_edges_from([(e.u, e.v, e.weight) for e in combo])
        if nx.is_connected(G) and len(G.nodes) == n:
            w = sum(e.weight for e in combo)
            if w < min_weight:
                min_weight = w
                best_tree = combo
    return list(best_tree), min_weight

# Алгоритм Прима
def prim(n, edges):
    # строим список смежности
    adj = {i: [] for i in range(n)}
    for e in edges:
        adj[e.u].append((e.weight, e.v))
        adj[e.v].append((e.weight, e.u))
    visited = [False] * n
    mst = []
    total_weight = 0
    # начинаем с вершины 0
    visited[0] = True
    heap = []
    for w, v in adj[0]:
        heapq.heappush(heap, (w, 0, v))
    while heap and len(mst) < n - 1:
        w, u, v = heapq.heappop(heap)
        if visited[v]:
            continue
        visited[v] = True
        mst.append(Edge(u, v, w))
        total_weight += w
        for w2, v2 in adj[v]:
            if not visited[v2]:
                heapq.heappush(heap, (w2, v, v2))
    return mst, total_weight

# Пример графа
edges = [
    Edge(0, 1, 10),
    Edge(0, 2, 6),
    Edge(0, 3, 5),
    Edge(1, 3, 15),
    Edge(2, 3, 4)
]

mst, weight = kruskal(4, edges)
print("Рёбра в MST:")
for edge in mst:
    print(f"{edge.u} — {edge.v} (вес: {edge.weight})")

print(f"Суммарный вес: {weight}")

# Запуск алгоритмов
mst_kruskal, w_kruskal = kruskal(4, edges)
mst_brute, w_brute = brute_force_mst(4, edges)
mst_prim, w_prim = prim(4, edges)

# Визуализация
G = nx.Graph()
G.add_weighted_edges_from([(e.u, e.v, e.weight) for e in edges])
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(18, 6))

# Крускал
plt.subplot(1, 3, 1)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='lightgray', node_size=600)
nx.draw_networkx_edges(
    nx.Graph([(e.u, e.v) for e in mst_kruskal]),
    pos, edge_color='green', width=2
)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(e.u, e.v): e.weight for e in edges})
plt.title(f"Крускал, вес = {w_kruskal}")

# Перебор
plt.subplot(1, 3, 2)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='lightgray', node_size=600)
nx.draw_networkx_edges(nx.Graph([(e.u, e.v) for e in mst_brute]), pos, edge_color='red', width=2)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(e.u, e.v): e.weight for e in edges})
plt.title(f"Перебор, вес = {w_brute}")

# Прим
plt.subplot(1, 3, 3)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='lightgray', node_size=600)
nx.draw_networkx_edges(nx.Graph([(e.u, e.v) for e in mst_prim]), pos, edge_color='blue', width=2)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(e.u, e.v): e.weight for e in edges})
plt.title(f"Прим, вес = {w_prim}")

plt.tight_layout()
plt.show()