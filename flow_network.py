import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import string
import time

class FlowNetwork:
    def __init__(self, graph, source, sink):
        self.graph = np.array(graph, dtype=float)
        self.source = source
        self.sink = sink
        self.num_vertices = len(graph)
        self.residual_graph = self.graph.copy()
        self.flow = np.zeros_like(self.graph)

    def bfs(self):
        visited = [False] * self.num_vertices
        parent = [-1] * self.num_vertices
        queue = deque([self.source])
        visited[self.source] = True

        while queue:
            u = queue.popleft()
            for v in range(self.num_vertices):
                if not visited[v] and self.residual_graph[u][v] > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u

        if visited[self.sink]:
            path = []
            s = self.sink
            while s != self.source:
                path.append(s)
                s = parent[s]
            path.append(self.source)
            path.reverse()
            return path
        return None

    def ford_fulkerson(self):
        path = self.bfs()
        while path:
            min_capacity = min(self.residual_graph[path[i]][path[i+1]] for i in range(len(path) - 1))
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.residual_graph[u][v] -= min_capacity
                self.residual_graph[v][u] += min_capacity
                self.flow[u][v] += min_capacity
            path = self.bfs()
        return sum(self.flow[self.source])


def generate_random_flow_network(num_vertices, min_capacity=1, max_capacity=10, density=0.3):
    graph = np.zeros((num_vertices, num_vertices))
    vertices = list(range(num_vertices))
    source = random.choice(vertices)
    sink = random.choice([v for v in vertices if v != source])

    path_vertices = [source]
    available = [v for v in vertices if v != source and v != sink]
    if available:
        intermediate = random.sample(available, min(len(available), random.randint(0, len(available))))
        path_vertices.extend(intermediate)
    path_vertices.append(sink)

    for i in range(len(path_vertices) - 1):
        graph[path_vertices[i]][path_vertices[i + 1]] = random.randint(min_capacity, max_capacity)

    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j and graph[i][j] == 0 and random.random() < density:
                graph[i][j] = random.randint(min_capacity, max_capacity)

    return graph, source, sink


def is_path_exists(graph, source, sink):
    n = len(graph)
    visited = [False] * n

    def dfs(u):
        if u == sink:
            return True
        visited[u] = True
        for v in range(n):
            if graph[u][v] > 0 and not visited[v]:
                if dfs(v):
                    return True
        return False

    return dfs(source)


def get_node_labels(num_vertices):
    labels = {}
    uppercase = list(string.ascii_uppercase)
    lowercase = list(string.ascii_lowercase)
    for i in range(num_vertices):
        if i < len(uppercase):
            labels[i] = uppercase[i]
        elif i < len(uppercase) + len(lowercase):
            labels[i] = lowercase[i - len(uppercase)]
        else:
            primary = (i - len(uppercase) - len(lowercase)) // len(uppercase) + 1
            secondary = (i - len(uppercase) - len(lowercase)) % len(uppercase)
            labels[i] = uppercase[primary - 1] + uppercase[secondary]
    return labels


def visualize_network(graph, source, sink, flow=None, max_flow=None):
    plt.figure(figsize=(12, 10))
    G = nx.DiGraph()
    labels = get_node_labels(len(graph))

    for i in range(len(graph)):
        G.add_node(i, label=labels[i])

    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i][j] > 0:
                G.add_edge(i, j, capacity=graph[i][j], label=f"{flow[i][j]:.0f}/{graph[i][j]:.0f}" if flow is not None else f"{graph[i][j]:.0f}")

    pos = nx.kamada_kawai_layout(G)
    node_colors = ['lightgreen' if n == source else 'lightblue' if n == sink else 'lightgray' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, edgecolors='black')
    nx.draw_networkx_edges(G, pos, arrowsize=20, width=1.5)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes()}, font_size=16, font_weight='bold')

    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=14, font_color='darkred', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    title = f"Flow Network (Source: {labels[source]}, Sink: {labels[sink]})"
    if max_flow is not None:
        title += f" - Max Flow: {max_flow:.0f}"
    plt.title(title, fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def print_network_info(graph, source, sink, labels, max_flow=None):
    print(f"\nSource: {labels[source]}, Sink: {labels[sink]}")
    if max_flow is not None:
        print(f"Maximum Flow: {max_flow}")
    print(f"Path exists: {is_path_exists(graph, source, sink)}\n")
    print("Capacities:")
    header = "    " + " ".join(f"{labels[j]:>4}" for j in range(len(graph)))
    print(header)
    for i in range(len(graph)):
        row = f"{labels[i]:2} " + " ".join(f"{graph[i][j]:4.0f}" for j in range(len(graph)))
        print(row)



class FlowNetwork:
    def __init__(self, graph, source, sink):
        self.graph = np.array(graph, dtype=float)
        self.source = source
        self.sink = sink
        self.num_vertices = len(graph)
        self.residual_graph = self.graph.copy()
        self.flow = np.zeros_like(self.graph)

    def bfs(self):
        visited = [False] * self.num_vertices
        parent = [-1] * self.num_vertices
        queue = deque([self.source])
        visited[self.source] = True

        while queue:
            u = queue.popleft()
            for v in range(self.num_vertices):
                if not visited[v] and self.residual_graph[u][v] > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u

        if visited[self.sink]:
            path = []
            s = self.sink
            while s != self.source:
                path.append(s)
                s = parent[s]
            path.append(self.source)
            path.reverse()
            return path
        return None

    def ford_fulkerson(self):
        start_time = time.time()  # Start the timer

        path = self.bfs()
        while path:
            min_capacity = min(self.residual_graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.residual_graph[u][v] -= min_capacity
                self.residual_graph[v][u] += min_capacity
                self.flow[u][v] += min_capacity
            path = self.bfs()

        end_time = time.time()  # End the timer
        execution_time = end_time - start_time  # Calculate execution time

        return sum(self.flow[self.source]), execution_time  # Return both max flow and execution time
