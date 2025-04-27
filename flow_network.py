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
        start_time = time.time()  # Start timing the algorithm
        path = self.bfs()

        while path:
            min_capacity = float('inf')
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                min_capacity = min(min_capacity, self.residual_graph[u][v])

            # Update residual capacities and reverse edges
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.residual_graph[u][v] -= min_capacity
                self.residual_graph[v][u] += min_capacity
                self.flow[u][v] += min_capacity

            path = self.bfs()

        max_flow = sum(self.flow[self.source])
        execution_time = time.time() - start_time

        # Ensure the flow is reflected in the graph for visualization
        G = nx.DiGraph()
        for i in range(len(self.graph)):
            for j in range(len(self.graph)):
                if self.graph[i][j] > 0:
                    G.add_edge(i, j, capacity=self.graph[i][j], flow=self.flow[i][j])

        return max_flow, execution_time

    def visualize_network(self, flow=None, max_flow=None):
        plt.figure(figsize=(12, 10))
        G = nx.DiGraph()
        n = len(self.graph)
        node_labels = get_node_labels(n)

        # Add nodes
        for i in range(n):
            G.add_node(i, label=node_labels[i])

        # Add edges with flow and capacity
        for i in range(n):
            for j in range(n):
                if self.graph[i][j] > 0:
                    if flow is not None:
                        G.add_edge(i, j, capacity=self.graph[i][j], flow=flow[i][j],
                                   label=f"{flow[i][j]:.0f}/{self.graph[i][j]:.0f}")
                    else:
                        G.add_edge(i, j, capacity=self.graph[i][j], label=f"{self.graph[i][j]:.0f}")

        # Node positioning and visualization setup
        pos = nx.kamada_kawai_layout(G)
        node_colors = ['lightgreen' if node == self.source else 'lightblue' if node == self.sink else 'lightgray' for
                       node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, edgecolors='black')
        nx.draw_networkx_edges(G, pos, arrowsize=20, width=1.5)

        # Node labels
        node_label_dict = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_label_dict, font_size=16, font_weight='bold')

        # Edge labels
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            forward_flow = data.get('flow', 0)
            backward_flow = G.edges[v, u].get('flow', 0) if G.has_edge(v, u) else 0
            net_flow = forward_flow - backward_flow
            edge_labels[(u, v)] = f"{net_flow}/{data['capacity']}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=14, font_color='darkred',
                                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),
                                     font_weight='bold', label_pos=0.5)

        source_label = node_labels[self.source]
        sink_label = node_labels[self.sink]

        # Update title to include max_flow if it's available
        if max_flow is not None:
            title = f"Flow Network (Source: {source_label}, Sink: {sink_label}) - Maximum Flow: {max_flow:.0f}"
        else:
            title = f"Flow Network (Source: {source_label}, Sink: {sink_label})"

        plt.title(title, fontsize=18)

        # If max_flow is provided, display it in a text box at the bottom of the graph
        if max_flow is not None:
            text_box = f"Maximum Flow: {max_flow:.0f}"
            plt.figtext(0.5, 0.01, text_box, fontsize=16,
                        bbox=dict(facecolor='lightgreen', edgecolor='green', boxstyle='round,pad=0.5'),
                        ha='center')

        plt.axis('off')
        plt.tight_layout()
        plt.show()


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
