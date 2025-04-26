import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import string


class FlowNetwork:
    def __init__(self, graph, source, sink):
        """
        Initialize a flow network

        Args:
            graph: A 2D list or numpy array where graph[u][v] is the capacity from u to v
            source: Source vertex
            sink: Sink vertex
        """
        self.graph = np.array(graph, dtype=float)
        self.source = source
        self.sink = sink
        self.num_vertices = len(graph)

        # Residual graph is initialized with the same capacities as original graph
        self.residual_graph = self.graph.copy()

        # Flow is initially 0
        self.flow = np.zeros_like(self.graph)

    def bfs(self):
        """
        Use BFS to find an augmenting path from source to sink in the residual graph

        Returns:
            path: List of vertices forming a path from source to sink, or None if no path exists
        """
        visited = [False] * self.num_vertices
        parent = [-1] * self.num_vertices

        queue = deque()
        queue.append(self.source)
        visited[self.source] = True

        while queue:
            u = queue.popleft()

            for v in range(self.num_vertices):
                if not visited[v] and self.residual_graph[u][v] > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u

        # If we reached sink in BFS, then there is a path
        if visited[self.sink]:
            path = []
            s = self.sink
            while s != self.source:
                path.append(s)
                s = parent[s]
            path.append(self.source)
            path.reverse()
            return path
        else:
            return None

    def ford_fulkerson(self):
        """
        Implement the Ford-Fulkerson algorithm to find the maximum flow

        Returns:
            max_flow: The maximum flow from source to sink
        """
        path = self.bfs()

        # While there is an augmenting path
        while path:
            # Find the minimum residual capacity along the path
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

            # Find next augmenting path
            path = self.bfs()

        # Calculate the maximum flow
        max_flow = sum(self.flow[self.source])

        return max_flow


def generate_random_flow_network(num_vertices, min_capacity=1, max_capacity=10, density=0.3):
    """
    Generate a random flow network with at least one path from source to sink

    Args:
        num_vertices: Number of vertices in the network
        min_capacity: Minimum capacity of an edge
        max_capacity: Maximum capacity of an edge
        density: Probability of an edge existing between any two vertices

    Returns:
        graph: Adjacency matrix representing the network
        source: Source vertex index
        sink: Sink vertex index
    """
    # Initialize with zeros
    graph = np.zeros((num_vertices, num_vertices))

    # Choose source and sink
    vertices = list(range(num_vertices))
    source = random.choice(vertices)
    remaining = [v for v in vertices if v != source]
    sink = random.choice(remaining)

    # Ensure there's at least one path from source to sink
    # Create a random path from source to sink
    current = source
    path_length = min(random.randint(2, num_vertices), num_vertices - 1)

    # Create vertices in the path
    path_vertices = [source]
    available_vertices = [v for v in range(num_vertices) if v != source and v != sink]

    # Select some random intermediate vertices for the path
    if len(available_vertices) > 0 and path_length > 2:
        num_intermediate = min(path_length - 2, len(available_vertices))
        intermediate_vertices = random.sample(available_vertices, num_intermediate)
        path_vertices.extend(intermediate_vertices)

    path_vertices.append(sink)

    # Create edges along this path with random capacities
    for i in range(len(path_vertices) - 1):
        u = path_vertices[i]
        v = path_vertices[i + 1]
        graph[u][v] = random.randint(min_capacity, max_capacity)

    # Add additional random edges
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j and graph[i][j] == 0 and random.random() < density:  # Only add edge if it doesn't exist
                graph[i][j] = random.randint(min_capacity, max_capacity)

    return graph, source, sink


def is_path_exists(graph, source, sink):
    """
    Check if there's a path from source to sink in the graph

    Args:
        graph: Adjacency matrix of the network
        source: Source vertex index
        sink: Sink vertex index

    Returns:
        True if path exists, False otherwise
    """
    n = len(graph)
    visited = [False] * n

    def dfs(u):
        if u == sink:
            return True
        visited[u] = True
        for v in range(n):
            if not visited[v] and graph[u][v] > 0:
                if dfs(v):
                    return True
        return False

    return dfs(source)


def get_node_labels(num_vertices):
    """
    Generate letter labels for nodes

    Args:
        num_vertices: Number of vertices

    Returns:
        Dictionary mapping node indices to letter labels
    """
    # Use uppercase letters first, then lowercase, then combinations if needed
    labels = {}
    uppercase = list(string.ascii_uppercase)
    lowercase = list(string.ascii_lowercase)

    for i in range(num_vertices):
        if i < len(uppercase):
            labels[i] = uppercase[i]
        elif i < len(uppercase) + len(lowercase):
            labels[i] = lowercase[i - len(uppercase)]
        else:
            # For more than 52 nodes, start using combinations
            primary = (i - len(uppercase) - len(lowercase)) // len(uppercase) + 1
            secondary = (i - len(uppercase) - len(lowercase)) % len(uppercase)
            labels[i] = uppercase[primary - 1] + uppercase[secondary]

    return labels


def visualize_network(graph, source, sink, flow=None, max_flow=None):
    """
    Visualize the flow network with letter labels for nodes and improved edge label visibility

    Args:
        graph: Adjacency matrix of the network
        source: Source vertex index
        sink: Sink vertex index
        flow: Optional flow matrix to show current flow values
        max_flow: Maximum flow value to display in the title
    """
    plt.figure(figsize=(12, 10))  # Larger figure size
    G = nx.DiGraph()

    n = len(graph)
    node_labels = get_node_labels(n)

    # Add nodes with letter labels
    for i in range(n):
        G.add_node(i, label=node_labels[i])

    # Add edges
    for i in range(n):
        for j in range(n):
            if graph[i][j] > 0:
                if flow is not None:
                    G.add_edge(i, j, capacity=graph[i][j], flow=flow[i][j],
                               label=f"{flow[i][j]:.0f}/{graph[i][j]:.0f}")
                else:
                    G.add_edge(i, j, capacity=graph[i][j],
                               label=f"{graph[i][j]:.0f}")

    # Use kamada_kawai_layout for better spacing between nodes
    pos = nx.kamada_kawai_layout(G)

    # Draw nodes with increased size
    node_colors = ['lightgreen' if node == source else 'lightblue' if node == sink else 'lightgray' for node in
                   G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, edgecolors='black')

    # Draw edges with more spacing
    nx.draw_networkx_edges(G, pos, arrowsize=20, width=1.5)

    # Draw node labels with larger font
    node_label_dict = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_label_dict, font_size=16, font_weight='bold')

    # Draw edge labels with improved visibility
    # Compute dynamic edge labels showing net flow/capacity
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        forward_flow = data.get('flow', 0)
        backward_flow = G.edges[v, u]['flow'] if G.has_edge(v, u) else 0
        net_flow = forward_flow - backward_flow
        edge_labels[(u, v)] = f"{net_flow}/{data['capacity']}"

    # Now draw the edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=14,
                                 font_color='darkred',
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),
                                 font_weight='bold',
                                 label_pos=0.5)

    source_label = node_labels[source]
    sink_label = node_labels[sink]

    # Create title with max flow information if available
    if max_flow is not None:
        title = f"Flow Network (Source: {source_label}, Sink: {sink_label}) - Maximum Flow: {max_flow:.0f}"
    else:
        title = f"Flow Network (Source: {source_label}, Sink: {sink_label})"

    plt.title(title, fontsize=18)

    # Add a text box showing the maximum flow value
    if max_flow is not None:
        # Add a text box at the bottom of the plot
        text_box = f"Maximum Flow: {max_flow:.0f}"
        plt.figtext(0.5, 0.01, text_box, fontsize=16,
                    bbox=dict(facecolor='lightgreen', edgecolor='green', boxstyle='round,pad=0.5'),
                    ha='center')

    plt.axis('off')

    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.show()


def print_network_info(graph, source, sink, node_labels, max_flow=None):
    """
    Print information about the network using letter labels
    """
    print(f"Source: {node_labels[source]}, Sink: {node_labels[sink]}")
    if max_flow is not None:
        print(f"Maximum Flow: {max_flow}")

    # Check if a path exists
    path_exists = is_path_exists(graph, source, sink)
    print(f"Path exists from source to sink: {path_exists}")

    print("\nCapacities:")
    # Print header row with letter labels
    header = "    " + " ".join(f"{node_labels[j]:>4}" for j in range(len(graph)))
    print(header)

    # Print capacity matrix with row labels
    for i in range(len(graph)):
        row = f"{node_labels[i]:2}" + " " + " ".join(f"{graph[i][j]:4.0f}" for j in range(len(graph)))
        print(row)


def test_with_different_configurations():
    """
    Test the Ford-Fulkerson implementation with different network configurations
    """
    # Test 1: Small network
    print("Test 1: Small network (5 vertices)")
    graph, source, sink = generate_random_flow_network(5, max_capacity=15, density=0.4)
    network = FlowNetwork(graph, source, sink)
    node_labels = get_node_labels(len(graph))

    print_network_info(graph, source, sink, node_labels)

    max_flow = network.ford_fulkerson()
    print(f"Maximum Flow: {max_flow}")

    visualize_network(graph, source, sink, network.flow, max_flow)

    # Test 2: Medium network
    print("\nTest 2: Medium network (8 vertices)")
    graph, source, sink = generate_random_flow_network(8, max_capacity=20, density=0.3)
    network = FlowNetwork(graph, source, sink)
    node_labels = get_node_labels(len(graph))

    print_network_info(graph, source, sink, node_labels)
    max_flow = network.ford_fulkerson()
    print(f"Maximum Flow: {max_flow}")

    visualize_network(graph, source, sink, network.flow, max_flow)

    # Test 3: Large network
    print("\nTest 3: Large network (12 vertices)")
    graph, source, sink = generate_random_flow_network(12, max_capacity=30, density=0.2)
    network = FlowNetwork(graph, source, sink)
    node_labels = get_node_labels(len(graph))

    print_network_info(graph, source, sink, node_labels)
    max_flow = network.ford_fulkerson()
    print(f"Maximum Flow: {max_flow}")

    visualize_network(graph, source, sink, network.flow, max_flow)


def custom_test():
    """
    Create a custom test with user-defined parameters
    """
    num_vertices = int(input("Enter number of vertices: "))
    min_capacity = int(input("Enter minimum capacity: "))
    max_capacity = int(input("Enter maximum capacity: "))
    density = float(input("Enter network density (0.0-1.0): "))

    graph, source, sink = generate_random_flow_network(
        num_vertices, min_capacity, max_capacity, density
    )

    network = FlowNetwork(graph, source, sink)
    node_labels = get_node_labels(num_vertices)

    print_network_info(graph, source, sink, node_labels)

    max_flow = network.ford_fulkerson()
    print(f"Maximum Flow: {max_flow}")

    visualize_network(graph, source, sink, network.flow, max_flow)


if __name__ == "__main__":
    print("Ford-Fulkerson Maximum Flow Algorithm")
    print("------------------------------------")

    while True:
        print("\nOptions:")
        print("1. Run predefined tests with different network configurations")
        print("2. Create a custom random network")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            test_with_different_configurations()
        elif choice == '2':
            custom_test()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")