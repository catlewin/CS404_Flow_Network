import numpy as np
from collections import deque
import random
from flow_network import FlowNetwork  # Import FlowNetwork class from flow_network.py

class EpidemicNetwork(FlowNetwork):
    def __init__(self, graph, source, sink, infection_prob):
        """
        Initialize an EpidemicNetwork that inherits from FlowNetwork

        Args:
            graph: A 2D list or numpy array where graph[u][v] is the transmission capacity from u to v
            source: Source vertex (initial infected individual)
            sink: Sink vertex (recovered or immune group)
        """
        # Initialize the parent class (FlowNetwork)
        super().__init__(graph, source, sink)
        self.infection_prob = infection_prob  # Set the infection probability
        self.infected_nodes = set()  # You can track infected nodes if needed

    def bfs(self):
        """
        Use BFS to find an augmenting path from source to sink in the residual graph
        (overridden to fit the disease model)

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

    def simulate_infection(self):
        """
        Simulate the spread of disease using the Ford-Fulkerson algorithm (overridden)
        Here, the flow represents the number of people infected, and the path represents the spread of infection.

        Returns:
            max_infected: The maximum number of individuals that can become infected
        """
        # Find the augmenting path (this finds the disease's possible transmission routes)
        path = self.bfs()

        # While there is an augmenting path (indicating more disease spread is possible)
        while path:
            # Find the minimum residual capacity along the path (the bottleneck capacity)
            min_capacity = float('inf')
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                min_capacity = min(min_capacity, self.residual_graph[u][v])

            # Update residual capacities and reverse edges to reflect the flow of infection
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.residual_graph[u][v] -= min_capacity
                self.residual_graph[v][u] += min_capacity
                self.flow[u][v] += min_capacity

            # Find the next augmenting path
            path = self.bfs()

        # The total number of infected individuals is the total flow from source to sink
        max_infected = sum(self.flow[self.source])

        return max_infected

    def visualize_network(self, flow=None, max_flow=None):
        """
        Visualize the epidemic network showing infected paths
        """
        # Call the parent class's visualize method
        super().visualize_network(flow=self.flow, max_flow=max_flow)

        # Display the infected nodes after the epidemic simulation
        print(f"Infected nodes: {self.infected_nodes}")
        print(f"Maximum number of individuals that got infected: {self.simulate_infection()}")


def generate_random_epidemic_network(num_vertices, min_transmission=1, max_transmission=10, density=0.3):
    """
    Generate a random epidemic network with at least one path from source to sink

    Args:
        num_vertices: Number of vertices (individuals) in the network
        min_transmission: Minimum transmission rate (capacity) of an edge
        max_transmission: Maximum transmission rate (capacity) of an edge
        density: Probability of an edge existing between any two vertices

    Returns:
        graph: Adjacency matrix representing the network
        source: Source vertex index (initial infected individual)
        sink: Sink vertex index (recovered or immune group)
    """
    # Initialize with zeros
    graph = np.zeros((num_vertices, num_vertices))

    # Choose source and sink
    vertices = list(range(num_vertices))
    source = random.choice(vertices)
    remaining = [v for v in vertices if v != source]
    sink = random.choice(remaining)

    # Ensure there's at least one path from source to sink
    current = source
    path_length = min(random.randint(2, num_vertices), num_vertices - 1)

    # Create vertices in the path
    path_vertices = [source]
    available_vertices = [v for v in range(num_vertices) if v != source and v != sink]

    if len(available_vertices) > 0 and path_length > 2:
        num_intermediate = min(path_length - 2, len(available_vertices))
        intermediate_vertices = random.sample(available_vertices, num_intermediate)
        path_vertices.extend(intermediate_vertices)

    path_vertices.append(sink)

    # Create edges along this path with random transmission rates
    for i in range(len(path_vertices) - 1):
        u = path_vertices[i]
        v = path_vertices[i + 1]
        graph[u][v] = random.randint(min_transmission, max_transmission)

    # Add additional random edges
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j and graph[i][j] == 0 and random.random() < density:
                graph[i][j] = random.randint(min_transmission, max_transmission)

    return graph, source, sink


def test_epidemic_simulation():
    # Example parameters, you can modify this based on your requirements
    num_vertices = 10  # Number of vertices in the epidemic network
    min_capacity = 1
    max_capacity = 20
    density = 0.3  # 30% density for the network
    infection_prob = 0.1  # Probability of infection spread between nodes

    # Generate a random epidemic network
    graph, source, sink = generate_random_epidemic_network(num_vertices, min_capacity, max_capacity, density)

    # Create the EpidemicNetwork instance
    epidemic_network = EpidemicNetwork(graph, source, sink, infection_prob)

    # Simulate the infection spread
    max_infected = epidemic_network.simulate_infection()

    # Output the result
    print(f"\nMaximum number of infected nodes: {max_infected}")

    # Visualize the epidemic network
    epidemic_network.visualize_network(epidemic_network.flow, max_infected)  # You can customize visualization if needed

