from flow_network import FlowNetwork, generate_random_flow_network, get_node_labels, print_network_info
from epidemic_network import EpidemicNetwork, generate_random_epidemic_network, test_epidemic_simulation


def run_predefined_tests():
    tests = [
        (5, 1, 15, 0.4),
        (8, 1, 20, 0.3),
        (12, 1, 30, 0.2),
    ]

    for idx, (n, min_c, max_c, density) in enumerate(tests, 1):
        print(f"\n--- Test {idx}: {n} vertices ---")
        graph, source, sink = generate_random_flow_network(n, min_c, max_c, density)
        network = FlowNetwork(graph, source, sink)
        labels = get_node_labels(n)

        print_network_info(graph, source, sink, labels)

        max_flow, execution_time = network.ford_fulkerson()  # Get both max flow and execution time
        print(f"Maximum Flow: {max_flow}")
        print(f"Execution Time: {execution_time:.6f} seconds")  # Display execution time

        network.visualize_network()



def run_custom_test():
    try:
        n = int(input("Enter number of vertices: "))
        min_capacity = int(input("Enter minimum capacity: "))
        max_capacity = int(input("Enter maximum capacity: "))
        density = float(input("Enter network density (0.0-1.0): "))
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
        return

    graph, source, sink = generate_random_flow_network(n, min_capacity, max_capacity, density)
    network = FlowNetwork(graph, source, sink)
    labels = get_node_labels(n)

    print_network_info(graph, source, sink, labels)

    max_flow, execution_time = network.ford_fulkerson()  # Get both max flow and execution time
    print(f"Maximum Flow: {max_flow}")
    print(f"Execution Time: {execution_time:.6f} seconds")  # Display execution time



def main():
    print("======================================")
    print("    Ford-Fulkerson Maximum Flow App   ")
    print("======================================")

    while True:
        print("\nOptions:")
        print("1. Run predefined tests")
        print("2. Create a custom random network")
        print("3. Run Epidemic Network Simulation")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            run_predefined_tests()
        elif choice == '2':
            run_custom_test()
        elif choice == '3':
            print("\nRunning Epidemic Network Simulation...")
            test_epidemic_simulation()  # Call the function to simulate the epidemic network
        elif choice == '4':
            print("Exiting. Thank you for using the app!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
