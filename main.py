from flow_network import FlowNetwork, generate_random_flow_network, get_node_labels, print_network_info
from epidemic_network import EpidemicNetwork, generate_random_epidemic_network, test_epidemic_simulation
from KS_Measles_Network import analyze_kansas_measles, get_highest_risk_counties

def run_predefined_tests():
    """Run predefined tests with different network sizes"""
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
        network.visualize_network(network.flow , max_flow=max_flow)

def run_performance_tests():
    test_configs = [
        # Small networks (n=10)
        (10, 1, 20, 0.1, "Small sparse network"),
        (10, 1, 20, 0.5, "Small medium-density network"),
        (10, 1, 20, 1.0, "Small dense network"),

        # Medium networks (n=100)
        (100, 1, 20, 0.1, "Medium sparse network"),
        (100, 1, 20, 0.5, "Medium medium-density network"),
        (100, 1, 20, 1.0, "Medium dense network"),

        # Large networks (n=1000)
        (1000, 1, 20, 0.1, "Large sparse network"),
        (1000, 1, 20, 0.5, "Large medium-density network"),
        (1000, 1, 20, 1.0, "Large dense network")
    ]

    print("\n=== Performance Tests ===")
    print("Testing Ford-Fulkerson algorithm performance with different network sizes and densities.")

    for idx, (n, min_c, max_c, density, description) in enumerate(test_configs, 1):
        print(f"\nTest {idx}: {description} (n={n}, density={density})")
        graph, source, sink = generate_random_flow_network(n, min_c, max_c, density)
        network = FlowNetwork(graph, source, sink)

        max_flow, execution_time = network.ford_fulkerson()
        print(f"Maximum Flow: {max_flow}")
        print(f"Execution Time: {execution_time:.6f} seconds")

        # Only visualize smaller networks
        if n <= 10:
            network.visualize_network(network.flow , max_flow=max_flow)


def run_kansas_measles_analysis():
    """Run Kansas measles outbreak analysis"""
    print("\n=== Kansas Measles Outbreak Analysis ===")
    results = analyze_kansas_measles(show_plot=True)

    # Show top at-risk counties
    print("\nTop 5 At-Risk Counties:")
    top_counties = get_highest_risk_counties(results, top_n=5)
    for county, risk in top_counties:
        print(f"- {county}: Risk score = {risk:.2f}")

    # Print major transmission paths
    print("\nMajor Transmission Paths:")
    for county, (source, flow) in results["major_contributors"].items():
        print(f"- {source} → {county}: Flow = {flow:.2f}")

    return results

def run_custom_test():
    """Run a custom test with user-defined parameters"""
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
    network.visualize_network(network.flow , max_flow=max_flow)



def main():
    """Main function to run the program"""
    print("======================================")
    print("    Network Flow Analysis Toolkit     ")
    print("======================================")

    while True:
        print("\nOptions:")
        print("1. Run basic flow network tests")
        print("2. Run performance tests")
        print("3. Run Kansas measles outbreak analysis")
        print("4. Create a custom random network")
        print("5. Run epidemic network simulation")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ").strip()

        if choice == '1':
            run_predefined_tests()
        elif choice == '2':
            run_performance_tests()
        elif choice == '3':
            run_kansas_measles_analysis()
        elif choice == '4':
            run_custom_test()
        elif choice == '5':
            print("\nRunning Epidemic Network Simulation...")
            test_epidemic_simulation()
        elif choice == '6':
            print("Exiting. Thank you for using the Network Flow Analysis Toolkit!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    main()
