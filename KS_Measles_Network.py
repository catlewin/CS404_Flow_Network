import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import time
from flow_network import FlowNetwork


def analyze_kansas_measles(show_plot=True):
    """
    Analyze and visualize Kansas measles outbreak transmission risk using network flow algorithms.

    Parameters:
    -----------
    show_plot : bool
        Whether to display the plot (True) or just return the results (False).

    Returns:
    --------
    dict
        Dictionary containing flow results, total risk per county, and major contributors.
    """
    # Process the population data
    population_data = {
        "Barber": 4233,
        "Clark": 1873,
        "Comanche": 1694,
        "Edwards": 2902,
        "Finney": 38469,
        "Ford": 34072,
        "Grant": 7352,
        "Gray": 5730,
        "Haskell": 3781,
        "Hamilton": 2517,
        "Hodgeman": 1721,
        "Kearny": 3980,
        "Kiowa": 2436,
        "Lane": 1575,
        "Meade": 4063,
        "Morton": 2485,
        "Ness": 2687,
        "Pratt": 9149,
        "Stanton": 2086,
        "Seward": 21276,
        "Scott": 5145,
        "Stevens": 5035
    }

    # Create a directed graph for flow analysis
    G_flow = nx.DiGraph()

    # Define counties with case data
    counties = {
        "Finney": {"cases": 3, "status": "Outbreak", "population": population_data["Finney"]},
        "Ford": {"cases": 3, "status": "Outbreak", "population": population_data["Ford"]},
        "Grant": {"cases": 3, "status": "Outbreak", "population": population_data["Grant"]},
        "Gray": {"cases": 6, "status": "Outbreak", "population": population_data["Gray"]},
        "Haskell": {"cases": 8, "status": "Outbreak", "population": population_data["Haskell"]},
        "Kiowa": {"cases": 6, "status": "Outbreak", "population": population_data["Kiowa"]},
        "Morton": {"cases": 3, "status": "Outbreak", "population": population_data["Morton"]},
        "Stevens": {"cases": 7, "status": "Outbreak", "population": population_data["Stevens"]},
        # Adding surrounding counties
        "Barber": {"cases": 0, "status": "Surrounding", "population": population_data["Barber"]},
        "Clark": {"cases": 0, "status": "Surrounding", "population": population_data["Clark"]},
        "Comanche": {"cases": 0, "status": "Surrounding", "population": population_data["Comanche"]},
        "Edwards": {"cases": 0, "status": "Surrounding", "population": population_data["Edwards"]},
        "Hamilton": {"cases": 0, "status": "Surrounding", "population": population_data["Hamilton"]},
        "Hodgeman": {"cases": 0, "status": "Surrounding", "population": population_data["Hodgeman"]},
        "Kearny": {"cases": 0, "status": "Surrounding", "population": population_data["Kearny"]},
        "Lane": {"cases": 0, "status": "Surrounding", "population": population_data["Lane"]},
        "Meade": {"cases": 0, "status": "Surrounding", "population": population_data["Meade"]},
        "Ness": {"cases": 0, "status": "Surrounding", "population": population_data["Ness"]},
        "Scott": {"cases": 0, "status": "Surrounding", "population": population_data["Scott"]},
        "Stanton": {"cases": 0, "status": "Surrounding", "population": population_data["Stanton"]},
        "Seward": {"cases": 0, "status": "Surrounding", "population": population_data["Seward"]},
        "Pratt": {"cases": 0, "status": "Surrounding", "population": population_data["Pratt"]}
    }

    # Calculate population-based metrics
    for county, data in counties.items():
        # Calculate case rate per 10,000 population
        data["case_rate"] = (data["cases"] / data["population"]) * 10000

    # Add nodes for counties
    for county, data in counties.items():
        G_flow.add_node(county, **data)

    # Define adjacency with capacity values
    adjacency = {
        "Finney": [("Kearny", 2), ("Gray", 2), ("Haskell", 2), ("Grant", 2), ("Scott", 2), ("Lane", 2),
                   ("Ness", 2), ("Hodgeman", 2)],
        "Ford": [("Gray", 2), ("Hodgeman", 2), ("Kiowa", 2), ("Edwards", 2), ("Clark", 2), ("Meade", 1)],
        "Grant": [("Finney", 1), ("Kearny", 2), ("Stanton", 2), ("Hamilton", 1), ("Haskell", 2),
                  ("Stevens", 2), ("Morton", 1), ("Seward", 1)],
        "Gray": [("Finney", 6), ("Haskell", 6), ("Hodgeman", 3), ("Ford", 6), ("Meade", 6)],
        "Haskell": [("Finney", 10), ("Gray", 10), ("Kearny", 5), ("Seward", 10), ("Stevens", 5), ("Grant", 10),
                    ("Meade", 3)],
        "Kiowa": [("Barber", 3), ("Ford", 6), ("Edwards", 6), ("Comanche", 6), ("Clark", 3), ("Pratt", 6)],
        "Morton": [("Stanton", 2), ("Grant", 1), ("Stevens", 2)],
        "Stevens": [("Morton", 8), ("Grant", 8), ("Haskell", 4), ("Seward", 8), ("Stanton", 4)]
    }

    # Add edges to the graph with population-weighted capacities
    for source, targets in adjacency.items():
        for target, base_risk in targets:
            # Get source and target data
            source_data = counties[source]
            target_data = counties[target]

            # Calculate capacity based on case rate and population factors
            case_rate_factor = 1 + (source_data["case_rate"] / 5)
            population_factor = np.log10(target_data["population"]) / 3
            transmission_capacity = base_risk * case_rate_factor * population_factor

            # Add edge with the calculated capacity
            G_flow.add_edge(source, target, capacity=transmission_capacity)

            # Also add reverse edge for movement in the other direction
            target_case_rate_factor = 1 + (target_data["case_rate"] / 5)
            source_population_factor = np.log10(source_data["population"]) / 3
            reverse_capacity = base_risk * target_case_rate_factor * source_population_factor

            G_flow.add_edge(target, source, capacity=reverse_capacity)

    # Approximate positions for Kansas counties in your analysis (x, y coordinates)
    # These are relative positions that roughly match the actual geographic layout
    county_positions = {
        # Outbreak counties
        "Finney": (3, 3),
        "Ford": (5, 3),
        "Grant": (2, 3),
        "Gray": (4, 3),
        "Haskell": (3, 2),
        "Kiowa": (6, 2),
        "Morton": (1, 1),
        "Stevens": (2, 1),

        # Surrounding counties
        "Hamilton": (1, 4),
        "Kearny": (2, 4),
        "Scott": (3, 4),
        "Lane": (4, 4),
        "Ness": (5, 4),
        "Hodgeman": (5, 3),
        "Edwards": (6, 3),
        "Stanton": (1, 2),
        "Seward": (3, 1),
        "Meade": (4, 1),
        "Clark": (5, 1),
        "Comanche": (7, 1),
        "Barber": (7, 2),
        "Pratt": (7, 3),
    }

    print("\nMaximum Flow Analysis (Transmission Risk) using Ford-Fulkerson Algorithm:")
    print("--------------------------------------------------------------------")

    # Identify outbreak and surrounding counties
    outbreak_counties = [county for county, data in counties.items() if data["status"] == "Outbreak"]
    surrounding_counties = [county for county, data in counties.items() if data["status"] == "Surrounding"]

    # Calculate maximum flow from each outbreak county to each surrounding county
    flow_results = {}
    execution_times = {"ford_fulkerson": []}

    # Create adjacency matrix representation for the flow graph
    counties_list = list(counties.keys())
    n = len(counties_list)
    county_index = {county: i for i, county in enumerate(counties_list)}  # Map county names to indices

    # Initialize adjacency matrix with zeros
    adjacency_matrix = np.zeros((n, n))

    # Fill the matrix with capacities from our graph
    for u, v, data in G_flow.edges(data=True):
        source_idx = county_index[u]
        target_idx = county_index[v]
        adjacency_matrix[source_idx][target_idx] = data['capacity']

    def get_node_labels(n):
        """Return node labels (county names) for visualization"""
        return counties_list

    # Calculate flows between counties
    start_time = time.time()
    for source in outbreak_counties:
        for target in surrounding_counties:
            if nx.has_path(G_flow, source, target):  # Check if there's a path between the counties
                try:
                    # Get indices for source and target
                    source_idx = county_index[source]
                    target_idx = county_index[target]

                    # Use your FlowNetwork class
                    network = FlowNetwork(adjacency_matrix, source_idx, target_idx)
                    flow_value, exec_time = network.ford_fulkerson()

                    # Store the result
                    if target not in flow_results:
                        flow_results[target] = []

                    flow_results[target].append((source, flow_value))
                    execution_times["ford_fulkerson"].append(exec_time)

                    print(f"Maximum transmission risk from {source} to {target}: {flow_value:.2f}")
                except Exception as e:
                    print(f"Error calculating flow from {source} to {target}: {e}")

    total_analysis_time = time.time() - start_time
    print(f"Total analysis time: {total_analysis_time:.2f} seconds")

    # Calculate total risk for surrounding counties and major contributors
    total_flows = {}
    major_contributors = {}

    if flow_results:
        print("\nSummarized Transmission Risk to Surrounding Counties:")
        print("----------------------------------------------------")

        # Calculate total incoming flow for each surrounding county
        for target, sources in flow_results.items():
            total_flow = sum(flow for _, flow in sources)
            total_flows[target] = total_flow

            # Find major contributor (source with highest flow)
            if sources:
                major_contributor = max(sources, key=lambda x: x[1])
                major_contributors[target] = major_contributor

        # Sort counties by total incoming flow
        sorted_counties = sorted(total_flows.items(), key=lambda x: x[1], reverse=True)

        # Print results
        for county, flow in sorted_counties:
            print(f"{county}: Total incoming transmission risk = {flow:.2f}")

            # List major contributors (sources)
            contributors = sorted(flow_results[county], key=lambda x: x[1], reverse=True)
            for source, source_flow in contributors:
                contribution_percent = (source_flow / flow) * 100
                print(f"  - From {source}: {source_flow:.2f} ({contribution_percent:.1f}%)")

    # Store results in dictionary to return
    results = {
        "flow_results": flow_results,
        "total_flows": total_flows,
        "major_contributors": major_contributors
    }

    # Store risk values in the county data for visualization
    for county in counties:
        if county in total_flows:
            counties[county]["risk"] = total_flows[county]
            if county in major_contributors:
                counties[county]["major_contributor"] = major_contributors[county][0]
                counties[county]["contributor_flow"] = major_contributors[county][1]
        else:
            counties[county]["risk"] = 0

    # Visualization part - only run if show_plot is True
    if show_plot:
        # Create a figure and axis with proper size
        fig, ax = plt.subplots(figsize=(14, 12))

        # Find maximum case rate and maximum risk for color normalization
        max_case_rate = max(data["case_rate"] for data in counties.values())
        max_risk = 0
        if total_flows:
            max_risk = max(total_flows.values())

        # Create a custom colormap for risk visualization
        risk_cmap = LinearSegmentedColormap.from_list('risk_cmap', ['#f7fbff', '#08306b'])

        # Define a simplified county shape (just a square for each county)
        def create_county_patch(pos, width=0.8, height=0.8):
            x, y = pos
            return Polygon([(x - width / 2, y - height / 2),
                            (x + width / 2, y - height / 2),
                            (x + width / 2, y + height / 2),
                            (x - width / 2, y + height / 2)],
                           closed=True)

        # Draw county boundaries
        for county, pos in county_positions.items():
            patch = create_county_patch(pos)
            county_data = counties[county]

            # Color based on status and risk
            if county_data["status"] == "Outbreak":
                # Color outbreak counties based on case rate
                color_intensity = county_data["case_rate"] / max_case_rate
                facecolor = plt.cm.YlOrRd(color_intensity)
                edgecolor = 'darkred'
                linewidth = 2
            else:
                # Color surrounding counties based on incoming risk
                if max_risk > 0 and county_data["risk"] > 0:
                    risk_intensity = county_data["risk"] / max_risk
                    facecolor = risk_cmap(risk_intensity)

                    # Thicker border for higher risk counties
                    linewidth = 1 + 2 * risk_intensity
                    edgecolor = 'navy'
                else:
                    facecolor = 'lightgray'
                    edgecolor = 'gray'
                    linewidth = 1

            ax.add_patch(plt.Polygon(patch.get_xy(),
                                     closed=True,
                                     facecolor=facecolor,
                                     edgecolor=edgecolor,
                                     linewidth=linewidth,
                                     alpha=0.6))

            # Add county name and info
            x, y = pos
            pop = county_data["population"]
            cases = county_data["cases"]

            # Display county name
            ax.text(x, y + 0.15, county, ha='center', va='center', fontsize=10, fontweight='bold')

            # Display population
            ax.text(x, y - 0.05, f"Pop: {pop:,}", ha='center', va='center', fontsize=8)

            # Display cases if any
            if cases > 0:
                ax.text(x, y - 0.2, f"Cases: {cases}", ha='center', va='center', fontsize=9, color='darkred')
                ax.text(x, y - 0.3, f"Rate: {county_data['case_rate']:.1f}", ha='center', va='center', fontsize=8,
                        color='darkred')

            # Display risk for surrounding counties
            if county_data["status"] == "Surrounding" and "risk" in county_data and county_data["risk"] > 0:
                ax.text(x, y - 0.2, f"Risk: {county_data['risk']:.1f}", ha='center', va='center', fontsize=9,
                        color='navy')
                if "major_contributor" in county_data:
                    contributor = county_data["major_contributor"]
                    ax.text(x, y - 0.3, f"Main src: {contributor}", ha='center', va='center', fontsize=7, color='navy')

        # Draw edges to represent transmission risk
        for u, v, data in G_flow.edges(data=True):
            # Only draw if both counties are in our position dictionary
            if u in county_positions and v in county_positions:
                source_pos = county_positions[u]
                target_pos = county_positions[v]

                # Get capacity for line thickness and alpha
                capacity = data['capacity']

                # Calculate arrow properties
                dx = target_pos[0] - source_pos[0]
                dy = target_pos[1] - source_pos[1]

                # Skip drawing if capacity is too small (to avoid visual clutter)
                if capacity > 1:
                    # Shorten arrows to avoid overlapping with county boxes
                    # Calculate shortened start and end points
                    length = np.sqrt(dx ** 2 + dy ** 2)
                    if length > 0:
                        shrink_factor = 0.4 / length  # Shrink by fixed amount
                        new_dx = dx * (1 - shrink_factor)
                        new_dy = dy * (1 - shrink_factor)

                        # Draw the edge
                        ax.arrow(source_pos[0] + dx * shrink_factor / 2,
                                 source_pos[1] + dy * shrink_factor / 2,
                                 new_dx, new_dy,
                                 head_width=0.1,
                                 head_length=0.1,
                                 fc='blue',
                                 ec='blue',
                                 length_includes_head=True,
                                 alpha=min(capacity / 15, 0.7),
                                 linewidth=capacity / 8)

        # Create special arrows for major contribution paths
        for county, data in counties.items():
            if county in surrounding_counties and "major_contributor" in data and data["risk"] > 0:
                source = data["major_contributor"]
                target = county

                if source in county_positions and target in county_positions:
                    source_pos = county_positions[source]
                    target_pos = county_positions[target]

                    # Calculate arrow properties
                    dx = target_pos[0] - source_pos[0]
                    dy = target_pos[1] - source_pos[1]

                    # Get flow value
                    flow_val = data["contributor_flow"]

                    # Calculate shortened start and end points
                    length = np.sqrt(dx ** 2 + dy ** 2)
                    if length > 0:
                        shrink_factor = 0.4 / length
                        new_dx = dx * (1 - shrink_factor)
                        new_dy = dy * (1 - shrink_factor)

                        # Draw the main contributor arrow in a different color
                        ax.arrow(source_pos[0] + dx * shrink_factor / 2,
                                 source_pos[1] + dy * shrink_factor / 2,
                                 new_dx, new_dy,
                                 head_width=0.12,
                                 head_length=0.12,
                                 fc='red',
                                 ec='red',
                                 length_includes_head=True,
                                 alpha=0.7,
                                 linewidth=flow_val / 6)

        # Create a legend
        outbreak_patch = mpatches.Patch(color='orange', label='Outbreak Counties')
        risk_patch = mpatches.Patch(color=risk_cmap(0.7), label='At-Risk Counties (blue intensity = risk level)')
        surrounding_patch = mpatches.Patch(color='lightgray', label='Low-Risk Surrounding Counties')
        arrow_patch = mpatches.Patch(color='blue', alpha=0.5, label='Transmission Path')
        major_arrow_patch = mpatches.Patch(color='red', alpha=0.7, label='Major Transmission Path')

        ax.legend(handles=[outbreak_patch, risk_patch, surrounding_patch, arrow_patch, major_arrow_patch],
                  loc='lower right', fontsize=9)

        # Create a colorbar for the outbreak case rates
        case_cmap = plt.cm.YlOrRd
        norm_case = plt.Normalize(0, max_case_rate)
        sm_case = plt.cm.ScalarMappable(cmap=case_cmap, norm=norm_case)
        sm_case.set_array([])  # Set an empty array
        cbar_case = fig.colorbar(sm_case, ax=ax, label='Case Rate per 10,000 Population', location='right', pad=0.01)

        # Create a second colorbar for the risk values
        if max_risk > 0:
            norm_risk = plt.Normalize(0, max_risk)
            sm_risk = plt.cm.ScalarMappable(cmap=risk_cmap, norm=norm_risk)
            sm_risk.set_array([])  # Set an empty array
            cbar_risk = fig.colorbar(sm_risk, ax=ax, label='Incoming Transmission Risk (Maximum Flow)',
                                     location='right', pad=0.07)

        # Set axis limits with some padding
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 5)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add a border to represent Kansas
        ax.plot([0, 8, 8, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=2)

        # Add title and subtitle
        ax.set_title("Measles Transmission Risk Network in Southwest Kansas", fontsize=16)
        plt.figtext(0.5, 0.005,
                    "County grid layout approximates geographic positions. Risk calculated using Ford-Fulkerson algorithm.",
                    ha='center', fontsize=10, style='italic')

        # Add a text box explaining the maximum flow analysis
        explanation_text = """Maximum Flow Analysis: 
        Risk values calculated using Ford-Fulkerson algorithm.
        Risk represents the maximum possible transmission capacity
        from outbreak counties to surrounding counties."""

        plt.figtext(0.02, 0.04, explanation_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

        # Show the plot
        plt.tight_layout()
        plt.show()

    return results


def get_highest_risk_counties(results, top_n=5):
    """
    Get the highest risk counties based on flow analysis results.

    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_kansas_measles.
    top_n : int
        Number of highest risk counties to return.

    Returns:
    --------
    list
        List of tuples (county_name, risk_value) sorted by risk (highest first).
    """
    total_flows = results.get("total_flows", {})
    if not total_flows:
        return []

    # Sort counties by risk and return top N
    return sorted(total_flows.items(), key=lambda x: x[1], reverse=True)[:top_n]


# This allows the script to be run directly or imported as a module
if __name__ == "__main__":
    analyze_kansas_measles(show_plot=True)