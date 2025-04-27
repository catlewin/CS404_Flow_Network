import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

# Process the population data
population_data = {
    "Finney": 38354,
    "Ford": 34072,
    "Grant": 7097,
    "Gray": 5730,
    "Haskell": 3591,
    "Kiowa": 2436,
    "Morton": 2485,
    "Stevens": 5035,
    "Kearny": 3808,
    "Hodgeman": 1673,
    "Edwards": 2731,
    "Stanton": 2071,
    "Seward": 21276,
    "Clark": 1873,
    "Comanche": 1694
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
    "Kearny": {"cases": 0, "status": "Surrounding", "population": population_data["Kearny"]},
    "Hodgeman": {"cases": 0, "status": "Surrounding", "population": population_data["Hodgeman"]},
    "Edwards": {"cases": 0, "status": "Surrounding", "population": population_data["Edwards"]},
    "Stanton": {"cases": 0, "status": "Surrounding", "population": population_data["Stanton"]},
    "Seward": {"cases": 0, "status": "Surrounding", "population": population_data["Seward"]},
    "Clark": {"cases": 0, "status": "Surrounding", "population": population_data["Clark"]},
    "Comanche": {"cases": 0, "status": "Surrounding", "population": population_data["Comanche"]}
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
    "Finney": [("Kearny", 3), ("Gray", 4), ("Haskell", 5), ("Grant", 4)],
    "Ford": [("Gray", 4), ("Hodgeman", 3), ("Kiowa", 4), ("Edwards", 3)],
    "Grant": [("Stanton", 3), ("Haskell", 4), ("Stevens", 4), ("Morton", 4)],
    "Gray": [("Finney", 5), ("Haskell", 6), ("Ford", 5)],
    "Haskell": [("Finney", 6), ("Gray", 7), ("Ford", 7), ("Seward", 6), ("Stevens", 7), ("Grant", 6)],
    "Kiowa": [("Ford", 5), ("Edwards", 4), ("Comanche", 4), ("Clark", 4)],
    "Morton": [("Stanton", 3), ("Grant", 4), ("Stevens", 4)],
    "Stevens": [("Morton", 4), ("Grant", 5), ("Haskell", 6), ("Seward", 5)]
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
    "Grant": (2, 2),
    "Gray": (4, 3),
    "Haskell": (3, 2),
    "Kiowa": (6, 2),
    "Morton": (1, 1),
    "Stevens": (2, 1),

    # Surrounding counties
    "Kearny": (2, 3),
    "Hodgeman": (5, 4),
    "Edwards": (6, 4),
    "Stanton": (1, 2),
    "Seward": (3, 1),
    "Clark": (7, 2),
    "Comanche": (7, 3)
}

# Create a figure and axis with proper size
fig, ax = plt.subplots(figsize=(14, 12))


# Define a simplified county shape (just a square for each county)
def create_county_patch(pos, width=0.8, height=0.8):
    x, y = pos
    return Polygon([(x - width / 2, y - height / 2),
                    (x + width / 2, y - height / 2),
                    (x + width / 2, y + height / 2),
                    (x - width / 2, y + height / 2)],
                   closed=True)


# Find maximum case rate for color normalization
max_case_rate = max(data["case_rate"] for data in counties.values())

# Draw county boundaries
county_colors = []
for county, pos in county_positions.items():
    patch = create_county_patch(pos)

    # Color based on status (outbreak or surrounding)
    if counties[county]["status"] == "Outbreak":
        # Normalize by max case rate for color intensity
        color_intensity = counties[county]["case_rate"] / max_case_rate
        facecolor = plt.cm.YlOrRd(color_intensity)
        county_colors.append(counties[county]["case_rate"])
        edgecolor = 'darkred'
        linewidth = 2
    else:
        facecolor = 'lightgray'
        county_colors.append(0)  # Zero case rate for surrounding counties
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
    pop = counties[county]["population"]
    cases = counties[county]["cases"]

    # Display county name
    ax.text(x, y + 0.15, county, ha='center', va='center', fontsize=10, fontweight='bold')

    # Display population
    ax.text(x, y - 0.05, f"Pop: {pop:,}", ha='center', va='center', fontsize=8)

    # Display cases if any
    if cases > 0:
        ax.text(x, y - 0.2, f"Cases: {cases}", ha='center', va='center', fontsize=9, color='darkred')
        ax.text(x, y - 0.3, f"Rate: {counties[county]['case_rate']:.1f}", ha='center', va='center', fontsize=8,
                color='darkred')

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

# Create a legend
outbreak_patch = mpatches.Patch(color='orange', label='Outbreak Counties')
surrounding_patch = mpatches.Patch(color='lightgray', label='Surrounding Counties')
arrow_patch = mpatches.Patch(color='blue', alpha=0.5, label='Transmission Risk')
ax.legend(handles=[outbreak_patch, surrounding_patch, arrow_patch], loc='lower right')

# Create a colorbar for the case rates - FIXED this part
cmap = plt.cm.YlOrRd
norm = plt.Normalize(0, max_case_rate)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Set an empty array
cbar = fig.colorbar(sm, ax=ax, label='Case Rate per 10,000 Population')

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
plt.figtext(0.5, 0.01, "County grid layout approximates geographic positions",
            ha='center', fontsize=10, style='italic')

# Show the plot
plt.tight_layout()
plt.show()