# Measels in Southwest Kansas

**A Maximum Flow Application**
________

**Introduction**

This project analyzes graphs using the Ford-Fulkerson Algorithm to calculate the maximum flow. With many real life applications, this project creates easy to understand visualizations of maximum flow. Originally developed to model generic networks, the program was expanded to include epidemic modeling and a specific case study: the 2025 Kansas measles outbreak.

_Goals_

The primary purpose of this project is to demonstrate the broad applicability of maximum flow analysis using Ford-Fulkerson algorithm, with a focus on epidemiological modeling. By building a versatile software tool, this project aims to:
- Illustrate algorithmic versatility,
- Bridge computational theory and public health practice,
- Provide epidemic modeling capabilities,
- Analyze the 2025 Kansas measles outbreak, and
- Support data-driven public health decision-making.

_Scope_

The scope of this project encompasses:
- Development of a general-purpose maximum flow calculator using the Ford-Fulkerson
- Extension for epidemic modeling, repurposing the base network components to simulate disease transmission
- Kansas-specific outbreak modeling using real county data and geographic proximity
- Performance benchmarking across different network sizes and densities

_Kansas Measles Outbreak Context_

In 2024, there were no cases of measles in Kansas. In contrast, as of April 2025, there have been 37 cases of measles in Southwest Kansas, as reported by the Kansas Department of Health and Environment. This follows national trends of measles cases surging: from 285 in 2024 to 884 in 2025, as reported by the CDC. In Kansas, 30 of the 37 cases are residents under 18 years old.

**Methodology**

This project originated from coursework in CS404 Introduction to Algorithms & Complexity, where my foundational understanding of Ford-Fulkerson was established. From here, my interest in epidemiology inspired me to create a more specific algorithm for epidemics. 

_Data Sources:_

For the Kansas County measles application, I used two data sources:
- Kansas Department of Health and Environment’s Division of Public Health report on Kansas’ 2025 Measles Outbreak. 
- US Census’ Annual Estimates of the Resident Population for Counties in Kansas.

_Program Creation:_

Development Tools & Libraries:
- _Language:_ Python
- _IDE & Version Control:_ PyCharm, GitHub
- _Libraries:_ networkx, matplotlib, numpy, collections.deque, random, time, and string
- _AI Tools for code generation:_ Claude.ai & ChatGPT

I started by creating a general ford-fulkerson algorithm for a graph network. From there, I implemented an inherited class for an epidemic network to model likelihood of infection. Finally, I programmed a more specific application for the Kansas measles outbreak to model the flow of measles between counties.

**Content**

_Overview of Program_

Program Architecture

_1. Flow Network:_ A general graph-based flow simulator with predefined tests. Each graph has a set number of nodes, a minimum and maximum capacity for the graph’s edges, and a density between 0-1 defining the likelihood of edges between the nodes, 0 meaning no nodes are connecting and 1 indicating all nodes are connected.
- Small: 5 nodes, minimum capacity 1, maximum capacity 15, with density 0.4.
- Medium :8 nodes, minimum capacity 1, maximum capacity 20, with density 0.3.
- Large: 12 nodes, minimum capacity 1, maximum capacity 30, with density 0.2.

_2. Flow Network Performance Tests:_ An option to benchmark the performance of the maximum flow analysis with a graph of 10, 100, and 1,000 nodes, running each with density 0.1, 0.5, and 1.0.

_3. Kansas Measles Network:_
- Nodes: Kansas Counties
- Sources: Counties with confirmed cases
- Sinks: Adjacent counties
- Edge Weights: Based on adjacency, case count, and population
- Objection: Analyze likelihood of disease spread to neighboring counties

_4. Custom Random Network:_ Graph and flow created by user-defined vertex count, capacity, and density.

_5. Epidemic Network:_ This extends the Flow Network with epidemiological framing:
- Nodes: Individuals
- Source: Original infection case
- Sink: Individual being analyzed for likelihood of infections
- Parameters: As currently implemented, the graph is randomly generated with the following settings:

      num_vertices = 10  # Number of vertices in the epidemic network
      min_capacity = 1
      max_capacity = 20
      density = 0.3  # 30% density for the network
      infection_prob = 0.1  # Probability of infection spread between nodes

Visualization

The graphs are visualized using matplotlib. While the flow and epidemic networks are visualized with traditional graphs, the Kansas network is modeled to look more similar to the layout of the state map.

_Results_

Flow Network Predefined Tests

            Small 
                  Nodes 5
                  Minimum capacity 1
                  Maximum capacity 15
                  Density 0.4
                  Calculated max flow: 1

<img width="748" alt="Screen Shot 2025-04-27 at 2 24 14 PM" src="https://github.com/user-attachments/assets/c6500bff-16da-475a-9ec7-0e1bf59c3bfd" />

            Medium 
                  Nodes 8
                  Minimum capacity 1
                  Maximum capacity 20
                  Density 0.3
                  Calculated max flow: 21


<img width="748" alt="Screen Shot 2025-04-27 at 2 24 28 PM" src="https://github.com/user-attachments/assets/2d9b754e-9eec-4710-b409-e8686750906d" />

            Large 
                  Nodes 12
                  Minimum capacity 1
                  Maximum capacity 30
                  Density 0.2
                  Calculated max flow: 18


<img width="748" alt="Screen Shot 2025-04-27 at 2 24 37 PM" src="https://github.com/user-attachments/assets/7d2b81c4-3286-464c-9fcb-ff03c43b22e7" />


_Flow Network – Testing Scalability & Timing_

| **Density** | **10 Nodes** | **100 Nodes** | **1,000 Nodes** |
|-------------|--------------|---------------|-----------------|
| 0.1         | *0.000078 s* | *0.012732 s*  | *9.959056 s*    |
| 0.5         | *0.000256 s* | *0.035640 s*  | *37.675265 s*   |
| 1.0         | *0.000303 s* | *0.060656 s*  | *68.298057 s*   |

Epidemic Network Simulation _- Maximum nodes infected: 35_


<img width="724" alt="Screen Shot 2025-04-30 at 6 58 01 PM" src="https://github.com/user-attachments/assets/663b51b1-0102-4dc3-8497-1bdad6b557f0" />



# Maximum Flow Analysis - County Transmission Risk

*Analysis performed using Ford-Fulkerson Algorithm (Total analysis time: 0.04 seconds)*

## Top 5 At-Risk Counties

| County | Risk Score |
|--------|------------|
| Seward | 429.18 |
| Meade | 267.96 |
| Kearny | 247.42 |
| Stanton | 195.42 |
| Pratt | 139.68 |

## Detailed Transmission Risk Analysis

### Seward County
**Total incoming transmission risk: 429.18**

| Source County | Risk Value | Percentage |
|---------------|------------|------------|
| Haskell | 121.73 | 28.4% |
| Stevens | 93.08 | 21.7% |
| Gray | 58.53 | 13.6% |
| Grant | 55.41 | 12.9% |
| Morton | 45.62 | 10.6% |
| Finney | 29.23 | 6.8% |
| Ford | 12.79 | 3.0% |
| Kiowa | 12.79 | 3.0% |

### Meade County
**Total incoming transmission risk: 267.96**

| Source County | Risk Value | Percentage |
|---------------|------------|------------|
| Grant | 42.63 | 15.9% |
| Gray | 42.63 | 15.9% |
| Haskell | 42.63 | 15.9% |
| Morton | 42.63 | 15.9% |
| Stevens | 42.63 | 15.9% |
| Finney | 29.23 | 10.9% |
| Ford | 12.79 | 4.8% |
| Kiowa | 12.79 | 4.8% |

### Kearny County
**Total incoming transmission risk: 247.42**

| Source County | Risk Value | Percentage |
|---------------|------------|------------|
| Grant | 38.52 | 15.6% |
| Gray | 38.52 | 15.6% |
| Haskell | 38.52 | 15.6% |
| Morton | 38.52 | 15.6% |
| Stevens | 38.52 | 15.6% |
| Finney | 29.23 | 11.8% |
| Ford | 12.79 | 5.2% |
| Kiowa | 12.79 | 5.2% |

### Stanton County
**Total incoming transmission risk: 195.42**

| Source County | Risk Value | Percentage |
|---------------|------------|------------|
| Finney | 28.31 | 14.5% |
| Grant | 28.31 | 14.5% |
| Gray | 28.31 | 14.5% |
| Haskell | 28.31 | 14.5% |
| Morton | 28.31 | 14.5% |
| Stevens | 28.31 | 14.5% |
| Ford | 12.79 | 6.5% |
| Kiowa | 12.79 | 6.5% |

### Pratt County
**Total incoming transmission risk: 139.68**

| Source County | Risk Value | Percentage |
|---------------|------------|------------|
| Kiowa | 46.95 | 33.6% |
| Finney | 13.25 | 9.5% |
| Ford | 13.25 | 9.5% |
| Grant | 13.25 | 9.5% |
| Gray | 13.25 | 9.5% |
| Haskell | 13.25 | 9.5% |
| Morton | 13.25 | 9.5% |
| Stevens | 13.25 | 9.5% |

### Edwards County
**Total incoming transmission risk: 136.48**

| Source County | Risk Value | Percentage |
|---------------|------------|------------|
| Kiowa | 43.76 | 32.1% |
| Finney | 13.25 | 9.7% |
| Ford | 13.25 | 9.7% |
| Grant | 13.25 | 9.7% |
| Gray | 13.25 | 9.7% |
| Haskell | 13.25 | 9.7% |
| Morton | 13.25 | 9.7% |
| Stevens | 13.25 | 9.7% |

### Comanche County
**Total incoming transmission risk: 131.00**

| Source County | Risk Value | Percentage |
|---------------|------------|------------|
| Kiowa | 38.27 | 29.2% |
| Finney | 13.25 | 10.1% |
| Ford | 13.25 | 10.1% |
| Grant | 13.25 | 10.1% |
| Gray | 13.25 | 10.1% |
| Haskell | 13.25 | 10.1% |
| Morton | 13.25 | 10.1% |
| Stevens | 13.25 | 10.1% |

### Hodgeman County
**Total incoming transmission risk: 115.84**

| Source County | Risk Value | Percentage |
|---------------|------------|------------|
| Finney | 15.04 | 13.0% |
| Gray | 15.04 | 13.0% |
| Haskell | 15.04 | 13.0% |
| Stevens | 15.04 | 13.0% |
| Grant | 15.04 | 13.0% |
| Morton | 15.04 | 13.0% |
| Ford | 12.79 | 11.0% |
| Kiowa | 12.79 | 11.0% |

### Clark County
**Total incoming transmission risk: 114.69**

| Source County | Risk Value | Percentage |
|---------------|------------|------------|
| Kiowa | 21.96 | 19.1% |
| Finney | 13.25 | 11.6% |
| Ford | 13.25 | 11.6% |
| Grant | 13.25 | 11.6% |
| Gray | 13.25 | 11.6% |
| Haskell | 13.25 | 11.6% |
| Morton | 13.25 | 11.6% |
| Stevens | 13.25 | 11.6% |

### Barber County
**Total incoming transmission risk: 114.22**

| Source County | Risk Value | Percentage |
|---------------|------------|------------|
| Kiowa | 21.49 | 18.8% |
| Finney | 13.25 | 11.6% |
| Ford | 13.25 | 11.6% |
| Grant | 13.25 | 11.6% |
| Gray | 13.25 | 11.6% |
| Haskell | 13.25 | 11.6% |
| Morton | 13.25 | 11.6% |
| Stevens | 13.25 | 11.6% |

### Lower Risk Counties

| County | Total Risk | Top Source |
|--------|------------|------------|
| Scott | 22.88 | Equal distribution (12.5% each) |
| Ness | 21.14 | Equal distribution (12.5% each) |
| Lane | 19.71 | Equal distribution (12.5% each) |
| Hamilton | 16.47 | Equal distribution (12.5% each) |

## Major Transmission Paths

| Source → Destination | Flow Value |
|----------------------|------------|
| Haskell → Seward | 121.73 |
| Kiowa → Pratt | 46.95 |
| Kiowa → Edwards | 43.76 |
| Grant → Meade | 42.63 |
| Kiowa → Comanche | 38.27 |
| Grant → Kearny | 38.52 |
| Finney → Stanton | 28.31 |
| Kiowa → Clark | 21.96 |
| Kiowa → Barber | 21.49 |
| Finney → Hodgeman | 15.04 |


<img width="761" alt="Screen Shot 2025-04-29 at 9 21 27 PM" src="https://github.com/user-attachments/assets/2117ac46-2d17-4be5-92b6-5f84f9efdd74" />


**Conclusion**

This program demonstrates the power and versatility of maximum flow analysis using Ford-Fulkerson. By modeling disease spread as a flow network, we can better understand potential outbreak trajectories and support public health resources.

_Future Improvements_

Epidemic Network: Add more user input to specify the desired network. This would allow for better analysis for disease outbreaks and enable reusability for the program.
Kansas Model: This model can be improved by
- Using transportation data, vaccination rates, and healthcare access to increase transmission estimation accuracy
- Using geopandas for accurate geographic visualization
- Modeling time-based progression of outbreaks for dynamic forecasting

While powerful, the algorithm becomes computationally expensive for large graphs (100+ nodes), highlighting a limitation in scalability.


**References**

Bureau, US Census. “County Population Totals and Components of Change: 2020-2024.” Census.Gov, 12 Mar. 2025, www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html 

“Measles Cases and Outbreaks.” Centers for Disease Control and Prevention, Centers for Disease Control and Prevention, www.cdc.gov/measles/data-research/index.html. Accessed 25 Apr. 2025. 

“Measles Data.” Measles Data | KDHE, KS, www.kdhe.ks.gov/2314/Measles-Data. Accessed 25 Apr. 2025. 
