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

I started by creating a general ford-fulkerson algorithm for a graph network. From there, I implemented an inherited class for an epidemic network to model likelihood of infection. Finally, I programmed a more specific application for the Kansas measles outbreak to model the likelihood of infection using flow networks to represent flow of infection.
