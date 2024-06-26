# Patchy Particles MD simulations

## Overview
This software suite simulates the dynamics of patchy particle systems. Cores of particles interact through excluded volume interactions while patches attract each other and can intersect. 

<img src="particle.svg"  width="300"/> <img src="chain.svg" width="300"/>

## Repository Structure
- `/progs`: Contains the core scripts
  - `initialisation.py`: Script to initialize the simulation setup.
  - `actions.py`: Conducts the simulations with predefined parameters.
  - `analysis.py`: Analyzes simulation results.
  - `drawing.py`: Sketches representations of patchy particles.
- `/scripts`: Contains script to run various simulations.
  - `run_sim.py`: 
- `/notebooks`: Jupyter notebooks for data analysis and visualization.
  - Includes notebooks for analyzing bonds, structure, mobility, and phase diagrams.

