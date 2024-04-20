Molecular dynamics simulations of patchy particles system.
Cores interact through excluded volumes interactions, while patches attract each other and can intersect. 

<img src="particle.svg" class="left-image">
<img src="chain.svg" class="right-image">

<style>
  .left-image {
    float: left;
    width: 300px;
    height: auto;
  }

  .right-image {
    float: right;
    width: 300px;
    height: auto;
  }
</style>

The folder /progs is used to initialise the system (initialisation.py), run the simulation (actions.py) and analyse trajectories (analysis.py). The file drawing.py allows to make sketches of patchy particles.

The file scripts/run_sim.py can be used to run different simulations

Data analysis and visualisation are shown in notebooks/ folder. It includes bonds, structure, mobility and phase_diagrams analysis.  
