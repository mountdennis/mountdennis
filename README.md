# mountdennis

This repository contains a simple 2D syringe simulation implemented in `syringe_simulation.py`. The geometry models a 6.35 mm diameter barrel that is 64 mm long, tapering over the last 5 mm into a 12 mm 27G needle. The simulation approximates incompressible flow with a moving piston and supports Carreau and power-law viscosity functions.

Run the basic simulation from the command line:

```bash
python3 syringe_simulation.py
```

A small Tkinter GUI is provided in `syringe_gui.py` to experiment with different needle lengths, barrel/stopper diameters, needle gauges (25G, 27G, 29G or 32G) and viscosity models. Launch it with:

```bash
python3 syringe_gui.py
```

The GUI displays the final average velocity after running the simulation with the chosen parameters.
