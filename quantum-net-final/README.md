This directory contains the three separate designs finalized in the paper. Each one is optimized for a different target.
- `area-opt` is optimized for area
- `fid-opt` is optimized for model fidelity
- `latency-opt` is optimized for latency

in each directory:
- `run.tcl` is the Vivado script to compile and implement the project
- `readout_ip.sv` is the top level file
- `/logicnets` contains the trained LogicNets RTL
- the timing and utilization reports are available at `timing_project.rpt` and `util_project.rpt` respectively
