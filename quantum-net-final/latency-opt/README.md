# LogicNets Readout IP - Quantum Net Final Designs

This repository contains the three hardware accelerator configurations for multiplexed qubit readout using LogicNets, as detailed in the associated paper.

## Project Configurations

*   **`fid-opt/`**: Optimized for high model fidelity (95.99% target).
*   **`area-opt/`**: Optimized for minimal FPGA resource utilization.
*   **`latency-opt/`**: Optimized for minimal inference latency.

## Directory Structure (Per Configuration)

Each configuration follows a standardized structure:

*   `rtl/`: SystemVerilog source files and LogicNets layer modules.
*   `verif/`: Verification environment.
    *   `tb/`: SystemVerilog testbenches.
    *   `scripts/`: Python data generators and Bash simulation wrappers.
*   `data/`: Bit-accurate stimulus and gold-standard label files (`.hex`).
*   `run.tcl`: Vivado script for full synthesis and implementation.
*   `util_project.rpt` / `timing_project.rpt`: Post-implementation results.

## Simulation Setup

To verify any of the designs:

1.  **Generate Test Data**:
    Navigate to `verif/scripts/` and run `python3 generate_tb_data.py`. This generates 10,000 randomized samples (50/50 class balance) in the `data/` directory.
2.  **Run Simulation**:
    Execute the provided shell scripts in `verif/scripts/`:
    *   `./run_tb_accel.sh`: Verifies the internal LogicNets pipeline.
    *   `./run_tb_readout_ip.sh`: Verifies the full AXI-Stream interface.

## Synthesis & Implementation

To rerun the hardware build:
```bash
vivado -mode batch -source run.tcl
```
Target Part: `xczu49dr-ffvf1760-2-e` (ZCU216/ZCU214)
Target Clock: 1.0 ns (1 GHz)
