# Setup for Differential Evolution
- set up python env with `de-env.yml`
- install pip dependencies:
- `pip install --no-cache-dir git+https://github.com/Xilinx/brevitas.git@67be9b58c1c63d3923cac430ade2552d0db67ba5`
- `pip install ./logicnets`
- download and unpack data from [Zenodo](https://zenodo.org/records/14427490) into `qick_data`
- Run differential_evolution.py

# Using Differential Evolution Script:

| Argument          | Description                                                                                                                       |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `--pop_size`      | Number of candidate solutions in the Differential Evolution population.                                                           |
| `--gens`          | Number of evolutionary generations to run.                                                                                        |
| `--F`             | Differential weight controlling mutation magnitude.                                                                               |
| `--CR`            | Crossover rate determining how often mutated parameters are inherited.                                                            |
| `--parallel_jobs` | Number of parallel worker processes used for evaluation.                                                                          |
| `--init_seed`     | Random seed for reproducible runs.                                                                                                |
| `--weights`       | Comma-separated weights for the cost function components (`compute_cost`), e.g. `0.2,0.6,0.2`. Must contain exactly three values. First weights fidelity, second weights area, third weights latency |
| `--log_fname`     | Path to the YAML file where logs are written.                                                                        |
| `--debug`         | Enable debug mode with additional logging and checks.                                                                             |


# Setup for LogicNets
[See LogicNets Repo](https://github.com/Xilinx/logicnets)
