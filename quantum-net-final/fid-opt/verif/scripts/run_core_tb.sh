#!/usr/bin/env bash
# run_core_tb.sh
# Generates stimulus then simulates tb_logicnet_core.sv with Vivado xsim.
# Usage: bash run_core_tb.sh [fidelity|area|latency] [n_samples]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIVADO_BIN="/mnt/vault1/kmhatre/Software/AMD/Vivado/2022.2/bin"
export PATH="$VIVADO_BIN:$PATH"

CONFIG="${1:-fidelity}"
N_SAMPLES="${2:-10000}"
DATA_DIR="/mnt/vault1/mfaroo19/quantum-logicnets/experiments/binary_classification/qick_data"

echo "=== Config: $CONFIG  N_SAMPLES: $N_SAMPLES ==="

# Step 1: Generate hex stimulus
echo "--- Generating stimulus ---"
python3 "$SCRIPT_DIR/generate_tb_data_core.py" \
    --config "$CONFIG" \
    --n-samples "$N_SAMPLES" \
    --data-dir "$DATA_DIR" \
    --out-dir "$SCRIPT_DIR"

# Step 2: Compile with xvlog
echo "--- xvlog: compiling logicnets ---"
echo '`timescale 1ns/1ps' > "$SCRIPT_DIR/timescale.v"
"$VIVADO_BIN/xvlog" -sv \
      "$SCRIPT_DIR/timescale.v" \
      "$SCRIPT_DIR/logicnets/myreg.v" \
      "$SCRIPT_DIR/logicnets/logicnet.v" \
      $(find "$SCRIPT_DIR/logicnets" -name 'layer*.v' | sort) \
      "$SCRIPT_DIR/tb_logicnet_core.sv"

# Step 3: Elaborate
echo "--- xelab: elaborating tb_logicnet_core ---"
"$VIVADO_BIN/xelab" -debug typical -top tb_logicnet_core -snapshot tb_core_snap

# Step 4: Simulate
echo "--- xsim: running simulation ---"
"$VIVADO_BIN/xsim" tb_core_snap -R

echo "=== Done ==="
