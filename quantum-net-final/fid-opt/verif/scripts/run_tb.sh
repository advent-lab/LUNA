#!/usr/bin/env bash
# run_tb.sh  —  fid-opt logicnet testbench
# Usage: bash run_tb.sh [n_samples]
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIVADO="/mnt/vault1/kmhatre/Software/AMD/Vivado/2022.2/bin"
N="${1:-10000}"

echo "=== Generating stimulus (N=$N) ==="
python3 "$DIR/generate_tb_data.py"

echo "=== Compiling ==="
echo '`timescale 1ns/1ps' > "$DIR/timescale.v"
"$VIVADO/xvlog" -sv \
    "$DIR/timescale.v" \
    "$DIR/logicnets/myreg.v" \
    "$DIR/logicnets/logicnet.v" \
    $(find "$DIR/logicnets" -name 'layer*.v' | sort) \
    "$DIR/tb_logicnet.sv"

echo "=== Elaborating ==="
"$VIVADO/xelab" -debug typical -top tb_logicnet -snapshot tb_logicnet_snap

echo "=== Simulating ==="
"$VIVADO/xsim" tb_logicnet_snap -R
