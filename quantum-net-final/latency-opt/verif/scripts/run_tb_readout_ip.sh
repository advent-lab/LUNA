#!/usr/bin/env bash
# run_tb_readout_ip.sh
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIVADO="/mnt/vault1/kmhatre/Software/AMD/Vivado/2022.2/bin"

# Project Paths
RTL_DIR="$DIR/../../rtl"
TB_DIR="$DIR/../tb"
DATA_DIR="$DIR/../../data"

echo "=== Generating stimulus ==="
python3 "$DIR/generate_tb_data.py"

echo "=== Compiling ==="
echo '`timescale 1ns/1ps' > timescale.v
"$VIVADO/xvlog" -sv \
    timescale.v \
    "$RTL_DIR/sum_signed.sv" \
    "$RTL_DIR/logicnets/myreg.v" \
    "$RTL_DIR/logicnets/logicnet.v" \
    $(find "$RTL_DIR/logicnets" -name 'layer*.v' | sort) \
    "$RTL_DIR/readout_ip.sv" \
    "$TB_DIR/tb_readout_ip.sv"

echo "=== Elaborating ==="
"$VIVADO/xelab" -debug typical -top tb_readout_ip -snapshot tb_readout_ip_snap

echo "=== Simulating ==="
ln -sf "$DATA_DIR"/*.hex .
"$VIVADO/xsim" tb_readout_ip_snap -R
