#!/usr/bin/env bash
# run_tb_accel.sh
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIVADO="/mnt/vault1/kmhatre/Software/AMD/Vivado/2022.2/bin"

# Project Paths
RTL_DIR="$DIR/../../rtl"
TB_DIR="$DIR/../tb"
DATA_DIR="$DIR/../../data"

echo "=== Compiling tb_accel ==="
echo '`timescale 1ns/1ps' > timescale.v
"$VIVADO/xvlog" -sv \
    timescale.v \
    "$RTL_DIR/sum_signed.sv" \
    "$RTL_DIR/logicnets/myreg.v" \
    "$RTL_DIR/logicnets/logicnet.v" \
    $(find "$RTL_DIR/logicnets" -name 'layer*.v' | sort) \
    "$RTL_DIR/readout_ip.sv" \
    "$TB_DIR/tb_accel.sv"

echo "=== Elaborating ==="
"$VIVADO/xelab" -debug typical -top tb_accel -snapshot tb_accel_snap

echo "=== Simulating ==="
ln -sf "$DATA_DIR"/*.hex .
"$VIVADO/xsim" tb_accel_snap -R
