"""
test_full_ip.py
---------------
cocotb testbench for `nn_classifier_wrapper`.

Flow per sample:
  1. Pulse `trigger` for one clock while the FSM is in S_IDLE.
  2. Stream SIG_LENGTH 32-bit I/Q packets on in_TDATA/in_TVALID (1 per clock).
  3. Wait for `out_WE` — latch `out_DATA` as the 2-bit prediction.

At the end: compute fidelity = 1 - (P(0|1) + P(1|0)) / 2 and compare to
the software target for the selected config.

Env vars (all optional):
  CONFIG          fidelity|area|latency   [fidelity]
  DATA_DIR        path to qick_data       [./qick_data]
  N_SAMPLES       how many to simulate    [256]
  VCD             set to 1 to dump waves

Run with:
  make -f Makefile.cocotb TEST=full_ip
"""

import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ref_model import (
    load_test_data, pack_iq_packet, fidelity,
)


# --------- config presets --------- #
CONFIGS = {
    'fidelity': dict(num_filter=2, shift_m=7, shift_n=1, sw_fid=95.995),
    'area'    : dict(num_filter=1, shift_m=9, shift_n=0, sw_fid=95.929),
    'latency' : dict(num_filter=2, shift_m=9, shift_n=1, sw_fid=95.885),
}

START       = 100
SIG_LENGTH  = 400
CLK_PERIOD_NS = 5.0      # 200 MHz


async def reset(dut):
    dut.ap_rst_n.value       = 0
    dut.trigger.value        = 0
    dut.in_TVALID.value      = 0
    dut.in_TDATA.value       = 0
    dut.config_AWADDR.value  = 0
    dut.config_AWVALID.value = 0
    for _ in range(8):
        await RisingEdge(dut.ap_clk)
    dut.ap_rst_n.value = 1
    for _ in range(4):
        await RisingEdge(dut.ap_clk)


async def drive_sample(dut, packets_row):
    """Trigger + stream one sample's worth of 32-bit packets."""
    # Pulse trigger for one clock
    dut.trigger.value = 1
    await RisingEdge(dut.ap_clk)
    dut.trigger.value = 0

    # Stream packets
    for pkt in packets_row:
        dut.in_TVALID.value = 1
        dut.in_TDATA.value  = int(pkt)
        await RisingEdge(dut.ap_clk)
    dut.in_TVALID.value = 0
    dut.in_TDATA.value  = 0


async def capture_prediction(dut, timeout_cycles=128):
    """Wait for the next cycle with out_WE==1 and return out_DATA."""
    for _ in range(timeout_cycles):
        await RisingEdge(dut.ap_clk)
        if int(dut.out_WE.value) == 1:
            return int(dut.out_DATA.value)
    raise TimeoutError(f'no out_WE within {timeout_cycles} cycles')


@cocotb.test()
async def run_full_ip(dut):
    config_name = os.environ.get('CONFIG', 'fidelity')
    cfg = CONFIGS[config_name]
    n_samples = int(os.environ.get('N_SAMPLES', '256'))

    data_dir = os.environ.get('DATA_DIR')
    if not data_dir:
        raise RuntimeError(
            'DATA_DIR env var is required. Set it via the Makefile '
            '(make DATA_DIR=/path/to/qick_data ...) or export it directly.'
        )
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    x_path = os.path.join(data_dir, '0528_X_test_0_770.npy')
    y_path = os.path.join(data_dir, '0528_y_test_0_770.npy')
    for p in (x_path, y_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f'Missing data file: {p}')

    dut._log.info(f'--- tb_full_ip ---')
    dut._log.info(f'config     = {config_name}')
    dut._log.info(f'data_dir   = {data_dir}')
    dut._log.info(f'N samples  = {n_samples}')
    dut._log.info(f'SW target  = {cfg["sw_fid"]:.3f} %')

    # ---- load data ----
    X_slice, Y = load_test_data(x_path, y_path, START, SIG_LENGTH, n_samples)
    N = X_slice.shape[0]
    dut._log.info(f'loaded     : X={X_slice.shape} Y={Y.shape}')

    # ---- build packets (N, SIG_LENGTH) of uint32 ----
    packets = np.empty((N, SIG_LENGTH), dtype=np.uint32)
    for n in range(N):
        for t in range(SIG_LENGTH):
            packets[n, t] = pack_iq_packet(
                int(X_slice[n, t, 0]), int(X_slice[n, t, 1]))

    # ---- clock + reset ----
    cocotb.start_soon(Clock(dut.ap_clk, CLK_PERIOD_NS, units='ns').start())
    await reset(dut)

    # ---- run ----
    preds = np.zeros(N, dtype=np.int32)
    for n in range(N):
        await drive_sample(dut, packets[n])
        preds[n] = await capture_prediction(dut)
        if (n % 128) == 0 or n == N - 1:
            dut._log.info(f'  [{n+1}/{N}] label={int(Y[n])} pred={int(preds[n])}')

    # ---- score ----
    fid, p01, p10, n0, n1 = fidelity(Y, preds)
    fid_pct = fid * 100.0
    dut._log.info(f'  N0={n0}  N1={n1}')
    dut._log.info(f'  P(1|0) = {p01:.6f}')
    dut._log.info(f'  P(0|1) = {p10:.6f}')
    dut._log.info(f'RTL fidelity = {fid:.6f}  ({fid_pct:.3f} %)')
    dut._log.info(f'SW  fidelity = {cfg["sw_fid"]:.3f} %')
    dut._log.info(f'delta        = {fid_pct - cfg["sw_fid"]:+.3f} pp')

    # Allow up to 1pp drift — anything larger is a real bug, not quantization.
    assert fid_pct >= cfg['sw_fid'] - 1.0, (
        f'RTL fidelity {fid_pct:.3f}% is >1pp below software target '
        f'{cfg["sw_fid"]:.3f}%')
