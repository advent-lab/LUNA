"""
test_logicnets_core.py
----------------------
cocotb testbench for `logicnet` in isolation.

Drives pre-computed 56-bit feature vectors into M0 and samples M4 after the
5-cycle pipeline latency. The feature vectors come from ref_model.py using
the same integration that training used.

KEY QUESTION this test answers:
  What sub-word order did LogicNets training actually bake into the truth
  tables?  The training python produces [I0..I(nf-1), Q0..Q(nf-1)] but the
  RTL wrapper concatenates {Q(nf-1), I(nf-1), ..., Q0, I0}.  These are not
  the same for num_filter >= 2.

This testbench sweeps FEATURE_ORDER across both packers and reports the
fidelity for each. The one that matches software (~95.995% for Fidelity
config) is the ordering the net actually expects.

Env vars:
  CONFIG          fidelity|area|latency   [fidelity]
  FEATURE_ORDER   training|rtl|both       [both]
  DATA_DIR        path to qick_data       [./qick_data]
  N_SAMPLES                               [512]
"""

import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ref_model import (
    load_test_data, extract_features_int, feature_bitwidth,
    PACKERS, fidelity,
)

CONFIGS = {
    'fidelity': dict(num_filter=2, shift_m=7, shift_n=1, sw_fid=95.995),
    'area'    : dict(num_filter=1, shift_m=9, shift_n=0, sw_fid=95.929),
    'latency' : dict(num_filter=2, shift_m=9, shift_n=1, sw_fid=95.885),
}

START      = 100
SIG_LENGTH = 400
CLK_PERIOD_NS = 5.0

# Logicnet pipeline depth: M0 -> layer0_reg -> layer0_inst ->
#                          layer1_reg -> layer1_inst ->
#                          layer2_reg -> layer2_inst ->
#                          layer3_reg -> layer3_inst -> M4
# There are 4 registers in the data path (layer0_reg..layer3_reg),
# so latency from M0 stable to M4 valid is 4 clocks, not 5.
LATENCY = 4


async def reset(dut):
    dut.rst.value = 1
    dut.M0.value  = 0
    for _ in range(8):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)


async def run_one_ordering(dut, X_slice, Y, cfg, order_name):
    """Stream features built with `order_name` packing and return fidelity."""
    packer = PACKERS[order_name]
    nf     = cfg['num_filter']
    sm     = cfg['shift_m']
    sn     = cfg['shift_n']
    bw     = feature_bitwidth(14, SIG_LENGTH, nf, sm, sn, filt_type=1)
    total  = bw * 2 * nf

    dut._log.info(f'[{order_name}] feature_bw={bw}  total={total} bits  (M0 is 56)')
    if total > 56:
        dut._log.warning(f'[{order_name}] total width exceeds 56; upper bits drop')

    feats = extract_features_int(X_slice, nf, sm, sn)
    N = feats.shape[0]

    # Build all words once
    words = np.empty(N, dtype=object)
    for n in range(N):
        w = packer(feats[n], nf, bw)
        words[n] = w & ((1 << 56) - 1)

    # Stream: drive M0 each cycle, sample M4 LATENCY cycles later.
    preds = np.zeros(N, dtype=np.int32)

    # Pre-fill the pipeline — drive the first LATENCY inputs
    for i in range(LATENCY):
        if i < N:
            dut.M0.value = int(words[i])
        await RisingEdge(dut.clk)

    # For each subsequent cycle, drive next input AND sample the prediction
    # for the input we drove LATENCY cycles ago.
    for k in range(N):
        drive_idx = k + LATENCY
        if drive_idx < N:
            dut.M0.value = int(words[drive_idx])
        else:
            dut.M0.value = 0
        await RisingEdge(dut.clk)
        # After this edge, M4 reflects inputs[k]
        preds[k] = int(dut.M4.value) & 0x3

    # Drain
    for _ in range(LATENCY + 2):
        await RisingEdge(dut.clk)

    fid, p01, p10, n0, n1 = fidelity(Y, preds)
    fid_pct = fid * 100.0
    dut._log.info(
        f'[{order_name}]  N0={n0} N1={n1}  '
        f'P(1|0)={p01:.4f}  P(0|1)={p10:.4f}  '
        f'fid={fid_pct:.3f}%  '
        f'(SW={cfg["sw_fid"]:.3f}%, delta={fid_pct - cfg["sw_fid"]:+.3f} pp)'
    )
    return fid_pct


@cocotb.test()
async def run_logicnets_core(dut):
    config_name = os.environ.get('CONFIG', 'fidelity')
    cfg = CONFIGS[config_name]
    order_mode = os.environ.get('FEATURE_ORDER', 'both')
    n_samples = int(os.environ.get('N_SAMPLES', '512'))

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

    dut._log.info(f'--- tb_logicnets_core ---')
    dut._log.info(f'config        = {config_name}')
    dut._log.info(f'feature order = {order_mode}')
    dut._log.info(f'data_dir      = {data_dir}')
    dut._log.info(f'N samples     = {n_samples}')

    X_slice, Y = load_test_data(x_path, y_path, START, SIG_LENGTH, n_samples)

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units='ns').start())
    await reset(dut)

    orders = ['training', 'rtl'] if order_mode == 'both' else [order_mode]
    results = {}
    for od in orders:
        results[od] = await run_one_ordering(dut, X_slice, Y, cfg, od)
        # Short gap + reset between orderings so state is clean
        await reset(dut)

    # Decide pass/fail based on the best of the two orderings
    best = max(results.values())
    best_order = max(results, key=results.get)
    dut._log.info(f'best ordering = {best_order!r}  fidelity = {best:.3f}%')

    if order_mode == 'both':
        dut._log.info('>>> Use the winning ordering in your RTL packing. <<<')

    assert best >= cfg['sw_fid'] - 0.5, (
        f'Best core fidelity {best:.3f}% (order={best_order}) is >0.5pp below '
        f'software target {cfg["sw_fid"]:.3f}%.  Check bit ordering.'
    )
