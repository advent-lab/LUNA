#!/usr/bin/env python3
"""
run_python_only.py
------------------
Pure-python sanity check of the preprocessing pipeline. Reads the QICK test
data, runs extract_features_int, and reports the software fidelity achievable
with a simple maximum-magnitude-I classifier baseline (and, if a trained
LogicNets weights file were available, it would run inference here too).

This exists so you can verify on a Linux server that:
  1. DATA_DIR is correct and the .npy files load
  2. The reference model agrees with the 95.995 % / 95.929 % / 95.885 % targets
     from the spec (using whatever baseline classifier you hook in)
  3. The feature shapes, bitwidths, and label distribution look sane

It does NOT run the RTL. For that, use `make -f Makefile.cocotb ...`.

Usage:
  python3 run_python_only.py --data-dir /scratch/qick_data
  python3 run_python_only.py --data-dir /scratch/qick_data --config area
  python3 run_python_only.py --data-dir /scratch/qick_data --n-samples 1024
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np

# Make ref_model importable whether we're invoked from cocotb_tb/ or elsewhere
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ref_model import (
    load_test_data, extract_features_int, feature_bitwidth,
    pack_feature_word_training, pack_feature_word_rtl, fidelity,
)


CONFIGS = {
    'fidelity': dict(num_filter=2, shift_m=7, shift_n=1, sw_fid=95.995),
    'area'    : dict(num_filter=1, shift_m=9, shift_n=0, sw_fid=95.929),
    'latency' : dict(num_filter=2, shift_m=9, shift_n=1, sw_fid=95.885),
}

START = 100
SIG_LENGTH = 400


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data-dir', required=True,
                    help='Path to qick_data/ containing 0528_*_0_770.npy')
    ap.add_argument('--config', choices=list(CONFIGS.keys()), default='fidelity')
    ap.add_argument('--n-samples', type=int, default=None,
                    help='Truncate the test set (default: use all)')
    ap.add_argument('--x-name', default='0528_X_test_0_770.npy')
    ap.add_argument('--y-name', default='0528_y_test_0_770.npy')
    args = ap.parse_args()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    x_path = os.path.join(data_dir, args.x_name)
    y_path = os.path.join(data_dir, args.y_name)
    for p in (x_path, y_path):
        if not os.path.isfile(p):
            sys.exit(f'ERROR: missing file {p}')

    cfg = CONFIGS[args.config]
    nf, sm, sn = cfg['num_filter'], cfg['shift_m'], cfg['shift_n']

    print(f'--- run_python_only ---')
    print(f'config     : {args.config}')
    print(f'data_dir   : {data_dir}')
    print(f'nf={nf}  shift_m={sm}  shift_n={sn}')
    print(f'SW target  : {cfg["sw_fid"]:.3f} %')

    X_slice, Y = load_test_data(x_path, y_path, START, SIG_LENGTH, args.n_samples)
    N = X_slice.shape[0]
    print(f'loaded     : X={X_slice.shape} Y={Y.shape} (N={N})')
    print(f'labels     : 0s={int((Y == 0).sum())}  1s={int((Y == 1).sum())}')

    feats = extract_features_int(X_slice, nf, sm, sn)
    bw = feature_bitwidth(14, SIG_LENGTH, nf, sm, sn, filt_type=1)
    total = bw * 2 * nf
    print(f'features   : shape={feats.shape}  bw={bw}  total={total} bits '
          f'(M0 is 56 bits)')
    print(f'feat range : I=[{feats[:, :nf].min()}, {feats[:, :nf].max()}]  '
          f'Q=[{feats[:, nf:].min()}, {feats[:, nf:].max()}]')

    # A trivial baseline: threshold on sum_I0. Tells you nothing about LogicNets
    # accuracy but it confirms the features carry signal. Real fidelity = cfg['sw_fid']
    # and only the trained LogicNets net reproduces it.
    thr = np.median(feats[:, 0])
    preds_baseline = (feats[:, 0] > thr).astype(np.int32)
    for flip in (0, 1):
        p = preds_baseline ^ flip
        f, p01, p10, n0, n1 = fidelity(Y, p)
        print(f'baseline(I0>med, flip={flip}) : '
              f'fid={f*100:.3f}%  P(1|0)={p01:.3f}  P(0|1)={p10:.3f}')

    # Show both packings for sample 0 so you can compare against RTL traces
    w_tr = pack_feature_word_training(feats[0], nf, bw)
    w_rt = pack_feature_word_rtl    (feats[0], nf, bw)
    print(f'sample[0]  : feats={list(feats[0])}')
    print(f'           : pack_training = 0x{w_tr:014x}')
    print(f'           : pack_rtl      = 0x{w_rt:014x}')
    if w_tr == w_rt:
        print('           : (packings identical for this config)')
    else:
        print('           : (packings DIFFER — only one matches the trained net)')

    print(f'\nTo evaluate the real LogicNets fidelity, run:')
    print(f'  make -f Makefile.cocotb TEST=core DATA_DIR={data_dir} '
          f'CONFIG={args.config}')


if __name__ == '__main__':
    main()
