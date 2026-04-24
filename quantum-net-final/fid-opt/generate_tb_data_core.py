#!/usr/bin/env python3
"""
generate_tb_data_core.py
------------------------
Generates hex stimulus files for tb_logicnet_core.sv (fid-opt).

Mirrors ref_model.py exactly:
  - Arithmetic right-shift for pre/post-accumulation shifts
  - Two packing orderings: 'training' and 'rtl'
  - feature_bw = 14, total = 56 bits for fidelity config (nf=2, shift_m=7, shift_n=1)

Outputs (in script directory):
  training_stimulus.hex   -- 56-bit words, training ordering
  rtl_stimulus.hex        -- 56-bit words, RTL ordering
  labels.hex              -- 1-bit labels (0 or 1)
"""
import numpy as np
import os, sys, argparse

# ---- Configurations (must match cocotb/test_logicnets_core.py) ---- #
CONFIGS = {
    'fidelity': dict(num_filter=2, shift_m=7, shift_n=1, sw_fid=95.995),
    'area'    : dict(num_filter=1, shift_m=9, shift_n=0, sw_fid=95.929),
    'latency' : dict(num_filter=2, shift_m=9, shift_n=1, sw_fid=95.885),
}
START      = 100
SIG_LENGTH = 400
DATA_DIR   = "/mnt/vault1/mfaroo19/quantum-logicnets/experiments/binary_classification/qick_data"


# ---- Exact copies of ref_model.py helpers ---- #
def arith_rshift(x, s):
    return x.astype(np.int64) >> int(s)


def feature_bitwidth(W_in, sig_length, num_filter, shift, n, filt_type=1):
    window_size = max(1, sig_length // max(1, num_filter))
    if filt_type == 1:
        max_val = int(window_size) * (2 ** max(0, W_in - int(shift)) - 1)
    else:
        max_val = int(window_size) * (2 ** max(0, W_in - int(shift)) - 1) \
                  * (2 ** max(0, n - 1))
    if max_val <= 0:
        return 1
    return max(1, int(max_val).bit_length() - int(n))


def extract_features_int(X_slice, num_filter, shift_m, shift_n):
    """Returns (N, 2*num_filter) array ordered [I0..I(nf-1), Q0..Q(nf-1)]."""
    N, flen, _ = X_slice.shape
    window_size = max(1, flen // max(1, num_filter))
    sig_I = arith_rshift(X_slice[:, :, 0], shift_m).astype(np.int32)
    sig_Q = arith_rshift(X_slice[:, :, 1], shift_m).astype(np.int32)
    feats = np.zeros((N, 2 * num_filter), dtype=np.int32)
    for w in range(num_filter):
        s, e = w * window_size, min((w + 1) * window_size, flen)
        if s >= e:
            continue
        feats[:, w]              = arith_rshift(np.sum(sig_I[:, s:e], axis=1, dtype=np.int64), shift_n)
        feats[:, num_filter + w] = arith_rshift(np.sum(sig_Q[:, s:e], axis=1, dtype=np.int64), shift_n)
    return feats


def bitrev(v, bw):
    """Reverse `bw` bits of integer v. MSB of v -> bit 0 of result."""
    v = int(v) & ((1 << bw) - 1)
    result = 0
    for b in range(bw):
        if (v >> (bw - 1 - b)) & 1:
            result |= (1 << b)
    return result


def pack_as_bitarray(feats_row, feature_order_indices, bw):
    """
    Mirrors the training pipeline exactly:
        bits = features_to_bitarray(feats, bits=bw)   # MSB-first per feature
        X_t  = torch.from_numpy(bits).float()         # flat[0]=MSB of feat[0]
    LogicNets maps flat[i] -> M0[i], where M0[i] is integer bit i (M0[0]=LSB).
    Therefore: each feature's bits are bit-reversed within bw bits, then placed
    at position k*bw inside the integer.

    feature_order_indices: list of indices into the feats array giving the
    order in which features appear in the flat bitarray.
    e.g. 'training' = [0,1,2,3]  ('rtl' varies by design)
    """
    mask = (1 << bw) - 1
    word = 0
    for slot, idx in enumerate(feature_order_indices):
        v = int(feats_row[idx]) & mask
        word |= bitrev(v, bw) << (slot * bw)
    return word


def load_data(x_path, y_path, n_samples=None):
    X = np.load(x_path, mmap_mode='r')
    Y = np.load(y_path, mmap_mode='r')
    if X.ndim == 2:
        X = X.reshape(X.shape[0], -1, 2)
    elif X.ndim > 3:
        X = X.reshape(-1, X.shape[-2], X.shape[-1])
    if Y.ndim == 2 and Y.shape[-1] >= 2:
        Y = np.argmax(Y, axis=-1)
    Y = Y.reshape(-1).astype(np.int32)
    end = START + SIG_LENGTH
    X_slice = X[:, START:end, :].astype(np.int32)
    if n_samples:
        X_slice = X_slice[:n_samples]
        Y       = Y[:n_samples]
    return X_slice, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',    choices=list(CONFIGS), default='fidelity')
    ap.add_argument('--n-samples', type=int, default=None)
    ap.add_argument('--data-dir',  default=DATA_DIR)
    ap.add_argument('--out-dir',   default='.')
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    nf, sm, sn = cfg['num_filter'], cfg['shift_m'], cfg['shift_n']
    bw  = feature_bitwidth(14, SIG_LENGTH, nf, sm, sn, filt_type=1)
    total = bw * 2 * nf
    print(f"Config={args.config}  nf={nf}  shift_m={sm}  shift_n={sn}  bw={bw}  total={total} bits")

    ddir = os.path.abspath(args.data_dir)
    X_slice, Y = load_data(
        os.path.join(ddir, '0528_X_test_0_770.npy'),
        os.path.join(ddir, '0528_y_test_0_770.npy'),
        args.n_samples,
    )
    N = X_slice.shape[0]
    print(f"Loaded: N={N}  labels: 0s={int((Y==0).sum())}  1s={int((Y==1).sum())}")

    feats = extract_features_int(X_slice, nf, sm, sn)
    print(f"Features: shape={feats.shape}  range I=[{feats[:,:nf].min()},{feats[:,:nf].max()}]  Q=[{feats[:,nf:].min()},{feats[:,nf:].max()}]")

    # Feature index order for two packers:
    # 'training': feats = [I0, I1, Q0, Q1] — training code feeds them in this order
    # 'rtl':      readout_ip.sv wires { Q1, I1, Q0, I0 } directly to M0
    #             (no bitarray transform — raw ints), but here we apply bitarray
    #             with index order [Q1, I1, Q0, I0] = [nf+nf-1, nf-1, nf, 0]
    nf = cfg['num_filter']
    training_order = list(range(2 * nf))             # [0,1,2,3] = [I0,I1,Q0,Q1]
    # RTL concatenation {Q(nf-1),..,Q0,I(nf-1),..,I0} → read order (LSB-first):
    # slot0=I0, slot1=Q0, slot2=I1, slot3=Q1  (interleaved, I before Q)
    rtl_order = []
    for w in range(nf):
        rtl_order.append(w)          # I_w
        rtl_order.append(nf + w)     # Q_w

    hex_width = (total + 3) // 4

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    f_tr  = open(os.path.join(out, 'training_stimulus.hex'), 'w')
    f_rtl = open(os.path.join(out, 'rtl_stimulus.hex'),      'w')
    f_lbl = open(os.path.join(out, 'labels.hex'),             'w')

    for n in range(N):
        w_tr  = pack_as_bitarray(feats[n], training_order, bw) & ((1 << total) - 1)
        w_rtl = pack_as_bitarray(feats[n], rtl_order,      bw) & ((1 << total) - 1)
        f_tr .write(f"{w_tr :{hex_width}X}\n".replace(' ', '0'))
        f_rtl.write(f"{w_rtl:{hex_width}X}\n".replace(' ', '0'))
        f_lbl.write(f"{int(Y[n]) & 1:01X}\n")

    f_tr.close(); f_rtl.close(); f_lbl.close()
    print(f"Written: training_stimulus.hex  rtl_stimulus.hex  labels.hex  ({N} samples)")
    print(f"SW target fidelity: {cfg['sw_fid']:.3f}%")
    print(f"training order indices: {training_order}")
    print(f"rtl      order indices: {rtl_order}")

    w_tr  = pack_as_bitarray(feats[0], training_order, bw) & ((1 << total) - 1)
    w_rtl = pack_as_bitarray(feats[0], rtl_order,      bw) & ((1 << total) - 1)
    print(f"sample[0] feats={list(feats[0])}")
    print(f"  training hex = 0x{w_tr:014x}")
    print(f"  rtl      hex = 0x{w_rtl:014x}")
    if w_tr == w_rtl:
        print("  (packings identical)")
    else:
        print("  (packings DIFFER — both will be tested by the SV testbench)")


if __name__ == '__main__':
    main()
