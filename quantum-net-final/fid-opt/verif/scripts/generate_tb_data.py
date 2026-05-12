#!/usr/bin/env python3
"""
generate_tb_data.py  (fid-opt)
-------------------------------
Generates logicnet_stimulus.hex and labels.hex for tb_logicnet.sv.

Pipeline matching the training code exactly:
  1. Arithmetic right-shift I/Q by shift_m=7
  2. Accumulate over window (window_size = sig_length // num_filter = 200)
  3. Arithmetic right-shift accumulation by shift_n=1
  4. Clip to feature_bw=14 bits (mask with (1<<14)-1)
  5. Convert to bit-vector with features_to_bitarray (MSB-first per feature)
  6. Map flat bit k -> M0[k]  => bitrev each feature, place at slot k*bw

Feature order: [I0, I1, Q0, Q1]  (same order training code flattens feats)
Result: M0[0..13]=bitrev(I0,14), M0[14..27]=bitrev(I1,14),
        M0[28..41]=bitrev(Q0,14), M0[42..55]=bitrev(Q1,14)
"""
import numpy as np
import os

# ---- Config (fid-opt, fidelity target) ----
NUM_TESTS   = 10000
START       = 100
SIG_LENGTH  = 400
NUM_FILTER  = 2
SHIFT_M     = 7
SHIFT_N     = 1
FEAT_BW     = 14       # feature_bitwidth(14, 400, 2, 7, 1) = 14
M0_BITS     = FEAT_BW * 2 * NUM_FILTER   # = 56

DATA_DIR  = "/mnt/vault1/mfaroo19/quantum-logicnets/experiments/binary_classification/qick_data"
X_FILE    = os.path.join(DATA_DIR, "0528_X_test_0_770.npy")
Y_FILE    = os.path.join(DATA_DIR, "0528_y_test_0_770.npy")


def arith_rshift(x, s):
    """Arithmetic right shift on signed int64 array."""
    return x.astype(np.int64) >> int(s)


def bitrev(v, bw):
    """Reverse bw bits: MSB of v -> bit 0 of result."""
    v = int(v) & ((1 << bw) - 1)
    result = 0
    for b in range(bw):
        if (v >> (bw - 1 - b)) & 1:
            result |= (1 << b)
    return result


def pack_m0(i0, i1, q0, q1):
    """Pack four feature integers into a 56-bit M0 word matching features_to_bitarray."""
    bw = FEAT_BW
    mask = (1 << bw) - 1
    return (
        bitrev(i0 & mask, bw)        |    # bits [0 :13]
        (bitrev(i1 & mask, bw) << 14) |   # bits [14:27]
        (bitrev(q0 & mask, bw) << 28) |   # bits [28:41]
        (bitrev(q1 & mask, bw) << 42)     # bits [42:55]
    )


def main():
    print(f"Loading data ...")
    X = np.load(X_FILE, mmap_mode='r').reshape(-1, 770, 2)
    Y = np.load(Y_FILE, mmap_mode='r').reshape(-1).astype(np.int32)

    # Select a mix of 0s and 1s
    idx0 = np.where(Y == 0)[0]
    idx1 = np.where(Y == 1)[0]
    
    n_half = NUM_TESTS // 2
    selected_indices = np.concatenate([idx0[:n_half], idx1[:n_half]])
    
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    X_slice = X[selected_indices, START:START+SIG_LENGTH, :].astype(np.int32)
    Y_slice = Y[selected_indices]

    window = SIG_LENGTH // NUM_FILTER   # = 200
    sig_I  = arith_rshift(X_slice[:, :, 0], SHIFT_M).astype(np.int32)
    sig_Q  = arith_rshift(X_slice[:, :, 1], SHIFT_M).astype(np.int32)

    # Accumulate windows and apply final shift
    feat_I = np.zeros((NUM_TESTS, NUM_FILTER), dtype=np.int64)
    feat_Q = np.zeros((NUM_TESTS, NUM_FILTER), dtype=np.int64)
    for w in range(NUM_FILTER):
        s, e = w * window, (w + 1) * window
        feat_I[:, w] = arith_rshift(np.sum(sig_I[:, s:e], axis=1, dtype=np.int64), SHIFT_N)
        feat_Q[:, w] = arith_rshift(np.sum(sig_Q[:, s:e], axis=1, dtype=np.int64), SHIFT_N)

    print(f"N={NUM_TESTS}  M0={M0_BITS} bits  (FEAT_BW={FEAT_BW})")
    print(f"I range: [{feat_I.min()}, {feat_I.max()}]   Q range: [{feat_Q.min()}, {feat_Q.max()}]")
    print(f"Labels: 0s={int((Y_slice==0).sum())}  1s={int((Y_slice==1).sum())}")

    # Define output path
    data_out = "../../data"
    if not os.path.exists(data_out):
        os.makedirs(data_out)

    with open(os.path.join(data_out, "logicnet_stimulus.hex"), "w") as fs, \
         open(os.path.join(data_out, "labels.hex"), "w") as fl, \
         open(os.path.join(data_out, "axis_stimulus.hex"), "w") as fa:
        for n in range(NUM_TESTS):
            # 1. LogicNet Direct Stimulus
            word = pack_m0(feat_I[n,0], feat_I[n,1], feat_Q[n,0], feat_Q[n,1])
            fs.write(f"{word:014X}\n")
            fl.write(f"{int(Y_slice[n]) & 1}\n")

            # 2. AXI-Stream Stimulus for readout_ip.sv
            for t in range(SIG_LENGTH):
                # Using the original (non-shifted) 14-bit data from X_slice
                i_raw = int(X_slice[n, t, 0]) & 0x3FFF
                q_raw = int(X_slice[n, t, 1]) & 0x3FFF
                axis_word = (i_raw << 18) | (q_raw << 4)
                fa.write(f"{axis_word:08X}\n")

    print(f"Done: {data_out}/logicnet_stimulus.hex, {data_out}/labels.hex, {data_out}/axis_stimulus.hex")

    # Sanity-check sample 0
    n = 0
    word = pack_m0(feat_I[n,0], feat_I[n,1], feat_Q[n,0], feat_Q[n,1])
    print(f"sample[0]: I0={feat_I[n,0]} I1={feat_I[n,1]} Q0={feat_Q[n,0]} Q1={feat_Q[n,1]}")
    print(f"           M0=0x{word:014x}  label={Y_slice[n]}")


if __name__ == "__main__":
    main()
