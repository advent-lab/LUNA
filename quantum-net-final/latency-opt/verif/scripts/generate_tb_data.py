#!/usr/bin/env python3
import numpy as np
import os

# Configuration based on `latency-opt` architecture
NUM_TESTS = 10000
START = 100
FLEN = 400
NUM_FILTER = 2  # NUM_WINDOWS
SHIFT_M = 9
SHIFT_N = 1
IQ_WIDTH_IN = 14
DATA_DIR = "/mnt/vault1/mfaroo19/quantum-logicnets/experiments/binary_classification/qick_data/"
X_TEST_FNAME = os.path.join(DATA_DIR, '0528_X_test_0_770.npy')
Y_TEST_FNAME = os.path.join(DATA_DIR, '0528_y_test_0_770.npy')

def limit_width(val, bits):
    """Truncate integer to specified bit width."""
    mask = (1 << bits) - 1
    return val & mask

def bitrev(v, bw):
    """Reverse bw bits: MSB of v -> bit 0 of result (matches features_to_bitarray)."""
    v = int(v) & ((1 << bw) - 1)
    result = 0
    for b in range(bw):
        if (v >> (bw - 1 - b)) & 1:
            result |= (1 << b)
    return result

def main():
    print(f"Loading data from {DATA_DIR}...")
    X = np.load(X_TEST_FNAME, mmap_mode='r')
    Y = np.load(Y_TEST_FNAME, mmap_mode='r')

    print("Original X shape:", X.shape)
    print("Original Y shape:", Y.shape)

    # The raw shape is (100000, 1540). Reshape to (N, 770, 2)
    X = X.reshape(X.shape[0], -1, 2)
    Y = Y.reshape(-1)
    
    # Select a mix of 0s and 1s
    idx0 = np.where(Y == 0)[0]
    idx1 = np.where(Y == 1)[0]
    
    # Take 500 of each
    n_half = NUM_TESTS // 2
    selected_indices = np.concatenate([idx0[:n_half], idx1[:n_half]])
    
    # Shuffle for a realistic mix
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    X_test = X[selected_indices]
    Y_test = Y[selected_indices]

    end = START + FLEN
    window_size = FLEN // max(1, NUM_FILTER)

    print(f"Generating SystemVerilog Testbench Data for {NUM_TESTS} samples...")
    
    # Define output path
    data_out = "../../data"
    if not os.path.exists(data_out):
        os.makedirs(data_out)

    with open(os.path.join(data_out, "axis_stimulus.hex"), "w") as f_axis, \
         open(os.path.join(data_out, "logicnet_stimulus.hex"), "w") as f_logicnet, \
         open(os.path.join(data_out, "labels.hex"), "w") as f_labels:
        
        for i in range(NUM_TESTS):
            # 1. Axis Stimulus for readout_ip.sv
            for t in range(START, end):
                I_raw = int(X_test[i, t, 0])
                Q_raw = int(X_test[i, t, 1])
                I_14b = limit_width(I_raw, IQ_WIDTH_IN)
                Q_14b = limit_width(Q_raw, IQ_WIDTH_IN)
                packed_word = (I_14b << 18) | (Q_14b << 4)
                f_axis.write(f"{packed_word:08X}\n")
            
            # Write expected label
            label = int(Y_test[i])
            f_labels.write(f"{limit_width(label, 2):01X}\n")

            # 2. LogicNet Stimulus for logicnets.sv
            I_shifted = [int(x) >> SHIFT_M for x in X_test[i, START:end, 0]]
            Q_shifted = [int(x) >> SHIFT_M for x in X_test[i, START:end, 1]]
            
            sum_I = []
            sum_Q = []
            
            for w in range(NUM_FILTER):
                s = w * window_size
                e = (w + 1) * window_size
                acc_I = sum(I_shifted[s:e])
                acc_Q = sum(Q_shifted[s:e])
                acc_I_final = limit_width(acc_I >> SHIFT_N, 12)
                acc_Q_final = limit_width(acc_Q >> SHIFT_N, 12)
                sum_I.append(acc_I_final)
                sum_Q.append(acc_Q_final)

            # feature_bw for latency-opt: feature_bitwidth(14,400,2,9,1) = 12 bits per feature
            # Training order: [I0, I1, Q0, Q1] mapped via features_to_bitarray (MSB-first per feature)
            # LogicNets maps flat_bit[k] -> M0[k], so each feature is bit-reversed in its slot.
            # M0[0..11]  = bitrev(I0, 12)
            # M0[12..23] = bitrev(I1, 12)
            # M0[24..35] = bitrev(Q0, 12)
            # M0[36..47] = bitrev(Q1, 12)
            FEAT_BW = 12
            logicnet_word = (
                bitrev(sum_I[0], FEAT_BW)        |   # M0[0:12]   = bitrev(I0)
                (bitrev(sum_I[1], FEAT_BW) << 12) |  # M0[12:24]  = bitrev(I1)
                (bitrev(sum_Q[0], FEAT_BW) << 24) |  # M0[24:36]  = bitrev(Q0)
                (bitrev(sum_Q[1], FEAT_BW) << 36)    # M0[36:48]  = bitrev(Q1)
            )
            f_logicnet.write(f"{logicnet_word:012X}\n")
            
    print(f"Test vectors generated: {data_out}/axis_stimulus.hex, {data_out}/logicnet_stimulus.hex, {data_out}/labels.hex")

if __name__ == "__main__":
    main()
