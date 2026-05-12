"""
ref_model.py
------------
Pure-python reference model for the QICK readout classifier. Matches the
training-time preprocessing (extract_features_int) bit-for-bit, and provides
packing helpers for the 32-bit I/Q AXIS word and the LogicNets feature vector.

IMPORTANT: sub-word ordering in the feature vector
--------------------------------------------------
The training code does:
    feats.append(np.concatenate([f_I, f_Q]))   # [I0, I1, ..., Q0, Q1, ...]
and then bit-packs MSB-first per sub-word:
    [I0_msb ... I0_lsb | I1_msb ... | ... | Q0 ... | ... ]

This module exposes TWO packers so you can check both against the RTL:
  - pack_feature_word_training : matches the training order (I0..I(nf-1) | Q0..Q(nf-1))
  - pack_feature_word_rtl      : matches readout_ip.sv concatenation (Q(nf-1),I(nf-1),...,Q0,I0)

For num_filter=2, these differ. Feed BOTH into tb_logicnets_core and see
which one reproduces the software fidelity — that tells you which ordering
was actually baked into the LogicNets truth-tables.
"""

from __future__ import annotations
import numpy as np


# ------------------------ bit math ------------------------ #
def arith_rshift(x: np.ndarray, s: int) -> np.ndarray:
    """NumPy >> on signed ints is arithmetic shift."""
    return x.astype(np.int64) >> int(s)


def feature_bitwidth(W_in: int, sig_length: int, num_filter: int,
                     shift: int, n: int, filt_type: int = 1) -> int:
    window_size = max(1, sig_length // max(1, num_filter))
    if filt_type == 1:
        max_val = int(window_size) * (2 ** max(0, (W_in - int(shift))) - 1)
    else:
        max_val = int(window_size) * (2 ** max(0, (W_in - int(shift))) - 1) \
                  * (2 ** max(0, (n) - 1))
    if max_val <= 0:
        return 1
    bitlen = int(max_val).bit_length()
    return max(1, bitlen - int(n))


# ------------------------ feature extraction ------------------------ #
def extract_features_int(X_slice: np.ndarray, num_filter: int,
                         shift_m: int, shift_n: int) -> np.ndarray:
    """
    Mirrors the training preprocessing exactly.
    Input:  X_slice of shape (N, sig_length, 2)  where [..., 0]=I, [..., 1]=Q
    Output: feats of shape (N, 2*num_filter), ordered [I0..I(nf-1), Q0..Q(nf-1)]
    """
    N, flen, _ = X_slice.shape
    window_size = max(1, flen // max(1, num_filter))

    sig_I = arith_rshift(X_slice[:, :, 0], shift_m).astype(np.int32)
    sig_Q = arith_rshift(X_slice[:, :, 1], shift_m).astype(np.int32)

    feats = np.zeros((N, 2 * num_filter), dtype=np.int32)
    for w in range(num_filter):
        s, e = w * window_size, min((w + 1) * window_size, flen)
        if s >= e:
            continue
        feats[:, w] = arith_rshift(
            np.sum(sig_I[:, s:e], axis=1, dtype=np.int64), shift_n)
        feats[:, num_filter + w] = arith_rshift(
            np.sum(sig_Q[:, s:e], axis=1, dtype=np.int64), shift_n)
    return feats


# ------------------------ AXIS packet ------------------------ #
def pack_iq_packet(i14: int, q14: int) -> int:
    """Build the 32-bit AXIS word: [31:18]=I, [17:4]=Q, [3:0]=0."""
    return ((int(i14) & 0x3FFF) << 18) | ((int(q14) & 0x3FFF) << 4)


# ------------------------ feature word packers ------------------------ #
def pack_feature_word_training(feats_row: np.ndarray, num_filter: int,
                               feature_bw: int) -> int:
    """
    Ordering: [I0, I1, ..., I(nf-1), Q0, Q1, ..., Q(nf-1)]
    First element in the feature vector becomes the MSB of the packed word,
    which is how features_to_bitarray + flatten feeds LogicNets training.
    """
    mask = (1 << feature_bw) - 1
    word = 0
    # features_to_bitarray walks features[0], features[1], ... and places bits
    # in order with MSB of features[0] first. In a Verilog {a, b, c} concat,
    # `a` occupies the upper bits. So features[0] is MSB-most.
    for k in range(2 * num_filter):
        sub = int(feats_row[k]) & mask
        word = (word << feature_bw) | sub
    return word


def pack_feature_word_rtl(feats_row: np.ndarray, num_filter: int,
                          feature_bw: int) -> int:
    """
    Ordering that matches readout_ip.sv concatenation:
        { Q(nf-1), I(nf-1), ..., Q0, I0 }
    I0 occupies the LSB; alternating Q, I pattern going up.
    """
    mask = (1 << feature_bw) - 1
    word = 0
    # Build MSB -> LSB
    for w in range(num_filter - 1, -1, -1):
        for ch in ('Q', 'I'):
            idx = (num_filter + w) if ch == 'Q' else w
            sub = int(feats_row[idx]) & mask
            word = (word << feature_bw) | sub
    return word


PACKERS = {
    'training': pack_feature_word_training,
    'rtl':      pack_feature_word_rtl,
}


# ------------------------ convenience loader ------------------------ #
def load_test_data(x_path: str, y_path: str, start: int, sig_length: int,
                   max_samples: int | None = None):
    """Load and slice the QICK test data to (N, sig_length, 2) and (N,)."""
    X = np.load(x_path, mmap_mode='r')
    Y = np.load(y_path, mmap_mode='r')

    # Handle flat layout: (N, T*2) -> (N, T, 2)
    if X.ndim == 2:
        X = X.reshape(X.shape[0], -1, 2)
    elif X.ndim > 3:
        X = X.reshape(-1, X.shape[-2], X.shape[-1])
    if Y.ndim == 2 and Y.shape[-1] >= 2:
        Y = np.argmax(Y, axis=-1)
    Y = np.asarray(Y).reshape(-1).astype(np.int32)

    end = start + sig_length
    if X.shape[1] < end:
        raise ValueError(f'X has only {X.shape[1]} time samples, need >= {end}')
    X_slice = X[:, start:end, :].astype(np.int32)

    N = X_slice.shape[0]
    if Y.shape[0] != N:
        raise ValueError(f'X/Y mismatch: {N} vs {Y.shape[0]}')

    if max_samples is not None and max_samples < N:
        X_slice = X_slice[:max_samples]
        Y = Y[:max_samples]
    return X_slice, Y


# ------------------------ fidelity ------------------------ #
def fidelity(labels: np.ndarray, preds: np.ndarray):
    """Returns (fidelity, P(1|0), P(0|1), N0, N1)."""
    labels = np.asarray(labels).astype(int)
    preds  = np.asarray(preds).astype(int)
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    e01 = int(((labels == 0) & (preds == 1)).sum())
    e10 = int(((labels == 1) & (preds == 0)).sum())
    p01 = e01 / n0 if n0 else 0.0
    p10 = e10 / n1 if n1 else 0.0
    return 1.0 - (p01 + p10) / 2.0, p01, p10, n0, n1
