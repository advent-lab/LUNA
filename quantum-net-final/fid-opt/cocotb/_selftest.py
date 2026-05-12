"""Quick checks for ref_model.py — no RTL, no real data."""
import numpy as np
from ref_model import (
    extract_features_int, feature_bitwidth,
    pack_iq_packet, pack_feature_word_training, pack_feature_word_rtl,
    fidelity,
)

# --- 1. feature_bw sanity ---
assert feature_bitwidth(14, 400, 2, 7, 1) == 14
assert feature_bitwidth(14, 400, 1, 9, 0) == 14   # window=400, (2^5-1)*400=12400, bitlen=14, bw=14
assert feature_bitwidth(14, 400, 2, 9, 1) == 12
print('feature_bw ok')

# --- 2. extract_features_int matches manual calc ---
X = np.zeros((2, 400, 2), dtype=np.int32)
X[0, :200, 0] = 128; X[0, 200:, 0] = 256
X[0, :200, 1] = 64;  X[0, 200:, 1] = 32
X[1] = -X[0]
feats = extract_features_int(X, 2, 7, 1)
# sample 0: I>>7 = [1]*200 + [2]*200, Q>>7 = [0]*200 + [0]*200
# window sums pre-n-shift: I0=200, I1=400, Q0=0, Q1=0
# after >>1:                I0=100, I1=200, Q0=0, Q1=0
expected = np.array([100, 200, 0, 0], dtype=np.int32)
assert np.array_equal(feats[0], expected), f'{feats[0]} != {expected}'
# sample 1: negated → I>>7 on -128 = -1, -256 = -2
#   Q values become -64 and -32.  Arith >>7 of -64 is -1 (not 0, because it's
#   sign-extending); likewise -32>>7 = -1. So Q sums: -200, -200 ; >>1 = -100, -100
#   I sums after >>1: -100, -200
expected_neg = np.array([-100, -200, -100, -100], dtype=np.int32)
assert np.array_equal(feats[1], expected_neg), f'{feats[1]} != {expected_neg}'
print(f'extract_features_int ok: sample0={feats[0]} sample1={feats[1]}')

# --- 3. packers behave differently for nf=2 ---
row = np.array([0x01, 0x02, 0x03, 0x04], dtype=np.int32)  # [I0,I1,Q0,Q1]
bw = 14
w_tr = pack_feature_word_training(row, 2, bw)
w_rt = pack_feature_word_rtl(row, 2, bw)
# training:  [I0 I1 Q0 Q1] MSB->LSB  = (1<<42)|(2<<28)|(3<<14)|4
exp_tr = (1 << 42) | (2 << 28) | (3 << 14) | 4
# rtl:       [Q1 I1 Q0 I0] MSB->LSB  = (4<<42)|(2<<28)|(3<<14)|1
exp_rt = (4 << 42) | (2 << 28) | (3 << 14) | 1
assert w_tr == exp_tr, f'{w_tr:014x} vs {exp_tr:014x}'
assert w_rt == exp_rt, f'{w_rt:014x} vs {exp_rt:014x}'
assert w_tr != w_rt, 'packers should differ for nf=2'
print(f'packers ok: training={w_tr:014x}  rtl={w_rt:014x}  (differ ✓)')

# --- 4. packers match for nf=1 ---
row1 = np.array([0x05, 0x07], dtype=np.int32)  # [I0, Q0]
bw1 = 14
w1_tr = pack_feature_word_training(row1, 1, bw1)
w1_rt = pack_feature_word_rtl(row1, 1, bw1)
# training: [I0 Q0] = (5<<14)|7
# rtl     : [Q0 I0] = (7<<14)|5
assert w1_tr == ((5 << 14) | 7)
assert w1_rt == ((7 << 14) | 5)
assert w1_tr != w1_rt, 'for nf=1 the orderings still differ (I first vs Q first)'
print(f'nf=1 packers ok: training={w1_tr:08x}  rtl={w1_rt:08x}')

# --- 5. two's-comp masking for negative features ---
row_neg = np.array([-1, -2, -3, -4], dtype=np.int32)
w_tr_neg = pack_feature_word_training(row_neg, 2, 14)
# each sub-word is its two's-comp in 14 bits
# -1 & 0x3FFF = 0x3FFF, -2=0x3FFE, -3=0x3FFD, -4=0x3FFC
exp = (0x3FFF << 42) | (0x3FFE << 28) | (0x3FFD << 14) | 0x3FFC
assert w_tr_neg == exp, f'{w_tr_neg:014x} vs {exp:014x}'
print('negative two\'s-comp packing ok')

# --- 6. pack_iq_packet layout ---
w = pack_iq_packet(0x3FFF, 0x0001)
assert ((w >> 18) & 0x3FFF) == 0x3FFF
assert ((w >> 4)  & 0x3FFF) == 0x0001
assert (w & 0xF) == 0
print('pack_iq_packet ok')

# --- 7. fidelity math ---
labels = np.array([0, 0, 0, 1, 1, 1])
preds  = np.array([0, 0, 1, 1, 1, 0])    # P(1|0)=1/3, P(0|1)=1/3
fid, p01, p10, n0, n1 = fidelity(labels, preds)
assert n0 == 3 and n1 == 3
assert abs(p01 - 1/3) < 1e-9 and abs(p10 - 1/3) < 1e-9
assert abs(fid - (1 - 1/3)) < 1e-9
print(f'fidelity ok: {fid:.4f} ({fid*100:.2f}%)')

print('\nall ref_model checks passed')
