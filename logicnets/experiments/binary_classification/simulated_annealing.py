# -----------------------
# Standard library
# -----------------------
import os
import math
import random
import time
import hashlib
import json

# -----------------------
# Scientific stack
# -----------------------
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import yaml
import argparse
import copy

# -----------------------
# PyTorch
# -----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
from train import train_quiet, test_quiet
from models import QuantumNeqModel


# Full simulated annealing integrating logicnets + extractors + resource models
import os
import math, random, time
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
#from logicnets_train_utils import train_logicnet, UnswNb15NeqModel
import torch.nn.init as init
from torch.nn import Parameter


# ----------------------
# Environment: keep your existing settings
# ----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ----------------------
# User data (must exist in your session)
# ----------------------
# Expectation: data_X_new and test_X_new are already loaded (same names as prior cells)
start_window = 0
end_window = 770
data_dir = "./qick_data"
#assert os.path.exists(f"{data_dir}/0528_X_train_{start_window}_{end_window}.npy"), "File does not exist "

x_train_path = os.path.join(data_dir, f'0528_X_train_{start_window}_{end_window}.npy')
y_train_path = os.path.join(data_dir, f'0528_y_train_{start_window}_{end_window}.npy')

data_X = np.load(x_train_path)
data_Y = np.load(y_train_path)

# Insure same dataset is loaded 
assert hashlib.md5(data_X).hexdigest() == 'b61226c86b7dee0201a9158455e08ffb',  "Checksum failed. Wrong file was loaded or file may be corrupted."
assert hashlib.md5(data_Y).hexdigest() == 'c59ce37dc7c73d2d546e7ea180fa8d31',  "Checksum failed. Wrong file was loaded or file may be corrupted."

print("Train Data Set:")
print("\tX Path        :", x_train_path)
print("\ty Path        :", y_train_path)
print("\tSize          :", len(data_X))
print("\tSample Shape  :", data_X[0].shape)
print("\tMean          :", data_X.mean())
print("\tStd. Dev.     :", data_X.std())

assert len(data_X[0]) == (end_window-start_window)*2, "ERROR: Specified window does not match loaded dataset shape"

data_X_new = data_X.reshape(90, 2, 5000, 770, 2)
data_Y_new = data_Y.reshape(90, 2, 5000)

file_index = 0

start_window = 0
end_window = 770
data_dir = "./qick_data"
#assert os.path.exists(f"{data_dir}/X_test_{start_window}_{end_window}.npy"), "File does not exist "

x_test_path = os.path.join(data_dir, f'0528_X_test_{start_window}_{end_window}.npy')
y_test_path = os.path.join(data_dir, f'0528_y_test_{start_window}_{end_window}.npy')

test_X = np.load(x_test_path)
test_Y = np.load(y_test_path)

# Insure same dataset is loaded 
assert hashlib.md5(test_X).hexdigest() == 'b7d85f42522a0a57e877422bc5947cde', "Checksum failed. Wrong file was loaded or file may be corrupted."
assert hashlib.md5(test_Y).hexdigest() == '8c9cce1821372380371ade5f0ccfd4a2', "Checksum failed. Wrong file was loaded or file may be corrupted."

print("Test Data Set:")
print("\tX Path        :", x_test_path)
print("\ty Path        :", y_test_path)
print("\tSize         :", len(test_X))
print("\tSample Shape :", test_X[0].shape)
print("\tSample Shape :", test_X.mean())
print("\tStd. Dev.    :", test_X.std())

assert len(test_X[0]) == (end_window-start_window)*2, "ERROR: Specified window does not match loaded dataset shape"

# index = [file_index, state_index, shot_index, time_index, ADC_index]
test_X_new = test_X.reshape(10, 2, 5000, 770, 2)
test_Y_new = test_Y.reshape(10, 2, 5000)

file_index = 0

signal_len = 750
data_X_trunc = data_X_new[:, :, :, :signal_len, :]  # (N,2,M,T,2)
test_X_trunc = test_X_new[:, :, :, :signal_len, :]

output_csv = "simulated_annealing_results_integrator_logicnet.csv"

# ----------------------
# Feature extractors (integrator + matched filter)
# ----------------------
def compute_weights(X, start, end):
    # compute simple matched-filter weights using class means (same as you used)
    mean_Ig = X[:,0,:,start:end,0].mean(axis=(0,1))
    mean_Ie = X[:,1,:,start:end,0].mean(axis=(0,1))
    mean_Qg = X[:,0,:,start:end,1].mean(axis=(0,1))
    mean_Qe = X[:,1,:,start:end,1].mean(axis=(0,1))
    weight_I = np.nan_to_num(np.abs(mean_Ig - mean_Ie) / (np.var(mean_Ig) + np.var(mean_Ie) + 1e-12))
    weight_Q = np.nan_to_num(np.abs(mean_Qg - mean_Qe) / (np.var(mean_Qg) + np.var(mean_Qe) + 1e-12))
    return weight_I, weight_Q

def quantize_weights(weights, n):
    scale = 1 << int(n)
    return np.round(weights * scale).astype(np.int32)

def extract_features_mf(X, start, flen, weight_I, weight_Q, n, shift, num_filter):
    # matched filter -> dot products over windows
    feats, labels = [], []
    end = start + flen
    int_weight_I = quantize_weights(weight_I, n)
    int_weight_Q = quantize_weights(weight_Q, n)
    window_size = max(1, flen // max(1, num_filter))

    for sample in range(X.shape[0]):
        for state in [0, 1]:
            for m in range(X.shape[2]):
                sig_I = (X[sample, state, m, start:end, 0].astype(np.int32) >> int(shift))
                sig_Q = (X[sample, state, m, start:end, 1].astype(np.int32) >> int(shift))
                f_I, f_Q = [], []
                for w in range(num_filter):
                    s, e = w*window_size, min((w+1)*window_size, flen)
                    if s >= e:
                        val_I = 0
                        val_Q = 0
                    else:
                        val_I = (int(np.dot(sig_I[s:e], int_weight_I[s:e])) >> int(n))
                        val_Q = (int(np.dot(sig_Q[s:e], int_weight_Q[s:e])) >> int(n))
                    f_I.append(val_I)
                    f_Q.append(val_Q)
                feats.append(np.concatenate([f_I, f_Q]))
                labels.append(state)
    return np.array(feats, dtype=np.int32), np.array(labels, dtype=np.int32)

def extract_features_int(X, start, flen, n, shift, num_filter):
    feats, labels = [], []
    end = start + flen
    window_size = max(1, flen // max(1, num_filter))

    for sample in range(X.shape[0]):
        for state in [0, 1]:
            for m in range(X.shape[2]):
                sig_I = (X[sample, state, m, start:end, 0].astype(np.int32) >> int(shift))
                sig_Q = (X[sample, state, m, start:end, 1].astype(np.int32) >> int(shift))
                f_I, f_Q = [], []
                for w in range(num_filter):
                    s, e = w*window_size, min((w+1)*window_size, flen)
                    if s >= e:
                        sum_I = 0
                        sum_Q = 0
                    else:
                        sum_I = (int(np.sum(sig_I[s:e])) >> int(n))
                        sum_Q = (int(np.sum(sig_Q[s:e])) >> int(n))
                    f_I.append(sum_I)
                    f_Q.append(sum_Q)
                feats.append(np.concatenate([f_I, f_Q]))
                labels.append(state)
    return np.array(feats, dtype=np.int32), np.array(labels, dtype=np.int32)

def features_to_bitarray(features, bits):
    bit_features = []   
    for feat in features:
        bit_list = []
        for x in feat:
            x_i = int(x)
            # two’s complement representation
            val = x_i & ((1 << bits)-1)
            bin_str = f"{val:0{bits}b}"
            bit_list.extend([b == "1" for b in bin_str])
        bit_features.append(bit_list)
    return np.array(bit_features, dtype=int)



# ----------------------
# NN fidelity trainer
# ----------------------
def train_model(X_train, Y_train, X_test, Y_test, input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train, Y_train, epochs=3, batch_size=512, validation_data=(X_test, Y_test), verbose=0)
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    P_1_given_0 = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    P_0_given_1 = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fidelity = 1.0 - 0.5*(P_1_given_0 + P_0_given_1)   # fraction in [0,1]
    return fidelity

# ----------------------
# Resource predictors (your functions integrated)
# ----------------------
def predict_resources_dp(M, W, N, fmax=0.66):
    lut_coef = np.array([2.65320370e+02, -1.0, -1.0, -2.205e+01, -1.7525e+01, 2.66453526e-15, 1.5375])
    lut_intercept = -4.196296296285368
    dsp_coef = np.array([-3.21666667e+00, 2.35922393e-15, 8.32667268e-15, 3.66666667e-01, 2.87500000e-01, -2.49800181e-16, -2.5e-02])
    dsp_intercept = -8.881784197001252e-14
    features = np.array([M, W, N, M*W, M*N, W*N, M*W*N])
    luts = float(lut_intercept + lut_coef @ features)
    dsps = float(dsp_intercept + dsp_coef @ features)
    luts = max(luts, 0.0); dsps = max(dsps, 0.0)
    latency_cycles = math.ceil(math.log2(max(2, M))) + 1
    clk_ns = 1e3 / (fmax * 1e3)
    latency_ns = latency_cycles * clk_ns
    return {"LUTs": luts, "DSPs": dsps, "latency_cycles": latency_cycles, "latency_ns": latency_ns, "type": "dp"}

def predict_resources_integrator(M, W, fmax=0.66):
    lut_intercept = 99.550142
    lut_coef = np.array([3.135467, 2.061605, 0.159635])
    features = np.array([M, W, M*W])
    luts = float(lut_intercept + lut_coef @ features)
    luts = max(luts, 0.0)
    latency_cycles = math.ceil(math.log2(max(2, M)))
    clk_ns = 1e3 / (fmax * 1e3)
    latency_ns = latency_cycles * clk_ns
    return {"LUTs": luts, "DSPs": 0.0, "latency_cycles": latency_cycles, "latency_ns": latency_ns, "type": "integrator"}

# ----------------------
# Logicnet LUT/fmax proxy (s_lut and power-law fit constants from earlier)
# ----------------------
s_lut_default = 0.0916561
A_default = 6705.366569019219
b_default = -0.106784160235
c_default = -0.164541971364

def luts_logicnet_proxy(total_raw, s_lut=s_lut_default):
    return s_lut * total_raw

def fmax_logicnet_proxy(max_raw, total_raw, A=A_default, b=b_default, c=c_default):
    return 660.0

def luts_per_neuron_raw(X, Y):
    X = int(X); Y = int(Y)
    if X >= 5:
        term = (2 ** (X - 4)) - ((-1) ** X)
        return (Y / 3.0) * term
    else:
        return (Y)/3.0

def raw_LUTs_from_spec(model_layers, feature_input_bw, in_fanin, hid_bw, hid_fanin, out_bw, out_fanin):
    total_raw = 0.0; max_per_neuron = 0.0
    for i, n_neurons in enumerate(model_layers):
        if i == 0:
            X = feature_input_bw * in_fanin
            Y = in_fanin
        else:
            X = hid_bw * hid_fanin
            Y = hid_fanin
        per = luts_per_neuron_raw(X, Y)
        total_raw += float(n_neurons) * per
        max_per_neuron = max(max_per_neuron, per)
    total_raw += luts_per_neuron_raw(out_bw*out_fanin,1)
    return float(total_raw), float(max_per_neuron)

# ----------------------
# Helpers: compute S and feature-bitwidth
# ----------------------
def feature_bitwidth_from_preproc(W_in, sig_length, num_filter, shift, n, filt_type=1):
    # estimated bits per integrator output using rough model
    window_size = sig_length // max(1, num_filter)
    if filt_type == 1:
        max_val = int(window_size) * (2 ** max(0, (W_in - int(shift))) - 1)
    else:
        max_val = int(window_size) * (2 ** max(0, (W_in - int(shift))) - 1) * (2 ** max(0, (n) - 1) ) 
    
    if max_val <= 0:
        return 1
    
    bitlen = int(max_val).bit_length()
    bw = max(1, bitlen - int(n))
    return bw

def compute_S_from_config(config, feature_bw):
    # total input bits to logicnets / output bits from preprocessing
    input_neurons = config["layers"][0]
    total_input_bits = config["beta_i"] * config["gamma_i"] * input_neurons
    pre_out_bits = (config["num_filter"] * 2) * feature_bw  # each feature has feature_bw bits, times num_filter windows times I/Q
    return float(total_input_bits) / max(1.0, pre_out_bits)


def latency_from_config(hidden_layers, num_filter, sig_length, filt_type=1):
    base = len(hidden_layers) + 1
    win_len = max(1, sig_length // max(1, num_filter))
    log_term = math.ceil(math.log2(win_len))
    extra = 1 if filt_type == 2 else 0
    return base + log_term + extra

# ----------------------
# Cost function (no freq penalty, DSP weighted more heavily)
# ----------------------
def compute_cost(result, num_windows, weights, debug):
    """
    cost = weighted sum over:
      - fidelity (inverted)
      - area (extractor LUT/DSP scaled by num_windows + logicnets proxy)
      - latency
    """
    w_fid, w_area, w_l = weights

    fidelity = float(result["fidelity"])
    fidelity = max(1e-6, fidelity)

    # fidelity penalty
    fid_pen = (1-fidelity)/(1-0.9) #scale [0.9, 1] to [0, 1]

    # latency penalty
    layers = result["logic_spec"]["layers"]
    preproc_latency = int(result.get("latency_cycles"))

    latency = len(layers) + 1 +preproc_latency
    l_pen = latency/14


    # extractor penalties
    scale = 2.0 * float(max(1, num_windows))
    lut_reg = float(result.get("LUTs", 0.0)) * scale
    dsp_reg = float(result.get("DSPs", 0.0)) * scale

    # logicnet penalty
    total_raw = float(result.get("total_raw", 1.0))
    logic_lut_pred = luts_logicnet_proxy(total_raw)

    total_luts = (logic_lut_pred + lut_reg)

    area = lut_reg + logic_lut_pred + 32* dsp_reg
    area_pen = total_luts/ (60000 + 32*200)






    # combine
    cost = (
        w_fid * fid_pen
        + w_area * area_pen
        + w_l * l_pen
    )

    info = {
        "fid_pen": fid_pen,
        "area_pen": area_pen,
        "latency_pen": l_pen,
        "total_LUTs_extractor": lut_reg,
        "total_DSPs_extractor": dsp_reg,
        "logic_lut_pred": logic_lut_pred,
        "total_latency": latency,
        "feature_bw": result.get("feature_bw"),
        "bits_in": int(result.get("bits_in", 0)),
    }

    if debug:
        print("\n[DEBUG] Cost breakdown:")
        print(f"  Fidelity={fidelity:.6f} -> fid_pen={fid_pen:.6f} (w={w_fid})")
        print(f"  LUTs extractor={lut_reg:.2f}")
        print(f"  DSPs extractor={dsp_reg:.2f} ")
        print(f"  Logic LUTs pred={logic_lut_pred:.2f}")
        print(f"  Area={area:.6f} -> area_pen={area_pen:.6f} (w_area={w_area})")
        print(f"  Latency={latency} -> latency_pen={l_pen:.6f} (w_l={w_l})")
        print(f"  Final cost={cost:.6f}")
        print(f"  Extra info: feature_bw={info['feature_bw']}, bits_in={info['bits_in']}")

    return cost, info



def apply_S_to_logic_spec(S, logic_spec, feature_bw, preproc):
    """
    Given desired S and current logic_spec, mutate logic_spec to approximate S.
    S = total_input_bits / pre_out_bits
    total_input_bits = beta_i * gamma_i * input_neurons
    pre_out_bits = (num_filter * 2) * feature_bw

    Constraints:
      - S in [2, 10]
      - n in [50, 600]
      - beta_i ∈ {1, 2, 3}
      - gamma_i ∈ [3, 7]
      - 4 < beta_i * gamma_i < 17
    """
    # Clamp S to range [2, 10]
    S = max(2, min(10, S))

    new = dict(logic_spec)
    num_filter = int(preproc.get("num_filter", 1))
    pre_out_bits = max(1, (num_filter * 2) * int(max(1, feature_bw)))
    desired_total_bits = max(1, int(round(S * pre_out_bits)))

    input_neurons = int(new["layers"][0])
    beta_i = int(new.get("beta_i", 1))
    gamma_i = int(new.get("gamma_i", 3))

    best = (beta_i, gamma_i, input_neurons)
    best_err = abs((beta_i * gamma_i * input_neurons) - desired_total_bits)

    beta_range = [1, 2, 3]
    gamma_range = list(range(3, 8))

    for b in beta_range:
        for g in gamma_range:
            if not (4 < b * g < 17):
                continue
            target_n = int(round(desired_total_bits / max(1, b * g)))
            cand_ns = [
                target_n,
                target_n - 20,
                target_n - 10,
                target_n - 5,
                target_n + 5,
                target_n + 10,
                target_n + 20,
            ]
            for n in cand_ns:
                # Clamp n to [50, 600]
                n = max(50, min(600, n))
                err = abs((b * g * n) - desired_total_bits)
                err += 0.01 * abs(n - input_neurons)  # small penalty for deviation
                if err < best_err:
                    best_err = err
                    best = (b, g, n)

    b_best, g_best, n_best = best
    new["beta_i"] = int(b_best)
    new["gamma_i"] = int(g_best)
    new_layers = list(new["layers"])
    new_layers[0] = int(n_best)
    new["layers"] = new_layers

    return new




# ----------------------
# run_experiment: extractor runs + compute resources + logicnet proxies
#   caching: only run NN if preprocessing changed.
# ----------------------
_nn_cache = {}
def _make_preproc_key(preproc):
    # preproc is dict of filt_type, num_filter, n, sig_start, sig_length, shift
    return (preproc["filt_type"], int(preproc["num_filter"]), int(preproc["n"]),
            int(preproc["sig_start"]), int(preproc["sig_length"]), int(preproc["shift"]))



from torch.utils.data import TensorDataset, DataLoader
import torch

def run_experiment(preproc, logic_spec, debug=False):
    """
    preproc: dict with keys filt_type (1=integrator,2=mf), num_filter,n,sig_start,sig_length,shift
    logic_spec: dict with keys layers (list), beta_i,gamma_i,beta,gamma,beta_o,gamma_o
    Returns result dict including fidelity, extractor LUT/DSP, logicnet proxy totals, etc.
    """



    # ------------------------------
    # Feature extraction
    # ------------------------------
    ft = preproc["filt_type"]
    start = int(preproc["sig_start"]); flen = int(preproc["sig_length"])
    n = int(preproc["n"]); shift = int(preproc["shift"]); num_filter = int(preproc["num_filter"])

    if debug:
        print(f"[DEBUG] filt_type={ft}, sig_start={start}, sig_length={flen}, n={n}, shift={shift}, num_filter={num_filter}")
    
    if debug:
        print(f"[DEBUG] BEGINNING FEATURE EXTRACTION")

    if ft == 2:
        weight_I, weight_Q = compute_weights(data_X_trunc, start, start+flen)
        Xtr, Ytr = extract_features_mf(data_X_trunc, start, flen, weight_I, weight_Q, n, shift, num_filter)
        Xte, Yte = extract_features_mf(test_X_trunc, start, flen, weight_I, weight_Q, n, shift, num_filter)
    elif ft == 1:
        Xtr, Ytr = extract_features_int(data_X_trunc, start, flen, n, shift, num_filter)
        Xte, Yte = extract_features_int(test_X_trunc, start, flen, n, shift, num_filter)
    else:
        raise ValueError("Invalid filt_type")
    
    if Xtr.size == 0 or Xte.size == 0:
        if debug:
            print("[DEBUG] Empty feature arrays, aborting.")
        return None

    # Bitwidth and feature conversion
    W_in = 14

    feature_bw = feature_bitwidth_from_preproc(W_in, flen, num_filter, shift, n, ft)
    output_bits = 2*num_filter*feature_bw

    if debug:
        print(f"[DEBUG] Computed feature bitwidth={feature_bw}, output_bits={output_bits}")

    Xtr_bits = features_to_bitarray(Xtr, bits=feature_bw)
    Xte_bits = features_to_bitarray(Xte, bits=feature_bw)

    if debug:
        print(f"[DEBUG] Xtr_bits.shape={Xtr_bits.shape}, Xte_bits.shape={Xte_bits.shape}")

    # ------------------------------
    # Build LogicNets config
    # ------------------------------
    local_logic_spec = dict(logic_spec)
    input_dim = int(Xtr.shape[1])
    local_layers = list(local_logic_spec.get("layers", []))
    if len(local_layers) == 0:
        local_layers = [300, 64, 8]
        if debug:
            print(f"[DEBUG] logic_spec['layers'] missing, using default {local_layers}")

    model_cfg = {
        "output_length": 1,
        "input_bitwidth": max(1, int(local_logic_spec.get("beta_i", feature_bw))),
        "hidden_layers": local_layers,
        "hidden_bitwidth": max(1, int(local_logic_spec.get("beta", 3))),
        "output_bitwidth": max(1, int(local_logic_spec.get("beta_o", 3))),
        "input_fanin": int(local_logic_spec.get("gamma_i", 1)),
        "hidden_fanin": int(local_logic_spec.get("gamma", 1)),
        "output_fanin": int(local_logic_spec.get("gamma_o", 1)),
    }

    if debug:
        print(f"[DEBUG] model_cfg={model_cfg}")

    # ------------------------------
    # Build datasets for train.py
    # ------------------------------
    Xtr_t = torch.from_numpy(Xtr_bits).float()
    Ytr_t = torch.from_numpy(Ytr).float()
    Xte_t = torch.from_numpy(Xte_bits).float()
    Yte_t = torch.from_numpy(Yte).float()

    dataset = {
        "train": TensorDataset(Xtr_t, Ytr_t),
        "valid": TensorDataset(Xtr_t, Ytr_t),  # reuse train as val
        "test":  TensorDataset(Xte_t, Yte_t),
    }
    x, y = dataset['train'][0]
    model_cfg['input_length'] = len(x)
    model_cfg['output_length'] = 1

    if debug:
        print(f"[DEBUG] Finalized model_cfg with input_length={model_cfg['input_length']}")

    # ------------------------------
    # Instantiate model
    # ------------------------------
    ln_model = QuantumNeqModel(model_cfg)
    if debug:
        print("[DEBUG] Model instantiated.")

    # ------------------------------
    # Train + test (silent)
    # ------------------------------
    train_cfg = {
        "epochs": 5,
        "batch_size": 512,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "seed": 304,
    }
    options_cfg = {"cuda": False, "log_dir": "./log_tmp"}

    if debug:
        print("[DEBUG] Starting training...")
    earlyStop, ln_model = train_quiet(ln_model, dataset, train_cfg, options_cfg)
    if debug:
        print("[DEBUG] Training finished. Starting test...")

    ln_model.eval()
    fidelity = test_quiet(ln_model, DataLoader(dataset["test"], batch_size=train_cfg['batch_size']), options_cfg["cuda"])
    if debug:
        print(f"[DEBUG] Test finished. Fidelity={fidelity:.4f}")

    # cache

    # ------------------------------
    # Resource estimation
    # ------------------------------
    M = max(2, int(math.ceil(preproc["sig_length"] / max(1, preproc["num_filter"])))) #number input to integrator/dp core
    W = max(1, int(max(1, 14 - int(preproc["shift"])))) #width input (bits)
    N = max(1, int(preproc["n"])) #width of weights (bits)

    if preproc["filt_type"] == 2:
        res_ext = predict_resources_dp(M, W, N)
        extractor_type = "dp"
    else:
        res_ext = predict_resources_integrator(M, W)
        extractor_type = "integrator"

    res_ext_scaled = {
    "LUTs": res_ext["LUTs"] * (2 * N),
    "DSPs": res_ext["DSPs"] * (2 * N),
    "latency_cycles": res_ext["latency_cycles"],  # <-- add this
    "latency_ns": res_ext["latency_ns"],
}

    total_raw, max_raw = raw_LUTs_from_spec(logic_spec["layers"], logic_spec["beta_i"], logic_spec["gamma_i"],
                                           logic_spec["beta"], logic_spec["gamma"], logic_spec["beta_o"], logic_spec["gamma_o"])
    logic_lut_pred = luts_logicnet_proxy(total_raw)

    result = {
        "fidelity": fidelity,
        "feature_bw": feature_bw,
        "preproc": preproc,
        "logic_spec": logic_spec,
        "extractor_type": extractor_type,
        "LUTs": res_ext_scaled["LUTs"],
        "DSPs": res_ext_scaled["DSPs"],
        "total_raw": total_raw,
        "max_raw": max_raw,
        "latency_cycles": res_ext_scaled["latency_cycles"], 
        "logic_lut_pred": logic_lut_pred,
        "bits_in": 0,
        "earlyStop": earlyStop

    }

    if debug:
        print(f"[DEBUG] Result={result}")

    return result


# ----------------------
# Neighbor generators
# ----------------------
def neighbor_preproc(preproc):
    """
    Generate a neighbor configuration by perturbing one parameter slightly.
    Local move operator for simulated annealing.
    """
    new = dict(preproc)
    choices = ["filt_type", "num_filter", "n", "sig_start", "shift"]
    choice = random.choice(choices)

    if choice == "filt_type":
        # small move = flip to the other option (binary choice)
        new["filt_type"] = 1 if new["filt_type"] == 2 else 2

    elif choice == "num_filter":
        # choose a factor of sig_length close to current num_filter
        sig_length = new["sig_length"]
        factors = [i for i in range(1, min(64, sig_length+1)) if sig_length % i == 0]
        if factors:
            # find index of current num_filter in sorted factors
            factors.sort()
            if new["num_filter"] in factors:
                idx = factors.index(new["num_filter"])
                step = random.choice([-1, 1])
                idx_new = max(0, min(len(factors)-1, idx + step))
                new["num_filter"] = factors[idx_new]
            else:
                new["num_filter"] = random.choice(factors)

    elif choice == "n":
        # local increment/decrement bounded between 0 and 14
        new["n"] = int(max(0, min(14, new["n"] + random.choice([-1, 1]))))

    elif choice == "sig_start":
        # move start by ±50 (instead of arbitrary reset)
        new["sig_start"] = int(max(0, min(100, new["sig_start"] + random.choice([-50, 50]))))
        new["sig_length"] = 500 - new["sig_start"]
        # adjust num_filter to be a divisor close to old one
        factors = [i for i in range(1, min(64, new["sig_length"]+1)) if new["sig_length"] % i == 0]
        if factors:
            factors.sort()
            if new["num_filter"] in factors:
                idx = factors.index(new["num_filter"])
                step = random.choice([-1, 1])
                idx_new = max(0, min(len(factors)-1, idx + step))
                new["num_filter"] = factors[idx_new]
            else:
                new["num_filter"] = random.choice(factors)
        else:
            new["num_filter"] = 1

    elif choice == "shift":
        # small move up/down bounded between 2 and 9
        new["shift"] = int(max(2, min(9, new["shift"] + random.choice([-1, 1]))))

    return new


def neighbor_logic(logic_spec, feature_bw, preproc):
    """
    Local neighbor operator for logic_spec.
    Mutations are small, smooth adjustments instead of big jumps.
    Constraints enforced inline:
      - 4 < beta_i*gamma_i < 17
      - 4 < beta*gamma < 17
      - 4 < beta_o*gamma_o < 20
    """
    new = dict(logic_spec)

    # Ensure layers list exists and has 3 or 4 entries (input + hidden)
    layers = list(new.get("layers", []))
    if len(layers) < 3:
        input_n = layers[0] if len(layers) > 0 else max(1, feature_bw*2)
        layers = [int(input_n), 64, 8]
    if len(layers) > 4:
        layers = layers[:4]
    target_len = random.choice([3, 4])
    while len(layers) < target_len:
        layers.append(max(2, int(layers[-1] // 2)))
    while len(layers) > target_len:
        layers = layers[:target_len]
    new["layers"] = layers

    choices = [
        "mutate_S", "mutate_hidden_neurons",
        "mutate_hidden_bw", "mutate_hidden_fanin",
        "mutate_output_bw", "mutate_output_fanin"
    ]
    choice = random.choice(choices)

    # compute current S
    try:
        current_S = compute_S_from_config(new, feature_bw)
    except Exception:
        current_S = 2.0
    valid_bg_pairs = [(b, g) for b in range(1, 4) for g in range(4, 20) if 4 < b * g < 17]
    valid_bgo_pairs = [(b, g) for b in range(1, 4) for g in range(5, 20) if 8 < b * g < 17]

    if choice == "mutate_S":
        factor = random.choice([-0.5, -0.2, -0.1 , 0.1, 0.2, 0.5])
        if current_S == 2:
            S_new = max(2, min(10, current_S + abs(factor)))
        elif current_S == 10:
            S_new = max(2, min(10, current_S - abs(factor)))
        else:
            S_new = max(2, min(10, current_S + factor))
    
        new = apply_S_to_logic_spec(S_new, new, feature_bw, preproc)

    elif choice == "mutate_hidden_neurons":
        for i in range(1, len(new["layers"])):
            if random.random() < 0.6:
                delta = random.choice([-4, -2, +2, +4])
                new["layers"][i] = max(2, new["layers"][i] + delta)

            # Enforce strictly decreasing layer sizes
            prev = new["layers"][i - 1]
            curr = new["layers"][i]
            if curr >= prev:
                new["layers"][i] = max(2, prev - random.randint(1, 4))

    elif choice == "mutate_hidden_bw":
        new["beta"] = max(1, int(new.get("beta", 3) + random.choice([-1, 1])))

    elif choice == "mutate_hidden_fanin":
        new["gamma"] = max(1, int(new.get("gamma", 3) + random.choice([-1, 1])))

    elif choice == "mutate_output_bw":
        new["beta_o"] = max(1, int(new.get("beta_o", 3) + random.choice([-1, 1])))

    elif choice == "mutate_output_fanin":
        new["gamma_o"] = max(1, int(new.get("gamma_o", 3) + random.choice([-1, 1])))


    # --- Utility: find closest valid pair to current values ---
    def clamp_to_valid_pairs(beta, gamma, valid_pairs, default):
        # Enforce lower bounds first
        beta = max(1, min(3, int(beta)))
        gamma = max(5, int(gamma))

        if (beta, gamma) in valid_pairs:
            return beta, gamma

        # Find closest valid pair by minimal |β−β'| + |γ−γ'|
        best_pair = min(valid_pairs, key=lambda p: abs(p[0] - beta) + abs(p[1] - gamma))
        return best_pair if best_pair else default


    # --- Apply constraints ---
    new["beta_i"], new["gamma_i"] = clamp_to_valid_pairs(
        new.get("beta_i", 2), new.get("gamma_i", 3), valid_bg_pairs, default=(1, 6)
    )
    new["beta"], new["gamma"] = clamp_to_valid_pairs(
        new.get("beta", 2), new.get("gamma", 5), valid_bg_pairs, default=(2, 5)
    )
    new["beta_o"], new["gamma_o"] = clamp_to_valid_pairs(
        new.get("beta_o", 2), new.get("gamma_o", 7), valid_bgo_pairs, default=(2, 7)
    )

    return new



def neighbor_combined(preproc, logic_spec, feature_bw):
    # Decide whether to mutate preproc, logic, or both
    which = random.choices(["preproc", "logic", "both"])
    new_pre = preproc
    new_logic = logic_spec
    if which == "preproc":
        new_pre = neighbor_preproc(preproc)
    elif which == "logic":
        new_logic = neighbor_logic(logic_spec, feature_bw, preproc)
    else:  # both
        new_pre = neighbor_preproc(preproc)
        # If preproc changes, feature_bw may change; we conservatively pass current feature_bw
        new_logic = neighbor_logic(logic_spec, feature_bw, new_pre)
    return new_pre, new_logic, which

#-----------------------
# YAML Template
#-----------------------
def make_yaml_template(T, move, iter_num, which, result, cost, info):
    """
    Create a structured YAML-friendly dict for logging each iteration.
    """
    return {
        "temperature": float(T),
        "iteration": int(iter_num),
        "move": int(move),
        "mutation_type": which,
        "fidelity": float(result.get("fidelity", 0.0)),
        "cost": float(cost),
        "fidelity_adj": float(info.get("fidelity_adj", 0.0)),
        "penalties": {
            "fid_pen": float(info.get("fid_pen", 0.0)),
            "area_pen": float(info.get("area_pen", 0.0)),
            "latency": float(info.get("total_latency", 0.0))
        },
        "resources": {
            "LUTs_extractor": float(info.get("total_LUTs_extractor", 0.0)),
            "DSPs_extractor": float(info.get("total_DSPs_extractor", 0.0)),
            "logic_lut_pred": float(info.get("logic_lut_pred", 0.0)),
        },
        "preprocessing": result.get("preproc", {}),
        "logic_spec": result.get("logic_spec", {}),
    }


def save_yaml_log(entry, yaml_file="annealing_log.yaml"):
    """Append a YAML entry cleanly."""
    with open(yaml_file, "a") as f:
        yaml.safe_dump([entry], f, sort_keys=False)
        f.write("\n---\n")  # YAML document separator


#-----------------------
# Hashable Configs
#-----------------------
def _make_hashable(d):
    """Recursively convert lists inside dicts to tuples for hashing."""
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = tuple(v)
        elif isinstance(v, dict):
            out[k] = _make_hashable(v)
        else:
            out[k] = v
    return out


def _hash_config(preproc, logic_spec):
    pre_hash = tuple(sorted(_make_hashable(preproc).items()))
    logic_hash = tuple(sorted(_make_hashable(logic_spec).items()))
    return (pre_hash, logic_hash)

def load_cache_from_yaml(yaml_file):
    """
    Load cached experiment results from a YAML log file written by simulated_annealing_full.
    Returns a dict: {hash: (result, cost, info)}.
    """
    cache = {}

    if not os.path.exists(yaml_file):
        print(f"[CACHE INIT] No previous YAML log found at {yaml_file}. Starting fresh.")
        return cache

    try:
        with open(yaml_file, "r") as f:
            # YAML entries are separated by '---' so safe_load_all gives a generator of dicts/lists
            docs = list(yaml.safe_load_all(f))

        # Handle both top-level lists and individual dicts per document
        total_entries = 0
        for doc in docs:
            if doc is None:
                continue

            # Each document might be a list (starting with '-') or a single dict
            entries = doc if isinstance(doc, list) else [doc]

            for entry in entries:
                try:
                    preproc = entry.get("preprocessing")
                    logic_spec = entry.get("logic_spec")
                    cost = entry.get("cost")
                    fidelity = entry.get("fidelity")

                    # Minimal fields for reconstructing cache
                    if preproc and logic_spec and cost is not None:
                        cfg_hash = _hash_config(preproc, logic_spec)

                        # Compose a minimal 'result' dict similar to run_experiment output
                        result = {
                            "fidelity": fidelity,
                            "feature_bw": preproc.get("num_filter", 1),
                            "resources": entry.get("resources", {}),
                            "penalties": entry.get("penalties", {}),
                        }

                        info = {
                            "iteration": entry.get("iteration"),
                            "move": entry.get("move"),
                            "mutation_type": entry.get("mutation_type"),
                        }

                        cache[cfg_hash] = (result, cost, info)
                        total_entries += 1

                except Exception as e:
                    print(f"[CACHE WARN] Skipping invalid entry: {e}")

        print(f"[CACHE LOAD] Loaded {len(cache)} cached configurations from {yaml_file}.")
    except Exception as e:
        print(f"[CACHE ERROR] Failed to parse {yaml_file}: {e}")

    return cache



#-----------------------
# Flatten config for CSV
#-----------------------
def _flatten_config(T, move, res, cost, info, preproc, logic_spec):
    """
    Flatten nested structures for CSV logging.
    """
    flat = {"temperature": T, "move": move}

    # --- result fields ---
    for k, v in res.items():
        if isinstance(v, (int, float, str)):
            flat[k] = v

    # --- cost + info ---
    flat["cost"] = cost
    for k, v in info.items():
        if isinstance(v, (int, float, str)):
            flat[k] = v

    # --- preproc ---
    flat["pre_filt_type"] = preproc.get("filt_type")
    flat["pre_num_filter"] = preproc.get("num_filter")
    flat["pre_n"] = preproc.get("n")
    flat["pre_sig_start"] = preproc.get("sig_start")
    flat["pre_sig_length"] = preproc.get("sig_length")
    flat["pre_shift"] = preproc.get("shift")

    # --- logic_spec ---
    layers = logic_spec.get("layers", [])
    flat["logic_layers"] = json.dumps(layers)
    flat["logic_beta_i"] = logic_spec.get("beta_i")
    flat["logic_gamma_i"] = logic_spec.get("gamma_i")
    flat["logic_beta"] = logic_spec.get("beta")
    flat["logic_gamma"] = logic_spec.get("gamma")
    flat["logic_beta_o"] = logic_spec.get("beta_o")
    flat["logic_gamma_o"] = logic_spec.get("gamma_o")

    return flat


#-----------------------
# Simulated Annealing
#-----------------------
def simulated_annealing_full(iters=60, T_start=5.0, alpha=0.94, move_per_temp=20,
                             init_preproc=None, init_logic=None,
                             weights=(1.0, 1.0, 1.0), k=1.0,
                             output_csv="annealing_log.csv",
                             yaml_file="annealing_log.yaml",
                             old_log=None,
                             debug=False, T_freeze=0.01, resume=False, resume_path=None):
    """
    Simulated annealing with caching of previously seen configurations.
    Supports resuming from a previous YAML log.
    """
    # --- load cache from old log if provided ---
    cache = load_cache_from_yaml(old_log) if old_log else {}

    # --- initialize if not resuming ---
    if not resume:
        if init_preproc is None:
            sig_start = 100
            sig_length = 400
            init_preproc = {
                "filt_type": 1, "num_filter": 2, "n": 1,
                "sig_start": sig_start, "sig_length": sig_length, "shift": 7
            }

        if init_logic is None:
            init_logic = {
                "layers": [145, 40, 15],
                "beta_i": 1, "gamma_i": 6,
                "beta": 2, "gamma": 6,
                "beta_o": 2, "gamma_o": 8
            }

        preproc = init_preproc
        logic_spec = init_logic

        print("[INFO] Starting initial run...")
        initial_res = run_experiment(preproc, logic_spec, debug)
        if initial_res is None:
            raise RuntimeError("Initial run_experiment failed.")

        initial_cost, initial_info = compute_cost(
            initial_res, num_windows=preproc["num_filter"], weights=weights, debug=debug
        )

        current = {
            "preproc": preproc,
            "logic_spec": logic_spec,
            "result": initial_res,
            "cost": initial_cost,
            "info": initial_info,
        }

        # cache first result
        cfg_hash = _hash_config(preproc, logic_spec)
        cache[cfg_hash] = (initial_res, initial_cost, initial_info)

        # log first result
        row0 = _flatten_config(T_start, 0, initial_res, initial_cost, initial_info, preproc, logic_spec)
        pd.DataFrame([row0]).to_csv(
            output_csv, mode="a", header=not os.path.exists(output_csv), index=False
        )

        yaml_entry = make_yaml_template(T_start, 0, 0, "init", initial_res, initial_cost, initial_info)
        save_yaml_log(yaml_entry, yaml_file)
        print(f"[INIT] cost={initial_cost:.4f}, fid={initial_res['fidelity']:.4f}")

        # --- main annealing loop setup ---
        T = float(T_start)
        iteration = 0
        accepted = []

    else:
        # --- resume from previous log ---
        preproc, logic_spec, iter_start, move_start, T, accepted = load_last_state(resume_path)
        print(f"[RESUME] From iter={iter_start}, move={move_start}, T={T:.3f}")

        # restore last state from cache if possible
        cfg_hash = _hash_config(preproc, logic_spec)
        if cfg_hash in cache:
            cand_res, cand_cost, cand_info = cache[cfg_hash]
        else:
            cand_res = run_experiment(preproc, logic_spec, debug)
            cand_cost, cand_info = compute_cost(
                cand_res, num_windows=preproc["num_filter"], weights=weights, debug=debug
            )
            cache[cfg_hash] = (cand_res, cand_cost, cand_info)

        current = {
            "preproc": preproc,
            "logic_spec": logic_spec,
            "result": cand_res,
            "cost": cand_cost,
            "info": cand_info,
        }
        iteration = iter_start
        T *= alpha

    # --- main simulated annealing loop ---
    while T > T_freeze and iteration < iters:
        iteration += 1
        for move in range(move_per_temp):
            curr_bw = current["result"].get("feature_bw", 1)
            cand_pre, cand_logic, which = neighbor_combined(preproc, logic_spec, curr_bw)
            cand_hash = _hash_config(cand_pre, cand_logic)

            if cand_hash in cache:
                cand_res, cand_cost, cand_info = cache[cand_hash]
                print(f"[CACHE HIT] iteration={iteration} using cached cost={cand_cost:.4f}")
            else:
                cand_res = run_experiment(cand_pre, cand_logic, debug)
                if cand_res is None:
                    print(f"[SKIP] NN run failed at iter={iteration}")
                    continue
                cand_cost, cand_info = compute_cost(
                    cand_res, num_windows=cand_pre["num_filter"], weights=weights, debug=debug
                )
                cache[cand_hash] = (cand_res, cand_cost, cand_info)

            # acceptance criterion
            dE = cand_cost - current["cost"]
            accept = (dE < 0) or (random.random() < math.exp(-dE / (max(1e-12, T) * k)))

            if accept:
                print(f"[ACCEPT] iter={iteration} T={T:.3f} move={move} which={which} cost={cand_cost:.4f}")
                preproc, logic_spec = cand_pre, cand_logic
                current = {
                    "preproc": preproc,
                    "logic_spec": logic_spec,
                    "result": cand_res,
                    "cost": cand_cost,
                    "info": cand_info,
                }

                row = _flatten_config(T, move, cand_res, cand_cost, cand_info, cand_pre, cand_logic)
                pd.DataFrame([row]).to_csv(output_csv, mode="a", header=False, index=False)

                yaml_entry = make_yaml_template(T, move, iteration, which, cand_res, cand_cost, cand_info)
                save_yaml_log(yaml_entry, yaml_file)
                accepted.append(row)
            else:
                print(f"[REJECT] iter={iteration} T={T:.3f} move={move} dE={dE:.4f}")

        T *= alpha  # cool down

    print(f"[DONE] Total accepted: {len(accepted)}")
    return accepted




def load_last_state(yaml_path):
    """
    Robust loader for multi-document YAML logs created by save_yaml_log().
    Returns:
        preproc, logic_spec, last_iter, last_move, last_temp, accepted_list
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"No YAML log found at {yaml_path}")

    with open(yaml_path, "r") as f:
        docs = list(yaml.safe_load_all(f))

    # Flatten docs: each doc can be None, a dict, or a list of dicts (because we wrote [entry])
    entries = []
    for doc in docs:
        if doc is None:
            continue
        if isinstance(doc, list):
            for it in doc:
                if isinstance(it, dict):
                    entries.append(it)
        elif isinstance(doc, dict):
            entries.append(doc)
        else:
            # skip unexpected doc types
            continue

    if not entries:
        raise ValueError(f"YAML file {yaml_path} contains no valid entries")

    # Last accepted entry (last real dict)
    last = entries[-1]

    # Keys used by make_yaml_template():
    # "temperature", "iteration", "move", "mutation_type", "fidelity", "cost",
    # "preprocessing" (preproc), "logic_spec", "penalties", "resources"
    temperature = last.get("temperature")
    iteration = last.get("iteration", 0)
    move = last.get("move", 0)
    preproc = last.get("preprocessing", last.get("preproc", None))
    logic_spec = last.get("logic_spec", None)

    # Print short resume summary
    print("=== Resuming from last accepted configuration ===")
    print(f"Entries found       : {len(entries)}")
    print(f"Iteration (last)    : {iteration}, Move: {move}, Temperature: {temperature}")
    print(f"Preproc (summary)   : {preproc}")
    print(f"Logic spec (summary): {logic_spec}")
    print("===============================================")

    # Build accepted[] history
    accepted_list = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        # Some older entries may not include all keys, be forgiving
        fid = e.get("fidelity", e.get("fid", None))
        acc = {
            "temperature": e.get("temperature"),
            "iteration": e.get("iteration"),
            "move": e.get("move"),
            "which": e.get("mutation_type", e.get("mutation", None)),
            "cost": e.get("cost"),
            "fidelity": fid,
            "preproc": e.get("preprocessing", e.get("preproc")),
            "logic_spec": e.get("logic_spec"),
        }
        accepted_list.append(acc)

    return preproc, logic_spec, iteration, move, temperature, accepted_list





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulated annealing with parameters.")

    parser.add_argument("--iters", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--T_start", type=float, default=1000.0, help="Initial temperature")
    parser.add_argument("--T_freeze", type=float, default=2.0, help="End temperature")
    parser.add_argument("--move_per_temp", type=int, default=10, help="Moves per temp")
    parser.add_argument("--alpha", type=float, default=0.94, help="Cooling rate")
    parser.add_argument("--k", type=float, default=0.949916, help="boltz")
    parser.add_argument(
        "--weights", type=float, nargs=3, default=[0.2, 0.6, 0.2],
        help="Tuple of 5 weights (space-separated): fid weight, extractor LUT weight, extractor DSP weight, logicnet LUT weight, latency weight"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--resume", action="store_true", help="Enable resume last state")
    parser.add_argument("--resume_path", type=str, default=None, help="Enable resumption")
    parser.add_argument("--yaml_file", type=str, default="yaml_log.yaml", help="Optional YAML config file")
    parser.add_argument("--output_csv", type=str, default="results.csv", help="Output CSV filename")
    parser.add_argument("--old_log", type=str, default="results.yaml", help="Old YAML Log")


    args = parser.parse_args()
    iters = args.iters
    k = args.k
    T_start = args.T_start
    alpha = args.alpha
    weights = tuple(args.weights)
    debug = args.debug
    output_csv = args.output_csv
    old_log = args.old_log
    yaml_file = args.yaml_file
    T_freeze = args.T_freeze
    move_per_temp = args.move_per_temp
    resume = args.resume
    resume_path = args.resume_path
    



    results = simulated_annealing_full(iters=iters, T_start=T_start, T_freeze=T_freeze, alpha=alpha, weights=weights, debug=debug, yaml_file=yaml_file, output_csv=output_csv, move_per_temp=move_per_temp, old_log=old_log, k=k, resume=resume, resume_path=resume_path)
    print("Done. saved to", output_csv)

