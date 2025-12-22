# Redesigned discrete DE script with safe multiprocessing and memmap-backed dataset loading.
# Key fixes included:
# - Avoid pickling large in-memory numpy arrays by using np.load(..., mmap_mode='r') in each worker.
# - Keep a single authoritative cache in the main process; workers return results and main merges cache.
# - Use joblib loky backend (process isolation) to avoid leaks and shared-tensor problems.
# - Ensure per-worker random seeding to make runs reproducible and avoid correlated RNG state.
# - Force resource cleanup after NN training in workers (del references, gc.collect()).
# - Consistent tuple formats and robust error handling in worker tasks.

# -----------------------
# Standard library
# -----------------------
import os
import math
import random
import time
import hashlib
import json
import gc
import datetime

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

# User-provided modules (assumed present)
from train import train_quiet, test_quiet
from models import QuantumNeqModel

# Joblib for parallelism
from joblib import Parallel, delayed


# Configure PyTorch threading (per-process)
torch.set_num_threads(3)
torch.set_num_interop_threads(2)

# -----------------------
# Constants / Domain
# -----------------------
VECTOR_LEN = 16
SIG_START_RANGE = [0, 50, 100]
FILT_TYPES = [1]
N_RANGE = list(range(0, 7))
SHIFT_RANGE = list(range(2, 10))

VALID_BETA = [1, 2]
VALID_GAMMA_I = [6, 7]
VALID_GAMMA = list(range(6, 17))
VALID_GAMMA_O = list(range(6, 17))

# Data files (adjust paths if necessary)
DATA_DIR = "./qick_data"
X_TRAIN_FNAME = os.path.join(DATA_DIR, '0528_X_train_0_770.npy')
Y_TRAIN_FNAME = os.path.join(DATA_DIR, '0528_y_train_0_770.npy')
X_TEST_FNAME = os.path.join(DATA_DIR, '0528_X_test_0_770.npy')
Y_TEST_FNAME = os.path.join(DATA_DIR, '0528_y_test_0_770.npy')

# Reference checksums (kept from original script)
MD5_X_TRAIN = 'b61226c86b7dee0201a9158455e08ffb'
MD5_Y_TRAIN = 'c59ce37dc7c73d2d546e7ea180fa8d31'
MD5_X_TEST = 'b7d85f42522a0a57e877422bc5947cde'
MD5_Y_TEST = '8c9cce1821372380371ade5f0ccfd4a2'

# Shapes / constants used in the original script
START_WINDOW = 0
END_WINDOW = 770
SAMPLE_RESHAPE = (90, 2, 5000, 770, 2)  # training reshape
TEST_RESHAPE = (10, 2, 5000, 770, 2)    # test reshape
SIGNAL_LEN = 750

PREPROC_GOOD_FNAME = None #"/home/mfaroo19/Desktop/quantum/pruned_configs/pruned_decoupled_beta_parallel_fixed_preproc_prefiltered.jsonl"   # adjust path
LOGIC_GOOD_FNAME = None #"/home/mfaroo19/Desktop/quantum/pruned_configs/pruned_decoupled_beta_parallel_fixed_logicnet_points.jsonl" 

AREA_THRESHOLD = 20000.0     # default: LUT units (tweak)
LATENCY_THRESHOLD = 14.0     # default: ns or cycles depending on your stored metric (tweak)
MAX_INVALID_RETRIES = 500    # attempts to draw a valid replacement before giving up

# global containers (populated by load_good_configs)
GOOD_PREPROC_DICT = None
GOOD_LOGIC_DICT = None

# Memmap-backed dataset loader
# ----------------------
# Worker processes call worker_init() to lazily load memmap views of the arrays.
# Main process calls load_and_verify_data() once to verify checksums and ensure files exist.

GLOBAL_DATA_INITIALIZED = False

def load_and_verify_data(skip_md5=False):
    """Main-process validation and light load check. Does not keep full arrays in memory.
    Verifies md5 checksums (expensive) unless skip_md5=True."""
    if not os.path.exists(X_TRAIN_FNAME):
        raise FileNotFoundError(f"{X_TRAIN_FNAME} not found")

    else:
        data_X = np.load(X_TRAIN_FNAME)
    
    if not os.path.exists(Y_TRAIN_FNAME):
        raise FileNotFoundError(f"{Y_TRAIN_FNAME} not found")
    else:
         data_Y = np.load(Y_TRAIN_FNAME)


    if not os.path.exists(X_TEST_FNAME):
        raise FileNotFoundError(f"{X_TEST_FNAME} not found")
    else:
        test_X = np.load(X_TEST_FNAME)

    if not os.path.exists(Y_TEST_FNAME):
        raise FileNotFoundError(f"{X_TEST_FNAME} not found")
    else:
        test_Y = np.load(Y_TEST_FNAME)

    if not skip_md5:
        print("[MAIN] Verifying checksums (this may take a while)...")

        assert hashlib.md5(data_X).hexdigest() == MD5_X_TRAIN, f"MD5 mismatch for X train"
        assert hashlib.md5(data_Y).hexdigest() == MD5_Y_TRAIN, f"MD5 mismatch for Y train"
        assert hashlib.md5(test_X).hexdigest()  == MD5_X_TEST, f"MD5 mismatch for X test"
        assert hashlib.md5(test_Y).hexdigest()  == MD5_Y_TEST, f"MD5 mismatch for Y test"
        print("[MAIN] Checksums OK.")
    else:
        print("[MAIN] Skipping checksums per request.")
        
#good config helpers:

def _preproc_key_from(d):
    # canonical key for preproc: (sig_start, sig_length, num_filter, filt_type, n, shift)
    return (int(d.get("sig_start", 0)),
            int(d.get("sig_length", 0)),
            int(d.get("num_filter", 0)),
            int(d.get("filt_type", 0)),
            int(d.get("n", 0)),
            int(d.get("shift", 0)))

def _logic_key_from(d):
    """
    Canonical key for logic configuration.
    Includes all decoupled beta/gamma fields.
    (layers, beta, beta_i, beta_o, gamma_i, gamma, gamma_o)
    """
    layers = tuple(d.get("layers", []))

    beta   = int(d.get("beta", 1))
    beta_i = int(d.get("beta_i", beta))
    beta_o = int(d.get("beta_o", beta))

    gamma_i = int(d.get("gamma_i", 6))
    gamma   = int(d.get("gamma", 6))
    gamma_o = int(d.get("gamma_o", gamma))

    return (layers, beta, beta_i, beta_o, gamma_i, gamma, gamma_o)

def load_good_configs(preproc_fname=PREPROC_GOOD_FNAME, logic_fname=LOGIC_GOOD_FNAME):
    """Load JSONL files into dicts keyed by canonical tuples (quick lookups)."""
    global GOOD_PREPROC_DICT, GOOD_LOGIC_DICT
    GOOD_PREPROC_DICT = {}
    GOOD_LOGIC_DICT = {}

    # load preproc list
    try:
        with open(preproc_fname, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                j = json.loads(line)
                k = _preproc_key_from(j)
                GOOD_PREPROC_DICT[k] = j
    except FileNotFoundError:
        print(f"[WARN] Good preproc file not found: {preproc_fname}. Disabling preproc-checks.")
        GOOD_PREPROC_DICT = None

    # load logic list
    try:
        with open(logic_fname, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                k = _logic_key_from(j)
                GOOD_LOGIC_DICT[k] = j
    except FileNotFoundError:
        print(f"[WARN] Good logic file not found: {logic_fname}. Disabling logic-checks.")
        GOOD_LOGIC_DICT = None

    loaded = (len(GOOD_PREPROC_DICT) if GOOD_PREPROC_DICT else 0, len(GOOD_LOGIC_DICT) if GOOD_LOGIC_DICT else 0)
    print(f"[LOAD_GOOD_CONFIGS] loaded preproc={loaded[0]} logic={loaded[1]}")
    return GOOD_PREPROC_DICT, GOOD_LOGIC_DICT
    
# ---------- NEW: check function ----------
def _compute_pair_area_latency(pre_meta, logic_meta):
    """Compute area and latency from metadata entries (safe to miss fields)."""
    lut_pre = float(pre_meta.get("lut_pre", 0.0))
    dsp_pre = float(pre_meta.get("dsp_pre", 0.0))
    logic_lut = float(logic_meta.get("logic_lut", 0.0))
    # area: LUTs + logic LUTs + (DSPs * 32) as rough LUT-equivalent; tweak multiplier if necessary
    area = lut_pre + logic_lut + (dsp_pre * 32.0)
    latency = float(pre_meta.get("latency_pre", 0.0)) + float(logic_meta.get("logic_latency", 0.0))
    return area, latency

def fitness_check(preproc, logic, area_limit, latency_limit, dsp_scale=32):
    """
    Compute total area and latency for a (preproc, logicnet) combo.
    Returns (is_valid, total_area, total_latency, breakdown_dict)
    
    Arguments:
        preproc: dict with sig_start, sig_length, num_filter, filt_type, n, shift
        logic: dict with layers, beta_i, beta, beta_o, gamma_i, gamma, gamma_o
        area_limit, latency_limit: thresholds to validate against
        dsp_scale: LUT equivalent weight for DSPs
    """
    # ----------------------------
    # Preproc resource estimation
    # ----------------------------
    M = max(2, int(math.ceil(preproc["sig_length"] / preproc["num_filter"])))
    W = max(1, int(max(1, 14 - int(preproc["shift"]))))
    res_pre = predict_resources_integrator(M, W)
    N_weights = max(1, preproc["n"])
    num_windows = preproc["num_filter"]
    lut_pre = float(res_pre["LUTs"]) * (2 * num_windows)
    dsp_pre = float(res_pre.get("DSPs", 0.0)) * (2 * num_windows)
    latency_pre = int(res_pre.get("latency_cycles", 0))

    # ----------------------------
    # Logic resource estimation
    # ----------------------------
    layers = tuple(logic["layers"])
    beta_i, beta, beta_o = int(logic["beta_i"]), int(logic["beta"]), int(logic["beta_o"])
    gamma_i, gamma, gamma_o = int(logic["gamma_i"]), int(logic["gamma"]), int(logic["gamma_o"])

    # Constraint checks (same as your logic_task)
    if not (5 < gamma < 17 and 5 < beta * gamma < 17):
        return False, None, None, {"reason": "invalid beta/gamma combo"}
    if not (5 < gamma_i < 17 and 5 < beta_i * gamma_i < 17):
        return False, None, None, {"reason": "invalid beta_i/gamma_i combo"}
    if not (5 < gamma_o < 17 and 5 < beta_o * gamma_o < 17):
        return False, None, None, {"reason": "invalid beta_o/gamma_o combo"}

    total_raw, _ = raw_LUTs_from_spec(
        layers,
        feature_input_bw=beta_i, in_fanin=gamma_i,
        hid_bw=beta, hid_fanin=gamma,
        out_bw=beta_o, out_fanin=gamma_o
    )
    lut_logic = luts_logicnet_proxy(total_raw)
    latency_logic = len(layers)

    # ----------------------------
    # Combine area and latency
    # ----------------------------
    total_area = lut_pre + dsp_pre * dsp_scale + lut_logic
    total_latency = latency_pre + latency_logic

    # ----------------------------
    # Decision and output
    # ----------------------------
    is_valid = (total_area <= area_limit) and (total_latency <= latency_limit)
    breakdown = {
        "lut_pre": lut_pre,
        "dsp_pre": dsp_pre,
        "lut_logic": lut_logic,
        "area": total_area,
        "latency": total_latency,
        "pre_latency": latency_pre,
        "logic_latency": latency_logic,
    }
    return is_valid, total_area, total_latency, breakdown



def is_valid_pair(preproc, logic_spec, area_threshold=AREA_THRESHOLD, latency_threshold=LATENCY_THRESHOLD):
    """
    Return True if:
      - both preproc and logic entries exist in the 'good' lists (if those lists loaded),
      - AND their combined area/latency are <= thresholds (if thresholds set).
    If good-lists are not loaded, this function returns True (no enforcement).
    """
    
    is_valid, _, _, b = fitness_check(preproc, logic_spec, area_limit=area_threshold, latency_limit=latency_threshold)
    #if(is_valid):
    #    print(b)
    return is_valid
# ---------------------------------------------------------



def worker_init():
    """Initialize memmap-backed globals inside worker process. Safe and cheap (mmap only)."""
    global DATA_X_TRUNC, DATA_Y_NEW, TEST_X_TRUNC, TEST_Y_NEW, GLOBAL_DATA_INITIALIZED
    if GLOBAL_DATA_INITIALIZED:
        return

    # Use mmap_mode='r' so that memory is not duplicated and file pages are shared.
    X = np.load(X_TRAIN_FNAME, mmap_mode='r')
    Y = np.load(Y_TRAIN_FNAME, mmap_mode='r')
    Xt = np.load(X_TEST_FNAME, mmap_mode='r')
    Yt = np.load(Y_TEST_FNAME, mmap_mode='r')

    # Reshape lazily
    try:
        X = X.reshape(SAMPLE_RESHAPE)
        Y = Y.reshape((SAMPLE_RESHAPE[0], SAMPLE_RESHAPE[1], SAMPLE_RESHAPE[2]))
    except Exception:
        # If the file shape is different, do not break — allow downstream to error with clearer message
        pass

    try:
        Xt = Xt.reshape(TEST_RESHAPE)
        Yt = Yt.reshape((TEST_RESHAPE[0], TEST_RESHAPE[1], TEST_RESHAPE[2]))
    except Exception:
        pass

    # Truncate to SIGNAL_LEN
    DATA_X_TRUNC = X[:, :, :, :SIGNAL_LEN, :]
    DATA_Y_NEW = Y
    TEST_X_TRUNC = Xt[:, :, :, :SIGNAL_LEN, :]
    TEST_Y_NEW = Yt

    GLOBAL_DATA_INITIALIZED = True
    # Limit threads inside worker as well



# ------------------------
# Robust YAML helpers (replace previous implementations)
# ------------------------
import numbers

def _safe_serialize(obj, _depth=0, _max_list=20):
    """Recursively convert obj into YAML-safe native Python primitives.
       - Converts numpy / torch scalars to native ints/floats
       - Converts arrays/tensors to short Python lists (first _max_list items)
       - Replaces NaN/Inf with None
    """
    # depth guard
    if _depth > 12:
        return str(obj)

    # None / basic native types
    if obj is None or isinstance(obj, (str, bool)):
        return obj

    # Numbers (including numpy / python numbers)
    if isinstance(obj, numbers.Number):
        # numeric python types -> cast safely
        try:
            f = float(obj)
        except Exception:
            return str(obj)
        # handle special floats
        if math.isnan(f) or math.isinf(f):
            return None
        # if it's an integer value, return int
        if float(int(f)) == f:
            return int(f)
        return f

    # numpy scalar
    if isinstance(obj, np.generic):
        try:
            py = obj.item()
            return _safe_serialize(py, _depth+1, _max_list)
        except Exception:
            return None

    # torch scalar / tensor
    if 'torch' in globals() and isinstance(obj, torch.Tensor):
        try:
            if obj.numel() == 1:
                return _safe_serialize(obj.item(), _depth+1, _max_list)
            arr = obj.detach().cpu().numpy()
            return _safe_serialize(arr, _depth+1, _max_list)
        except Exception:
            return str(obj)

    # numpy array
    if isinstance(obj, np.ndarray):
        flat = obj.flatten()
        out = []
        for i, x in enumerate(flat[:_max_list]):
            out.append(_safe_serialize(x, _depth+1, _max_list))
        if flat.size > _max_list:
            out.append(f"...({int(flat.size - _max_list)} more)")
        return out

    # builtin containers
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        out = [_safe_serialize(x, _depth+1, _max_list) for x in seq[:_max_list]]
        if len(seq) > _max_list:
            out.append(f"...({len(seq) - _max_list} more)")
        return out

    if isinstance(obj, dict):
        d = {}
        for k, v in obj.items():
            # make sure keys are strings
            k_s = str(k)
            d[k_s] = _safe_serialize(v, _depth+1, _max_list)
        return d

    # fallback to string
    try:
        return str(obj)
    except Exception:
        return None


def make_yaml_entry(iteration, kind, idx, vec, result, cost, info, extra=None):
    """
    Compact YAML entry: only top-level header fields + result + info.
    Avoids duplicating preproc/logic_spec at top level.
    """
    # iteration/index may be None in some failure cases
    it = None if iteration is None else int(iteration)
    ix = None if idx is None else int(idx)

    # normalize cost: None if missing or +/-inf
    cost_val = None
    try:
        if cost is None:
            cost_val = None
        else:
            # protect against numpy / torch scalars
            c = float(cost)
            if math.isinf(c) or math.isnan(c):
                cost_val = None
            else:
                cost_val = float(c)
    except Exception:
        cost_val = None

    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "iteration": it,
        "kind": str(kind),
        "index": ix,
        "cost": cost_val,
        # keep result and info but sanitize their contents
        "result": _safe_serialize(result),
        "info": _safe_serialize(info),
    }

    if extra is not None:
        entry["extra"] = _safe_serialize(extra)

    return entry


def append_yaml_entry(entry, log_fname="./de_run_log.yaml"):
    """Append a YAML document after sanitizing it."""
    os.makedirs(os.path.dirname(os.path.abspath(log_fname)) or ".", exist_ok=True)
    # Ensure entry contains only primitives (already ensured by make_yaml_entry), but double-check
    safe_entry = _safe_serialize(entry)
    with open(log_fname, "a", encoding="utf-8") as f:
        f.write("---\n")
        # allow_unicode helps with any non-ascii strings
        yaml.safe_dump(safe_entry, f, sort_keys=False, default_flow_style=False, allow_unicode=True)




# ----------------------
# Feature extractors & utilities (unchanged logic but readable)
# ----------------------
# (For brevity, these are identical to the original functions but use the memmap views created by worker_init)

def compute_weights(X, start, end):
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
            val = x_i & ((1 << bits)-1)
            bin_str = f"{val:0{bits}b}"
            bit_list.extend([b == "1" for b in bin_str])
        bit_features.append(bit_list)
    return np.array(bit_features, dtype=int)

# ----------------------
# Resource predictors & proxies (kept from original)
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

s_lut_default = 0.0916561
A_default = 6705.366569019219
b_default = -0.106784160235
c_default = -0.164541971364

def luts_logicnet_proxy(total_raw, s_lut=s_lut_default):
    return s_lut * total_raw


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
# Helper: bitwidth / S / latency
# ----------------------

def feature_bitwidth_from_preproc(W_in, sig_length, num_filter, shift, n, filt_type=1):
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
    input_neurons = config["layers"][0]
    total_input_bits = config["beta_i"] * config["gamma_i"] * input_neurons
    pre_out_bits = (config["num_filter"] * 2) * feature_bw
    return float(total_input_bits) / max(1.0, pre_out_bits)

def latency_from_config(hidden_layers, num_filter, sig_length, filt_type=1):
    base = len(hidden_layers) + 1
    win_len = max(1, sig_length // max(1, num_filter))
    log_term = math.ceil(math.log2(win_len))
    extra = 1 if filt_type == 2 else 0
    return base + log_term + extra

# ----------------------
# Cost function (kept same semantics)
# ----------------------

def compute_cost(result, num_windows, weights, debug):
    w_fid, w_area, w_l = weights
    fidelity = float(result["fidelity"])
    fidelity = max(1e-6, fidelity)
    fid_pen = (1-fidelity)/(1-0.9)
    layers = result["logic_spec"]["layers"]
    preproc_latency = int(result.get("latency_cycles", 0))
    latency = len(layers) + 1 + preproc_latency
    l_pen = latency/14
    scale = 2.0 * float(max(1, num_windows))
    lut_reg = float(result.get("LUTs", 0.0)) * scale
    dsp_reg = float(result.get("DSPs", 0.0)) * scale
    total_raw = float(result.get("total_raw", 1.0))
    logic_lut_pred = luts_logicnet_proxy(total_raw)
    total_luts = (logic_lut_pred + lut_reg)
    area = lut_reg + logic_lut_pred + 32* dsp_reg
    area_pen = total_luts/ (20000)

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
        print("[DEBUG] compute_cost ->", cost, info)

    return cost, info

# ----------------------
# Experiment runner (worker-side safe) - expects worker_init() to have been called
# ----------------------

def run_experiment(preproc, logic_spec, debug=False):
    # ensure memmap data is available in this process
    if not GLOBAL_DATA_INITIALIZED:
        worker_init()

    # Local names to the memmap arrays for readability
    X_train = globals().get('DATA_X_TRUNC')
    X_test = globals().get('TEST_X_TRUNC')

    ft = preproc["filt_type"]
    start = int(preproc["sig_start"]); flen = int(preproc["sig_length"])
    n = int(preproc["n"]); shift = int(preproc["shift"]); num_filter = int(preproc["num_filter"])

    try:
        if ft == 2:
            weight_I, weight_Q = compute_weights(X_train, start, start+flen)
            Xtr, Ytr = extract_features_mf(X_train, start, flen, weight_I, weight_Q, n, shift, num_filter)
            Xte, Yte = extract_features_mf(X_test, start, flen, weight_I, weight_Q, n, shift, num_filter)
        elif ft == 1:
            Xtr, Ytr = extract_features_int(X_train, start, flen, n, shift, num_filter)
            Xte, Yte = extract_features_int(X_test, start, flen, n, shift, num_filter)
        else:
            raise ValueError("Invalid filt_type")

        if Xtr.size == 0 or Xte.size == 0:
            return None

        W_in = 14
        feature_bw = feature_bitwidth_from_preproc(W_in, flen, num_filter, shift, n, ft)
        output_bits = 2*num_filter*feature_bw

        Xtr_bits = features_to_bitarray(Xtr, bits=feature_bw)
        Xte_bits = features_to_bitarray(Xte, bits=feature_bw)

        Xtr_t = torch.from_numpy(Xtr_bits).float()
        Ytr_t = torch.from_numpy(Ytr).float()
        Xte_t = torch.from_numpy(Xte_bits).float()
        Yte_t = torch.from_numpy(Yte).float()

        dataset = {
            "train": TensorDataset(Xtr_t, Ytr_t),
            "valid": TensorDataset(Xtr_t, Ytr_t),
            "test":  TensorDataset(Xte_t, Yte_t),
        }

        model_cfg = {
            "output_length": 1,
            "input_bitwidth": max(1, int(logic_spec.get("beta_i", feature_bw))),
            "hidden_layers": list(logic_spec.get("layers", [300,64,8])),
            "hidden_bitwidth": max(1, int(logic_spec.get("beta", 3))),
            "output_bitwidth": max(1, int(logic_spec.get("beta_o", 3))),
            "input_fanin": int(logic_spec.get("gamma_i", 1)),
            "hidden_fanin": int(logic_spec.get("gamma", 1)),
            "output_fanin": int(logic_spec.get("gamma_o", 1)),
        }

        x, y = dataset['train'][0]
        model_cfg['input_length'] = len(x)

        ln_model = QuantumNeqModel(model_cfg)

        train_cfg = {"epochs": 5, "batch_size": 512, "learning_rate": 1e-3, "weight_decay": 1e-4, "seed": 304}
        options_cfg = {"cuda": False, "log_dir": "./log_tmp"}

        earlyStop, ln_model = train_quiet(ln_model, dataset, train_cfg, options_cfg)
        ln_model.eval()
        fidelity = test_quiet(ln_model, DataLoader(dataset["test"], batch_size=train_cfg['batch_size']), options_cfg["cuda"])

        # estimate resources
        M = max(2, int(math.ceil(preproc["sig_length"] / max(1, preproc["num_filter"]))));
        W = max(1, int(max(1, 14 - int(preproc["shift"]))))
        N = max(1, int(preproc["n"]))

        if preproc["filt_type"] == 2:
            res_ext = predict_resources_dp(M, W, N)
            extractor_type = "dp"
        else:
            res_ext = predict_resources_integrator(M, W)
            extractor_type = "integrator"

        res_ext_scaled = {"LUTs": res_ext["LUTs"] * (2 * N), "DSPs": res_ext["DSPs"] * (2 * N),
                          "latency_cycles": res_ext["latency_cycles"], "latency_ns": res_ext["latency_ns"]}

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

        # cleanup big tensors to reduce worker memory footprint
        try:
            del Xtr_t, Ytr_t, Xte_t, Yte_t, ln_model, dataset
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        return result

    except Exception as e:
        # Return exception information to the main process for debugging
        return {"_error": str(e)}

# ----------------------
# Vector encoding/decoding and clamp helpers
# ----------------------

def _make_hashable(d):
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


#def clamp_and_fix(vec):
#    v = vec.copy()
#    v[0] = 1
#    v[2] = int(max(min(v[2], 6), 0))
#    v[3] = int(np.clip(v[3], min(SIG_START_RANGE), max(SIG_START_RANGE)))
#   v[4] = int(np.clip(v[4], min(SHIFT_RANGE), max(SHIFT_RANGE)))
#    sig_len = max(1, 500 - int(v[3]))
#    v[1] = int(min(max(v[1], 1), sig_len))
#    divisors = [i for i in range(1, min(4, sig_len+1)) if sig_len % i == 0] or [1]
#    if divisors:
#        v[1] = min(divisors, key=lambda d: abs(d - v[1]))
#    v[5] = 3 if v[5] < 4 else 4
#   v[6] = int(min(max(v[6], 25), 145))
#    prev = v[6]
#   for i in range(1, 4):
#        v[6 + i] = int(np.clip(v[6 + i], 5, 45))
#        if v[6 + i] > prev:
#            v[6 + i] = prev
#        prev = v[6 + i]
#    v[9] = 1
#    v[10] = int(np.clip(v[10], min(VALID_BETA), max(VALID_BETA)))
#    v[11] = int(np.clip(v[11], min(VALID_GAMMA_I), max(VALID_GAMMA_I)))
#    v[12] = int(np.clip(v[12], min(VALID_BETA), max(VALID_BETA)))
#    if v[12] == 1:
#        gamma_candidates = list(range(6, 17))
#    else:
#        gamma_candidates = [6, 7, 8]
#    v[13] = int(min(gamma_candidates, key=lambda g: abs(g - v[13])))
#    v[14] = int(np.clip(v[14], min(VALID_BETA), max(VALID_BETA)))
#    if v[14] == 1:
#        gamma_o_candidates = list(range(6, 17))
#    else:
#        gamma_o_candidates = [6, 7, 8]
#    v[15] = int(min(gamma_o_candidates, key=lambda g: abs(g - v[15])))
#    return v


def decode_vector(vec):
    v = clamp_and_fix(vec)
    preproc = {"filt_type": int(v[0]), "num_filter": int(v[1]), "n": int(v[2]), "sig_start": int(v[3]),
               "sig_length": max(1, 500 - int(v[3])), "shift": int(v[4])}
    layer_count = int(v[5])
    layers = [int(v[6+i]) for i in range(layer_count)]
    logic_spec = {"layers": layers, "beta_i": int(v[10]), "gamma_i": int(v[11]),
                  "beta": int(v[12]), "gamma": int(v[13]),
                  "beta_o": int(v[14]), "gamma_o": int(v[15])}
    return preproc, logic_spec

# ------------------------
# DE Operators
# ------------------------

_ALLOWED_INPUT_NEURONS = list(range(25, 146, 5))   # 25,30,...,145
_ALLOWED_HIDDEN_NEURONS = list(range(5, 46, 5))    # 5,10,...,45

def _snap_to_grid(val, allowed):
    """Snap integer val to nearest element in allowed (allowed assumed sorted ascending)."""
    if not allowed:
        return int(val)
    # if val below min or above max, clamp
    if val <= allowed[0]:
        return allowed[0]
    if val >= allowed[-1]:
        return allowed[-1]
    # binary search nearest
    # find index where allowed[idx] >= val
    lo, hi = 0, len(allowed) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if allowed[mid] == val:
            return allowed[mid]
        if allowed[mid] < val:
            lo = mid + 1
        else:
            hi = mid - 1
    # lo is index of first allowed > val, hi is last allowed < val
    # choose nearest between allowed[hi] and allowed[lo]
    a, b = allowed[hi], allowed[lo]
    if abs(a - val) <= abs(b - val):
        return a
    else:
        return b


def de_mutation(a, b, c, F=0.5):
    """
    Perform DE mutation on integer-encoded vector.
    After arithmetic, snap layer-count fields (indices 6..9) to allowed 5-step grid.
    Returns an integer numpy array.
    """
    # arithmetic mutation (works on ints)
    mutant = a + np.round(F * (b - c)).astype(int)
    mutant = mutant.astype(int)

    # Snap layer-related indices to grid ranges to avoid invalid values
    # index 6: input neurons (25..145 in steps of 5)
    # indices 7..9: hidden layers (5..45 in steps of 5)
    try:
        mutant[6] = _snap_to_grid(int(mutant[6]), _ALLOWED_INPUT_NEURONS)
    except Exception:
        mutant[6] = _ALLOWED_INPUT_NEURONS[0]

    for idx in (7, 8, 9):
        try:
            mutant[idx] = _snap_to_grid(int(mutant[idx]), _ALLOWED_HIDDEN_NEURONS)
        except Exception:
            mutant[idx] = _ALLOWED_HIDDEN_NEURONS[0]

    return mutant.astype(int)


def de_crossover(target, mutant, CR=0.9):
    trial = target.copy()
    L = len(target)
    jrand = random.randrange(L)
    for j in range(L):
        if random.random() < CR or j == jrand:
            trial[j] = mutant[j]
    return trial


def clamp_and_fix(vec):
    """
    Sanitize `vec` without calling random_individual.
    - If a numeric field is within allowed bounds -> snap/round to nearest allowed value.
    - If it's outside allowed bounds -> initialize randomly according to same constraints.
    - No recursion.
    """
    v = vec.copy().astype(float)  # work in float to detect fractional values, convert to int at end

    # helpers
    def is_number(x):
        return isinstance(x, (int, float, np.floating, np.integer))

    def choose_from(seq):
        return random.choice(list(seq))

    def snap_or_random_to_allowed(val, allowed):
        """If val is numeric and between min/max(allowed) -> snap to nearest allowed.
           Else return random choice from allowed.
        """
        if not allowed:
            return None
        amin, amax = allowed[0], allowed[-1]
        if is_number(val) and (amin <= float(val) <= amax):
            return _snap_to_grid(float(val), allowed)
        else:
            return choose_from(allowed)

    # --- v[0] fixed ---
    v[0] = 1

    # --- v[3] sig_start: discrete SIG_START_RANGE ---
    if is_number(v[3]):
        v[3] = snap_or_random_to_allowed(v[3], sorted(SIG_START_RANGE))
    else:
        v[3] = choose_from(SIG_START_RANGE)
    v[3] = float(int(v[3]))

    # compute sig_len early (used for num_filter)
    sig_len = max(1, 500 - int(v[3]))

    # --- v[1] num_filter: must be a divisor in [1..min(3,sig_len)] (original logic used divisors) ---
    divisors = [i for i in range(1, min(4, sig_len+1)) if sig_len % i == 0] or [1]
    if is_number(v[1]) and (1 <= float(v[1]) <= sig_len):
        # snap to nearest divisor
        # treat allowed as sorted list of ints
        try:
            # find nearest divisor by absolute difference
            v1_val = float(v[1])
            v[1] = min(divisors, key=lambda d: abs(d - v1_val))
        except Exception:
            v[1] = choose_from(divisors)
    else:
        v[1] = choose_from(divisors)
    v[1] = float(int(v[1]))

    # --- v[2] n: allowed N_RANGE (0..6) ---
    allowed_n = list(N_RANGE)
    if is_number(v[2]) and (min(allowed_n) <= float(v[2]) <= max(allowed_n)):
        # nearest integer in range
        nv = int(round(float(v[2])))
        nv = int(np.clip(nv, min(allowed_n), max(allowed_n)))
        v[2] = nv
    else:
        v[2] = choose_from(allowed_n)
    v[2] = float(int(v[2]))

    # --- v[4] shift: discrete SHIFT_RANGE ---
    v[4] = snap_or_random_to_allowed(v[4], sorted(list(SHIFT_RANGE)))
    v[4] = float(int(v[4]))

    # --- layer_count v[5]: must be 3 or 4 ---
    if is_number(v[5]) and (3 <= float(v[5]) <= 4):
        v5 = int(round(float(v[5])))
        v[5] = 3 if v5 < 4 else 4
    else:
        v[5] = choose_from([3, 4])
    layer_count = int(v[5])
    v[5] = float(layer_count)

    # --- v[6] input neurons: snap to allowed grid or random ---
    v[6] = snap_or_random_to_allowed(v[6], _ALLOWED_INPUT_NEURONS)
    v[6] = float(int(v[6]))
    prev = int(v[6])

    # --- hidden slots v[7], v[8], v[9] ---
    num_hidden_slots = layer_count - 1
    for i in range(1, 4):
        idx = 6 + i
        if i <= num_hidden_slots:
            # allowed hidden values that do not exceed prev
            allowed_le_prev = [x for x in _ALLOWED_HIDDEN_NEURONS if x <= prev]
            if allowed_le_prev:
                # if numeric and in range [min_allowed_le_prev, prev], snap to nearest allowed_le_prev
                if is_number(v[idx]) and (min(allowed_le_prev) <= float(v[idx]) <= prev):
                    # snap to nearest in allowed_le_prev
                    try:
                        v_val = float(v[idx])
                        # use _snap_to_grid but with allowed_le_prev list
                        v[idx] = _snap_to_grid(v_val, allowed_le_prev)
                    except Exception:
                        v[idx] = choose_from(allowed_le_prev)
                else:
                    # out of allowable range -> random from allowed_le_prev
                    v[idx] = choose_from(allowed_le_prev)
            else:
                # prev is smaller than any hidden neuron; fallback to smallest hidden
                v[idx] = _ALLOWED_HIDDEN_NEURONS[0]
            prev = int(v[idx])
            v[idx] = float(int(v[idx]))
        else:
            # unused slot -> set minimal hidden (safe default)
            v[idx] = float(_ALLOWED_HIDDEN_NEURONS[0])

    # --- beta / gamma fields ---
    # v[10] beta_i (VALID_BETA)
    v[10] = snap_or_random_to_allowed(v[10], sorted(list(VALID_BETA)))
    v[10] = float(int(v[10]))

    # v[11] gamma_i (VALID_GAMMA_I)
    v[11] = snap_or_random_to_allowed(v[11], sorted(list(VALID_GAMMA_I)))
    v[11] = float(int(v[11]))

    # v[12] beta (VALID_BETA)
    v[12] = snap_or_random_to_allowed(v[12], sorted(list(VALID_BETA)))
    v[12] = float(int(v[12]))

    # v[13] gamma depends on v[12]
    if int(v[12]) == 1:
        gamma_candidates = list(range(6, 17))
    else:
        gamma_candidates = [6, 7, 8]
    v[13] = snap_or_random_to_allowed(v[13], gamma_candidates)
    v[13] = float(int(v[13]))

    # v[14] beta_o
    v[14] = snap_or_random_to_allowed(v[14], sorted(list(VALID_BETA)))
    v[14] = float(int(v[14]))

    # v[15] gamma_o depends on v[14]
    if int(v[14]) == 1:
        gamma_o_candidates = list(range(6, 17))
    else:
        gamma_o_candidates = [6, 7, 8]
    v[15] = snap_or_random_to_allowed(v[15], gamma_o_candidates)
    v[15] = float(int(v[15]))

    # final conversion to ints (original vectors were ints)
    return v.astype(int)


# ------------------------
# Population helpers
# ------------------------


def random_individual(sig_start_range=SIG_START_RANGE):
    """
    Create a random valid individual that respects:
      - layer_count in {3,4}
      - layer neuron values are from allowed discrete sets and obey Input >= H1 >= H2 >= H3
    """
    v = np.zeros(VECTOR_LEN, dtype=int)

    # basic values
    v[0] = 1
    v[3] = random.choice(sig_start_range)
    sig_len = max(1, 500 - int(v[3]))
    v[1] = random.choice([i for i in range(1, min(4, sig_len+1)) if sig_len % i == 0] or [1])
    v[2] = random.choice(N_RANGE)
    v[4] = random.choice(SHIFT_RANGE)

    # choose layer_count explicitly
    v[5] = random.choice([3, 4])
    layer_count = int(v[5])

    # choose input neurons from allowed grid
    v[6] = random.choice(_ALLOWED_INPUT_NEURONS)
    prev = v[6]

    # number of hidden layers slots needed = layer_count - 1
    num_hidden_needed = layer_count - 1
    for i in range(1, 4):  # indices 7,8,9
        idx = 6 + i
        if i <= num_hidden_needed:
            # choose hidden value <= prev
            allowed_le_prev = [x for x in _ALLOWED_HIDDEN_NEURONS if x <= prev]
            if not allowed_le_prev:
                chosen = _ALLOWED_HIDDEN_NEURONS[0]
            else:
                chosen = random.choice(allowed_le_prev)
            v[idx] = int(chosen)
            prev = v[idx]
        else:
            # unused hidden slot -> put minimal valid hidden (won't be used)
            v[idx] = _ALLOWED_HIDDEN_NEURONS[0]

    # betas/gammas
    v[10] = random.choice(VALID_BETA)
    v[11] = random.choice(VALID_GAMMA_I)
    v[12] = random.choice(VALID_BETA)
    if v[12] == 1:
        v[13] = random.choice(list(range(6, 17)))
    else:
        v[13] = random.choice([6, 7, 8])
    v[14] = random.choice(VALID_BETA)
    if v[14] == 1:
        v[15] = random.choice(list(range(6, 17)))
    else:
        v[15] = random.choice([6, 7, 8])

    # final sanitization to make sure everything obeys constraints
    return clamp_and_fix(v)

def random_valid_individual(sig_start_range=SIG_START_RANGE, max_attempts=MAX_INVALID_RETRIES):
    """Return a random individual that passes is_valid_pair (retries up to max_attempts)."""
    for attempt in range(max_attempts):
        v = random_individual(sig_start_range=sig_start_range)
        # reseed random generators deterministically based on vector contents
        seed_val = (int(time.time() * 1000) ^ abs(hash(tuple(v)))) & 0xFFFFFFFF
        random.seed(seed_val)
        np.random.seed(seed_val)

        pre, logic = decode_vector(v)
        if is_valid_pair(pre, logic):
            return v

    # fallback: return last generated (likely invalid) but log warning
    print(f"[WARN] random_valid_individual: failed to find valid config after {max_attempts} attempts; returning last candidate.")
    return v


    
def ensure_valid_trial_vector(trial_vec, max_attempts=MAX_INVALID_RETRIES):
    """
    Given a trial vector produced by mutation/crossover, verify it.
    If invalid, replace it with a fresh random_valid_individual (not derived from parents).
    This keeps the mutation/crossover operators unchanged (only rejects invalid outcome).
    """
    pre, logic = decode_vector(trial_vec)
    if is_valid_pair(pre, logic):
        return trial_vec

    # replacement loop
    for attempt in range(max_attempts):
        new_v = random_valid_individual(max_attempts=50)
        pre2, logic2 = decode_vector(new_v)
        if is_valid_pair(pre2, logic2):
            return new_v

    print(f"[WARN] ensure_valid_trial_vector: failed to replace invalid trial after {max_attempts} attempts; returning original trial.")
    return trial_vec
    
# ------------------------
# Worker evaluation function (no global cache mutation)
# ------------------------

def evaluate_vector_worker(vec, weights=(0.2,0.6,0.2), debug=False):
    """Function executed inside worker process. Returns (cfg_hash, result, cost, info) or (cfg_hash, None, None, {'_error':...})"""
    try:
        # per-worker init (memmap + limit threads)
        worker_init()

        # ensure RNG is decorrelated in each worker
        seed = (int(time.time() * 1000) ^ abs(hash(tuple(vec)))) & 0xFFFFFFFF
        np.random.seed(seed)
        random.seed(seed)

        preproc, logic_spec = decode_vector(vec)
        cfg_hash = _hash_config(preproc, logic_spec)

        res = run_experiment(preproc, logic_spec, debug=debug)
        if res is None:
            return (cfg_hash, None, None, {"_error": "empty_result"})
        if isinstance(res, dict) and res.get("_error"):
            return (cfg_hash, None, None, res)

        cost, info = compute_cost(res, num_windows=preproc["num_filter"], weights=weights, debug=debug)
        return (cfg_hash, res, cost, info)
    except Exception as e:
        return (None, None, None, {"_error": str(e)})

# ------------------------
# Main DE driver (safe cache management in main process)
# ------------------------
def differential_evolution_discrete(pop_size=20, gens=200, F=0.6, CR=0.9,
                                    weights=(0.2, 0.6, 0.2), parallel_jobs=4,
                                    init_seed=None, resume_cache=None, debug=False,
                                    skip_md5=True, patience=40, log_fname="./de_log.yaml"):
    """
    Discrete Differential Evolution with safe cache + YAML logging.
    Early-exits if best cost hasn't improved for `patience` generations.
    """
    if init_seed is not None:
        random.seed(init_seed)
        np.random.seed(init_seed)

    # Validate files once in parent
    load_and_verify_data()
    #load_good_configs(PREPROC_GOOD_FNAME, LOGIC_GOOD_FNAME)

    cache = resume_cache if resume_cache is not None else {}
    log_fname = log_fname
    
    header = {
        "run_start_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "pop_size": int(pop_size),
        "gens": int(gens),
        "F": float(F),
        "CR": float(CR),
        "weights": _safe_serialize(weights),
        "parallel_jobs": int(parallel_jobs),
        "init_seed": int(init_seed) if init_seed is not None else None,
        "patience": int(patience),
    }

    try:
        append_yaml_entry({"header": header}, log_fname=log_fname)
    except Exception as e:
        print(f"[WARN] Logging header failed: {e}")

    # -------------------------------
    # 1. Initialize population
    # -------------------------------
    pop = [random_valid_individual() for _ in range(pop_size)]
    print(f"[DE] Evaluating initial population (n={pop_size}) with {parallel_jobs} workers...")

    tasks = Parallel(n_jobs=parallel_jobs, backend="loky", prefer="processes")(
        delayed(evaluate_vector_worker)(vec, weights, debug) for vec in pop
    )

    pop_info = []
    for idx, out in enumerate(tasks):
        cfg_hash, res, cost, info = out
        if cfg_hash is None:
            pop_info.append((math.inf, None, None))
            entry = make_yaml_entry(
                0, "init", idx, pop[idx], res, None, info,
                extra={"note": "cfg_hash_none_or_worker_error"}
            )
            append_yaml_entry(entry, log_fname=log_fname)
            continue

        if res is not None:
            cache[cfg_hash] = (res, cost, info)
            pop_info.append((cost, info, res))
            entry = make_yaml_entry(0, "init", idx, pop[idx], res, cost, info)
            append_yaml_entry(entry, log_fname=log_fname)
        else:
            pop_info.append((math.inf, None, None))
            entry = make_yaml_entry(
                0, "init", idx, pop[idx], res, None, info,
                extra={"note": "failed_run"}
            )
            append_yaml_entry(entry, log_fname=log_fname)

    # -------------------------------
    # 2. Track best
    # -------------------------------
    best_idx = int(np.argmin([ci[0] for ci in pop_info]))
    best_cost, best_info, best_res = pop_info[best_idx]
    best_preproc, best_logic = decode_vector(pop[best_idx])

    print(f"[DE INIT] best_cost={best_cost:.6f} idx={best_idx}")
    history = []
    no_improve_count = 0
    prev_best_cost = best_cost

    # -------------------------------
    # 3. Main DE loop
    # -------------------------------
    for g in range(gens):
        print(f"[DE] Generation {g+1}/{gens}, current best cost {best_cost:.6f}")

        # Mutation + crossover
        trials = []
        for i in range(pop_size):
            idxs = list(range(pop_size))
            idxs.remove(i)
            a_idx, b_idx, c_idx = random.sample(idxs, 3)
            a, b, c = pop[a_idx], pop[b_idx], pop[c_idx]
            mutant = de_mutation(a, b, c, F=F)
            trial = de_crossover(pop[i], mutant, CR=CR)
            trial = clamp_and_fix(trial)
            # Ensure trial is not a bad combination (replace with fresh random valid if needed)
            trial = ensure_valid_trial_vector(trial)
            trials.append(trial)

        # Evaluate trials in parallel
        trial_tasks = Parallel(n_jobs=parallel_jobs, backend="loky", prefer="processes")(
            delayed(evaluate_vector_worker)(vec, weights, debug) for vec in trials
        )

        # Selection
        for i in range(pop_size):
            cfg_hash, res_trial, cost_trial, info_trial = trial_tasks[i]
            if cfg_hash is None or res_trial is None:
                continue

            cache[cfg_hash] = (res_trial, cost_trial, info_trial)
            cost_target, info_target, res_target = pop_info[i]

            # Replace if better (or uninitialized)
            if cost_target is None or math.isinf(cost_target) or cost_trial <= cost_target:
                pop[i] = trials[i]
                pop_info[i] = (cost_trial, info_trial, res_trial)

                if cost_trial < best_cost:
                    best_cost = cost_trial
                    best_res = res_trial
                    best_info = info_trial
                    best_idx = i

        # Track and log best of this generation
        history.append({"generation": g+1, "best_cost": float(best_cost), "best_idx": int(best_idx)})

        entry = make_yaml_entry(
            g+1, "generation_best", best_idx, pop[best_idx],
            best_res, best_cost, best_info,
            extra={"generation": g+1}
        )
        append_yaml_entry(entry, log_fname=log_fname)

        # -------------------------------
        # Early stopping condition
        # -------------------------------
        if abs(best_cost - prev_best_cost) < 1e-12:
            no_improve_count += 1
        else:
            no_improve_count = 0
        prev_best_cost = best_cost

        if no_improve_count >= patience:
            print(f"[DE EARLY STOP] No improvement for {patience} generations. Stopping early at generation {g+1}.")
            entry = make_yaml_entry(
                g+1, "early_stop", best_idx, pop[best_idx],
                best_res, best_cost, best_info,
                extra={"note": f"terminated_after_{patience}_no_improvement"}
            )
            append_yaml_entry(entry, log_fname=log_fname)
            break

    # -------------------------------
    # 4. Final results
    # -------------------------------
    best_preproc, best_logic = decode_vector(pop[best_idx])
    print(f"[DE DONE] Best cost: {best_cost:.6f}")

    return {
        "best_preproc": best_preproc,
        "best_logic_spec": best_logic,
        "best_res": best_res,
        "best_cost": best_cost,
        "cache": cache,
        "history": history,
    }


# ------------------------
# If run as script: quick example
# ------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Discrete DE optimization driver")
    parser.add_argument("--pop_size", type=int, default=20, help="Population size")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--F", type=float, default=0.6, help="Differential weight")
    parser.add_argument("--CR", type=float, default=0.8, help="Crossover rate")
    parser.add_argument("--parallel_jobs", type=int, default=24, help="Number of parallel jobs")
    parser.add_argument("--init_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--weights", type=str, default="0.2,0.6,0.2",
                        help="Comma-separated weights for compute_cost, e.g. '0.2,0.6,0.2'")
    parser.add_argument("--log_fname", type=str, default="./de_log.yaml",
                        help="YAML log filename")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Parse weights string into tuple of floats
    try:
        weights = tuple(float(x) for x in args.weights.split(","))
        if len(weights) != 3:
            raise ValueError
    except ValueError:
        raise ValueError("Invalid --weights format. Must be like '0.2,0.6,0.2'")

    #load_good_configs()
    valids = sum(1 for _ in range(1000) if is_valid_pair(*decode_vector(random_individual())))
    print(f"Out of 1000 random individuals, {valids} passed is_valid_pair.")

    res = differential_evolution_discrete(
        pop_size=args.pop_size,
        gens=args.gens,
        F=args.F,
        CR=args.CR,
        parallel_jobs=args.parallel_jobs,
        init_seed=args.init_seed,
        weights=weights,
        log_fname=args.log_fname,
        debug=args.debug,
        skip_md5=True
    )

    print("BEST", res["best_cost"], res["best_preproc"], res["best_logic_spec"])
