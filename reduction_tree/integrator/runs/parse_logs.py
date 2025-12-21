#!/usr/bin/env python3
"""
parse_vivado_logs.py

Search folders named M{num}_W{width} and extract:
 - LUTs (from util.rpt, "LUT as Logic" -> Used)
 - WNS  (from timing.rpt, "Worst Slack ... Xns" -> X)

Usage:
    python parse_vivado_logs.py                # scans current directory (non-recursive), prints CSV to stdout
    python parse_vivado_logs.py --root /path/to/topdir --recursive --out out.csv
"""

from pathlib import Path
import re
import argparse
import csv
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FOLDER_RE = re.compile(r'^M(\d+)_W(\d+)$', re.IGNORECASE)

def parse_util_rpt(path: Path):
    """
    Parse util.rpt to extract 'LUT as Logic' -> Used value (int).
    Returns int or None if not found.
    """
    if not path.exists():
        return None
    text = path.read_text(errors='ignore')
    # Find a line that has 'LUT as Logic' (case-insensitive)
    for line in text.splitlines():
        if 'lut as logic' in line.lower():
            # find integer-like tokens on the line (commas allowed)
            nums = re.findall(r'[\d,]+(?:\.\d+)?', line)
            if not nums:
                continue
            # pick first integer-looking token (should be the "Used" column)
            first = nums[0].replace(',', '')
            # sometimes floats may appear (unlikely for Used) -> try integer cast
            try:
                val = int(float(first))
                return val
            except Exception:
                try:
                    return int(first)
                except Exception:
                    return None
    # fallback: search anywhere for pattern like "| LUT as Logic | 3135 |"
    m = re.search(r'\|\s*LUT as Logic\s*\|\s*([\d,]+)', text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1).replace(',', ''))
        except:
            return None
    return None

def parse_timing_rpt(path: Path):
    """
    Parse timing.rpt to extract the WNS (Worst Slack) in ns.
    Returns float (ns) or None if not found.
    Looks for patterns like:
      "Worst Slack        0.175ns"
      "Worst Slack        -0.123ns,"
      "Worst Negative Slack ... -0.123"
      "Setup : ... Worst Slack        0.175ns,  Total Violation ..."
    """
    if not path.exists():
        return None
    text = path.read_text(errors='ignore')

    # try several likely regex patterns (case-insensitive)
    patterns = [
        r'Worst Slack\s*[:\s,]*([+-]?\d+(?:\.\d+)?)\s*ns',                # Worst Slack ... 0.175ns
        r'Worst Negative Slack[^\d\-\+]*([+-]?\d+(?:\.\d+)?)',           # Worst Negative Slack ... -0.123
        r'Worst Slack[^\d\-\+]*([+-]?\d+(?:\.\d+)?)',                    # looser: Worst Slack ... 0.175
        r'Worst negative slack[^\d\-\+]*([+-]?\d+(?:\.\d+)?)',           # variant casing
        # some Vivado prints like: "Worst Slack        -0.123ns,  Total Violation"
        r'Worst Slack[^\n]*?([+-]?\d+(?:\.\d+)?)ns'
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except:
                continue

    # fallback: sometimes line contains "Worst Slack" with commas etc
    for line in text.splitlines():
        if 'worst slack' in line.lower():
            m = re.search(r'([+-]?\d+(?:\.\d+)?)\s*ns', line, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except:
                    pass
    return None

def find_target_dirs(root: Path, recursive: bool):
    """Yield directories that match M{num}_W{width} under root."""
    if recursive:
        for p in root.rglob('*'):
            if p.is_dir() and FOLDER_RE.match(p.name):
                yield p
    else:
        for p in root.iterdir():
            if p.is_dir() and FOLDER_RE.match(p.name):
                yield p

def parse_folder_name(name: str):
    m = FOLDER_RE.match(name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def main():
    ap = argparse.ArgumentParser(description="Parse Vivado util.rpt and timing.rpt across folders named M{n}_W{w}")
    ap.add_argument("--root", "-r", type=str, default=".", help="Root directory to scan (default: current dir)")
    ap.add_argument("--recursive", action="store_true", help="Search recursively under root")
    ap.add_argument("--out", "-o", type=str, default=None, help="Output CSV file (defaults to stdout)")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        logging.error("Root path '%s' does not exist or is not a directory.", root)
        sys.exit(2)

    rows = []
    for d in sorted(find_target_dirs(root, args.recursive), key=lambda p: p.name):
        num_inputs, width = parse_folder_name(d.name)
        util_path = d / "util.rpt"
        timing_path = d / "timing.rpt"

        luts = parse_util_rpt(util_path)
        wns = parse_timing_rpt(timing_path)

        if luts is None:
            logging.debug("Could not find 'LUT as Logic' in %s", util_path)
        if wns is None:
            logging.debug("Could not find WNS in %s", timing_path)

        rows.append({
            "folder": str(d),
            "num_inputs": num_inputs,
            "width": width,
            "luts": luts if luts is not None else "",
            "wns_ns": wns if wns is not None else ""
        })

    # Output CSV
    fieldnames = ["folder", "num_inputs", "width", "luts", "wns_ns"]
    if args.out:
        outpath = Path(args.out)
        with outpath.open("w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        logging.info("Wrote results to %s", outpath)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

if __name__ == "__main__":
    main()

