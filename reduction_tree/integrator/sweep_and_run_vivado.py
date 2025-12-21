#!/usr/bin/env python3
import os
import subprocess
import argparse
import csv
from mako.template import Template
from textwrap import dedent
from pathlib import Path

def cmd_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True,
                        help="Path to Mako template (e.g., sum_signed.mako)")
    parser.add_argument("--out-dir", required=True,
                        help="Directory where all runs/results are stored")
    parser.add_argument("--part", required=True,
                        help="FPGA part (e.g., xczu7ev-ffvc1156-2-i)")
    parser.add_argument("--top", required=True,
                        help="Top-level module name")
    parser.add_argument("--vivado-bin", default="vivado",
                        help="Path to Vivado binary")
    parser.add_argument("--M", type=int, nargs="+", required=True,
                        help="List of numbers of inputs (space-separated)")
    parser.add_argument("--W", type=int, nargs="+", required=True,
                        help="List of bit widths (space-separated)")
    parser.add_argument("--csv", required=True,
                        help="Output CSV filename")
    parser.add_argument("--clk-period", type=float, default=10.0,
                        help="Clock period in ns for XDC")
    return parser.parse_args()

def render_template(template_fname, out_fname, M, W):
    template_basename = os.path.basename(template_fname)
    header_text = dedent("""\
        ////////////////////////////////////////////////////////////////////////////////
        // THIS FILE WAS AUTOMATICALLY GENERATED FROM ${filename}
        // DO NOT EDIT
        ////////////////////////////////////////////////////////////////////////////////

    """)
    header = Template(header_text).render(filename=template_basename)

    with open(template_fname, "r") as f:
        template = Template(f.read())
    rendered = template.render(M=M, W=W)
    output = header + rendered

    Path(os.path.dirname(out_fname)).mkdir(parents=True, exist_ok=True)
    with open(out_fname, "w") as f:
        f.write(output)

def write_vivado_tcl(tcl_path, top, part, verilog_file, report_dir, xdc_file):
    tcl_script = f"""
read_verilog -sv {verilog_file}
read_xdc {xdc_file}
synth_design -top {top} -part {part}
report_utilization -file {report_dir}/util.rpt
report_timing_summary -file {report_dir}/timing.rpt
exit
"""
    with open(tcl_path, "w") as f:
        f.write(tcl_script)

def write_xdc(xdc_path, clk_period_ns):
    """Write a basic XDC file with one clock"""
    with open(xdc_path, "w") as f:
        f.write(f"create_clock -period {clk_period_ns} [get_ports clk]\n")

def run_vivado(vivado_bin, tcl_path, log_path):
    with open(log_path, "w") as logf:
        subprocess.run([vivado_bin, "-mode", "batch", "-source", tcl_path],
                       stdout=logf, stderr=logf, check=True)

def parse_reports(report_dir):
    util_file = os.path.join(report_dir, "util.rpt")
    timing_file = os.path.join(report_dir, "timing.rpt")

    lut, ff = None, None
    fmax = None

    # Parse utilization
    if os.path.exists(util_file):
        with open(util_file) as f:
            for line in f:
                if "Slice LUTs" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        lut = parts[1]
                if "Slice Registers" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        ff = parts[1]

    # Parse timing summary
    if os.path.exists(timing_file):
        with open(timing_file) as f:
            for line in f:
                if "Delay" in line and "ns" in line:
                    try:
                        delay = float(line.split()[0])
                        fmax = 1000.0 / delay  # MHz
                    except Exception:
                        pass

    return lut, ff, fmax

def main():
    args = cmd_line_args()

    results = []
    for M in args.M:
        for W in args.W:
            run_name = f"M{M}_W{W}"
            run_dir = os.path.join(args.out_dir, run_name)
            Path(run_dir).mkdir(parents=True, exist_ok=True)

            # Generate SV
            verilog_file = os.path.join(run_dir, f"{args.top}.sv")
            render_template(args.template, verilog_file, M, W)

            # Generate XDC
            xdc_file = os.path.join(run_dir, f"{args.top}.xdc")
            write_xdc(xdc_file, clk_period_ns=args.clk_period)

            # Generate TCL
            tcl_file = os.path.join(run_dir, "run.tcl")
            write_vivado_tcl(tcl_file, args.top, args.part, verilog_file, run_dir, xdc_file)

            log_file = os.path.join(run_dir, "vivado.log")
            print(f"[INFO] Running Vivado for {run_name}...")
            run_vivado(args.vivado_bin, tcl_file, log_file)

            # Parse reports
            lut, ff, fmax = parse_reports(run_dir)
            results.append({"M": M, "W": W, "LUTs": lut, "FFs": ff, "Fmax_MHz": fmax})

    # Write results to CSV
    with open(args.csv, "w", newline="") as csvfile:
        fieldnames = ["M", "W", "LUTs", "FFs", "Fmax_MHz"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[INFO] Results written to {args.csv}")

if __name__ == "__main__":
    main()
