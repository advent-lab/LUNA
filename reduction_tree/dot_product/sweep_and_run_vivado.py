#!/usr/bin/env python3
import os
import subprocess
import argparse
import csv
import re
from mako.template import Template

def write_sv(filename, template_fname, M, W, N, top="dotprod"):
    tmpl = Template(filename=template_fname)
    with open(filename, "w") as f:
        f.write(tmpl.render(M=M, W=W, N=N, top=top))

def write_xdc(xdc_path, clk_period_ns):
    with open(xdc_path, "w") as f:
        f.write(f"create_clock -period {clk_period_ns} [get_ports clk]\n")

def write_vivado_tcl(tcl_path, top, part, verilog_file, report_dir, xdc_file):
    tcl_script = f"""
# Auto-generated TCL
read_verilog -sv {verilog_file}
read_xdc {xdc_file}
synth_design -top {top} -part {part}
report_utilization -file {report_dir}/util.rpt
report_timing_summary -file {report_dir}/timing.rpt
exit
"""
    with open(tcl_path, "w") as f:
        f.write(tcl_script)

def run_vivado(run_dir, vivado_bin, tcl_path):
    """
    Run Vivado quietly in batch mode, redirecting all output to vivado.log
    """
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "vivado.log")
    with open(log_path, "w") as logf:
        subprocess.run(
            [vivado_bin, "-mode", "batch", "-source", tcl_path],
            cwd=run_dir,
            stdout=logf,
            stderr=logf,
            check=True
        )

def scrape_reports(report_dir):
    util_file = os.path.join(report_dir, "util.rpt")
    timing_file = os.path.join(report_dir, "timing.rpt")

    lut_used = 0
    dsp_used = 0
    wns = 0.0

    # -------------------
    # Scrape utilization
    # -------------------
    if os.path.exists(util_file):
        with open(util_file) as f:
            for line in f:
                if re.search(r'\bCLB LUTs\*', line):
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 2:
                        try:
                            lut_used = int(parts[1])
                        except ValueError:
                            print(f"[WARN] Could not parse LUTs from line: {line.strip()}")
                elif re.search(r'\bDSPs\b', line):
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 2:
                        try:
                            dsp_used = int(parts[1])
                        except ValueError:
                            print(f"[WARN] Could not parse DSPs from line: {line.strip()}")
    else:
        print(f"[WARN] Missing utilization report: {util_file}")

    # -------------------
    # Scrape timing report (WNS)
    # -------------------
    setup_pattern = re.compile(
        r"Setup\s*:\s*\d+\s+Failing Endpoints,\s+Worst Slack\s+(-?\d+\.\d+)ns"
    )
    if os.path.exists(timing_file):
        with open(timing_file) as f:
            for line in f:
                m = setup_pattern.search(line)
                if m:
                    try:
                        wns = float(m.group(1))
                    except ValueError:
                        print(f"[WARN] Could not parse WNS from line: {line.strip()}")
                    break
    else:
        print(f"[WARN] Missing timing report: {timing_file}")

    return lut_used, dsp_used, wns

def sweep(args):
    results = []
    for M in args.M:
        for W in args.W:
            for N in args.N:
                run_name = f"M{M}_W{W}_N{N}"
                run_dir = os.path.abspath(os.path.join(args.out_dir, run_name))
                os.makedirs(run_dir, exist_ok=True)
                verilog_file = os.path.join(run_dir, "design.sv")
                xdc_file = os.path.join(run_dir, "design.xdc")
                tcl_file = os.path.join(run_dir, "run.tcl")
                report_dir = run_dir

                write_sv(verilog_file, args.template, M, W, N, top=args.top)
                write_xdc(xdc_file, args.clk)
                write_vivado_tcl(tcl_file, args.top, args.part, verilog_file, report_dir, xdc_file)

                print(f"\n[INFO] Running {run_name} quietly ...")
                run_vivado(run_dir, args.vivado_bin, tcl_file)

                lut_used, dsp_used, wns = scrape_reports(report_dir)
                print(f"[RESULT] M={M}, W={W}, N={N}, LUTs={lut_used}, DSPs={dsp_used}, WNS={wns}")

                results.append((M, W, N, lut_used, dsp_used, wns))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, help="Path to mako template (.sv.mako)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--top", required=True)
    parser.add_argument("--vivado-bin", default="vivado")
    parser.add_argument("--M", nargs="+", type=int, required=True)
    parser.add_argument("--W", nargs="+", type=int, required=True)
    parser.add_argument("--N", nargs="+", type=int, required=True)
    parser.add_argument("--csv", default="results.csv")
    parser.add_argument("--clk", type=float, default=1.0)
    args = parser.parse_args()

    results = sweep(args)

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "W", "N", "LUTs", "DSPs", "WNS"])
        writer.writerows(results)

    print(f"\n[INFO] Results written to {args.csv}")

