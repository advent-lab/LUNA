
read_verilog -sv runs/M450_W12/sum_signed.sv
read_xdc runs/M450_W12/sum_signed.xdc
synth_design -top sum_signed -part xczu7ev-ffvc1156-2-e
report_utilization -file runs/M450_W12/util.rpt
report_timing_summary -file runs/M450_W12/timing.rpt
exit
