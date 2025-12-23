
read_verilog -sv ./vivado_runs/M16_W16/sum_signed.sv
read_xdc ./vivado_runs/M16_W16/sum_signed.xdc
synth_design -top sum_signed -part xczu7ev-ffvc1156-2-i
report_utilization -file ./vivado_runs/M16_W16/util.rpt
report_timing_summary -file ./vivado_runs/M16_W16/timing.rpt
exit
