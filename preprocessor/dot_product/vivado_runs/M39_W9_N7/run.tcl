
# Auto-generated TCL
read_verilog -sv /mnt/vault1/mfaroo19/reduction_tree/dot_product/vivado_runs/M39_W9_N7/design.sv
read_xdc /mnt/vault1/mfaroo19/reduction_tree/dot_product/vivado_runs/M39_W9_N7/design.xdc
synth_design -top dotprod -part xczu7ev-ffvc1156-2-i
report_utilization -file /mnt/vault1/mfaroo19/reduction_tree/dot_product/vivado_runs/M39_W9_N7/util.rpt
report_timing_summary -file /mnt/vault1/mfaroo19/reduction_tree/dot_product/vivado_runs/M39_W9_N7/timing.rpt
exit
