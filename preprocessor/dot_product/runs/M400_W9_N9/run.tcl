
# Auto-generated TCL
read_verilog -sv /mnt/vault1/mfaroo19/reduction_tree/dot_product/runs/M400_W9_N9/design.sv
read_xdc /mnt/vault1/mfaroo19/reduction_tree/dot_product/runs/M400_W9_N9/design.xdc
synth_design -top dotprod -part xczu7ev-ffvc1156-2-e
report_utilization -file /mnt/vault1/mfaroo19/reduction_tree/dot_product/runs/M400_W9_N9/util.rpt
report_timing_summary -file /mnt/vault1/mfaroo19/reduction_tree/dot_product/runs/M400_W9_N9/timing.rpt
exit
