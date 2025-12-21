# 1. CREATE PROJECT
# Create a new Vivado project
create_project balanced-classifier ./balanced-classifer -part xczu7ev-ffvc1156-2-e -force

# Set the current project for subsequent commands
#set_property board_part xilinx.com:zcu106:part0:1.4
# Note: You may need to change the 'board_part' or remove this line if you are only using the 'part'

# 2. ADD SOURCES
# Read Verilog/SystemVerilog files and set them as Design Sources
add_files -fileset sources_1 -norecurse ./readout_ip.sv ./sum_signed.sv

# Read the LogicNets directory as a source (assuming it contains modules/includes)
add_files -fileset sources_1 -norecurse "./logicnets"
# Add the directory to the global include path for synthesis
set_property include_dirs "logicnets/" [current_fileset]

# Read the XDC file and set it as a Constraints Source
read_xdc ./constraints.xdc

# Set the top module name
set_property top nn_classifier_wrapper [current_fileset]

# 3. RUN FLOW STEPS
# Run Synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Run Implementation (including Opt, Place, Phys Opt, Route)
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# 4. REPORTING (Uses project run results)
# Open the synthesized and implemented design checkpoints
open_run impl_1

# Generate the reports from the implemented design
report_utilization -file ./util_project.rpt
report_timing_summary -file ./timing_project.rpt

# 5. SAVE AND EXIT
# Save the project (saves all settings and checkpoints)
save_project

# Exit
exit
