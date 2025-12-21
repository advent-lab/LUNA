module layer2 (input [4:0] M0, output [4:0] M1);

wire [4:0] layer2_N0_wire = {M0[4], M0[3], M0[2], M0[1], M0[0]};
layer2_N0 layer2_N0_inst (.M0(layer2_N0_wire), .M1(M1[0:0]));

wire [4:0] layer2_N1_wire = {M0[4], M0[3], M0[2], M0[1], M0[0]};
layer2_N1 layer2_N1_inst (.M0(layer2_N1_wire), .M1(M1[1:1]));

wire [4:0] layer2_N2_wire = {M0[4], M0[3], M0[2], M0[1], M0[0]};
layer2_N2 layer2_N2_inst (.M0(layer2_N2_wire), .M1(M1[2:2]));

wire [4:0] layer2_N3_wire = {M0[4], M0[3], M0[2], M0[1], M0[0]};
layer2_N3 layer2_N3_inst (.M0(layer2_N3_wire), .M1(M1[3:3]));

wire [4:0] layer2_N4_wire = {M0[4], M0[3], M0[2], M0[1], M0[0]};
layer2_N4 layer2_N4_inst (.M0(layer2_N4_wire), .M1(M1[4:4]));

endmodule