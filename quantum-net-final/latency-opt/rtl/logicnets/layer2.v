module layer2 (input [34:0] M0, output [14:0] M1);

wire [5:0] layer2_N0_wire = {M0[31], M0[23], M0[21], M0[19], M0[10], M0[9]};
layer2_N0 layer2_N0_inst (.M0(layer2_N0_wire), .M1(M1[0:0]));

wire [5:0] layer2_N1_wire = {M0[34], M0[25], M0[18], M0[11], M0[8], M0[1]};
layer2_N1 layer2_N1_inst (.M0(layer2_N1_wire), .M1(M1[1:1]));

wire [5:0] layer2_N2_wire = {M0[29], M0[21], M0[18], M0[11], M0[7], M0[3]};
layer2_N2 layer2_N2_inst (.M0(layer2_N2_wire), .M1(M1[2:2]));

wire [5:0] layer2_N3_wire = {M0[33], M0[32], M0[25], M0[23], M0[6], M0[0]};
layer2_N3 layer2_N3_inst (.M0(layer2_N3_wire), .M1(M1[3:3]));

wire [5:0] layer2_N4_wire = {M0[34], M0[31], M0[17], M0[13], M0[6], M0[0]};
layer2_N4 layer2_N4_inst (.M0(layer2_N4_wire), .M1(M1[4:4]));

wire [5:0] layer2_N5_wire = {M0[33], M0[31], M0[30], M0[26], M0[10], M0[0]};
layer2_N5 layer2_N5_inst (.M0(layer2_N5_wire), .M1(M1[5:5]));

wire [5:0] layer2_N6_wire = {M0[32], M0[29], M0[28], M0[24], M0[9], M0[4]};
layer2_N6 layer2_N6_inst (.M0(layer2_N6_wire), .M1(M1[6:6]));

wire [5:0] layer2_N7_wire = {M0[19], M0[12], M0[11], M0[9], M0[7], M0[3]};
layer2_N7 layer2_N7_inst (.M0(layer2_N7_wire), .M1(M1[7:7]));

wire [5:0] layer2_N8_wire = {M0[26], M0[18], M0[13], M0[7], M0[4], M0[2]};
layer2_N8 layer2_N8_inst (.M0(layer2_N8_wire), .M1(M1[8:8]));

wire [5:0] layer2_N9_wire = {M0[27], M0[24], M0[23], M0[22], M0[21], M0[19]};
layer2_N9 layer2_N9_inst (.M0(layer2_N9_wire), .M1(M1[9:9]));

wire [5:0] layer2_N10_wire = {M0[29], M0[28], M0[27], M0[26], M0[9], M0[2]};
layer2_N10 layer2_N10_inst (.M0(layer2_N10_wire), .M1(M1[10:10]));

wire [5:0] layer2_N11_wire = {M0[25], M0[23], M0[22], M0[21], M0[6], M0[5]};
layer2_N11 layer2_N11_inst (.M0(layer2_N11_wire), .M1(M1[11:11]));

wire [5:0] layer2_N12_wire = {M0[33], M0[24], M0[20], M0[12], M0[7], M0[1]};
layer2_N12 layer2_N12_inst (.M0(layer2_N12_wire), .M1(M1[12:12]));

wire [5:0] layer2_N13_wire = {M0[25], M0[24], M0[23], M0[21], M0[9], M0[3]};
layer2_N13 layer2_N13_inst (.M0(layer2_N13_wire), .M1(M1[13:13]));

wire [5:0] layer2_N14_wire = {M0[29], M0[22], M0[21], M0[16], M0[9], M0[0]};
layer2_N14 layer2_N14_inst (.M0(layer2_N14_wire), .M1(M1[14:14]));

endmodule