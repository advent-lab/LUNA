module layer0 (input [27:0] M0, output [24:0] M1);

wire [5:0] layer0_N0_wire = {M0[27], M0[21], M0[19], M0[18], M0[15], M0[1]};
layer0_N0 layer0_N0_inst (.M0(layer0_N0_wire), .M1(M1[0:0]));

wire [5:0] layer0_N1_wire = {M0[26], M0[21], M0[19], M0[10], M0[7], M0[4]};
layer0_N1 layer0_N1_inst (.M0(layer0_N1_wire), .M1(M1[1:1]));

wire [5:0] layer0_N2_wire = {M0[27], M0[25], M0[24], M0[13], M0[8], M0[3]};
layer0_N2 layer0_N2_inst (.M0(layer0_N2_wire), .M1(M1[2:2]));

wire [5:0] layer0_N3_wire = {M0[22], M0[20], M0[19], M0[5], M0[3], M0[1]};
layer0_N3 layer0_N3_inst (.M0(layer0_N3_wire), .M1(M1[3:3]));

wire [5:0] layer0_N4_wire = {M0[25], M0[24], M0[21], M0[20], M0[16], M0[2]};
layer0_N4 layer0_N4_inst (.M0(layer0_N4_wire), .M1(M1[4:4]));

wire [5:0] layer0_N5_wire = {M0[27], M0[26], M0[21], M0[13], M0[6], M0[0]};
layer0_N5 layer0_N5_inst (.M0(layer0_N5_wire), .M1(M1[5:5]));

wire [5:0] layer0_N6_wire = {M0[23], M0[16], M0[8], M0[4], M0[3], M0[2]};
layer0_N6 layer0_N6_inst (.M0(layer0_N6_wire), .M1(M1[6:6]));

wire [5:0] layer0_N7_wire = {M0[22], M0[13], M0[8], M0[3], M0[1], M0[0]};
layer0_N7 layer0_N7_inst (.M0(layer0_N7_wire), .M1(M1[7:7]));

wire [5:0] layer0_N8_wire = {M0[21], M0[20], M0[17], M0[16], M0[4], M0[2]};
layer0_N8 layer0_N8_inst (.M0(layer0_N8_wire), .M1(M1[8:8]));

wire [5:0] layer0_N9_wire = {M0[26], M0[25], M0[20], M0[7], M0[6], M0[0]};
layer0_N9 layer0_N9_inst (.M0(layer0_N9_wire), .M1(M1[9:9]));

wire [5:0] layer0_N10_wire = {M0[26], M0[21], M0[18], M0[17], M0[2], M0[0]};
layer0_N10 layer0_N10_inst (.M0(layer0_N10_wire), .M1(M1[10:10]));

wire [5:0] layer0_N11_wire = {M0[26], M0[25], M0[21], M0[13], M0[12], M0[2]};
layer0_N11 layer0_N11_inst (.M0(layer0_N11_wire), .M1(M1[11:11]));

wire [5:0] layer0_N12_wire = {M0[22], M0[21], M0[18], M0[9], M0[6], M0[3]};
layer0_N12 layer0_N12_inst (.M0(layer0_N12_wire), .M1(M1[12:12]));

wire [5:0] layer0_N13_wire = {M0[26], M0[22], M0[19], M0[18], M0[8], M0[5]};
layer0_N13 layer0_N13_inst (.M0(layer0_N13_wire), .M1(M1[13:13]));

wire [5:0] layer0_N14_wire = {M0[27], M0[26], M0[25], M0[22], M0[10], M0[3]};
layer0_N14 layer0_N14_inst (.M0(layer0_N14_wire), .M1(M1[14:14]));

wire [5:0] layer0_N15_wire = {M0[26], M0[20], M0[17], M0[10], M0[3], M0[1]};
layer0_N15 layer0_N15_inst (.M0(layer0_N15_wire), .M1(M1[15:15]));

wire [5:0] layer0_N16_wire = {M0[18], M0[16], M0[9], M0[6], M0[3], M0[1]};
layer0_N16 layer0_N16_inst (.M0(layer0_N16_wire), .M1(M1[16:16]));

wire [5:0] layer0_N17_wire = {M0[27], M0[14], M0[6], M0[2], M0[1], M0[0]};
layer0_N17 layer0_N17_inst (.M0(layer0_N17_wire), .M1(M1[17:17]));

wire [5:0] layer0_N18_wire = {M0[25], M0[19], M0[16], M0[12], M0[9], M0[5]};
layer0_N18 layer0_N18_inst (.M0(layer0_N18_wire), .M1(M1[18:18]));

wire [5:0] layer0_N19_wire = {M0[25], M0[21], M0[18], M0[13], M0[2], M0[0]};
layer0_N19 layer0_N19_inst (.M0(layer0_N19_wire), .M1(M1[19:19]));

wire [5:0] layer0_N20_wire = {M0[27], M0[16], M0[6], M0[4], M0[3], M0[2]};
layer0_N20 layer0_N20_inst (.M0(layer0_N20_wire), .M1(M1[20:20]));

wire [5:0] layer0_N21_wire = {M0[18], M0[16], M0[8], M0[3], M0[1], M0[0]};
layer0_N21 layer0_N21_inst (.M0(layer0_N21_wire), .M1(M1[21:21]));

wire [5:0] layer0_N22_wire = {M0[22], M0[20], M0[16], M0[12], M0[9], M0[5]};
layer0_N22 layer0_N22_inst (.M0(layer0_N22_wire), .M1(M1[22:22]));

wire [5:0] layer0_N23_wire = {M0[25], M0[21], M0[19], M0[11], M0[7], M0[0]};
layer0_N23 layer0_N23_inst (.M0(layer0_N23_wire), .M1(M1[23:23]));

wire [5:0] layer0_N24_wire = {M0[23], M0[18], M0[16], M0[14], M0[13], M0[5]};
layer0_N24 layer0_N24_inst (.M0(layer0_N24_wire), .M1(M1[24:24]));

endmodule