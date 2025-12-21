module layer1 (input [129:0] M0, output [39:0] M1);

wire [5:0] layer1_N0_wire = {M0[127], M0[118], M0[93], M0[49], M0[45], M0[23]};
layer1_N0 layer1_N0_inst (.M0(layer1_N0_wire), .M1(M1[0:0]));

wire [5:0] layer1_N1_wire = {M0[122], M0[93], M0[67], M0[29], M0[25], M0[15]};
layer1_N1 layer1_N1_inst (.M0(layer1_N1_wire), .M1(M1[1:1]));

wire [5:0] layer1_N2_wire = {M0[124], M0[103], M0[81], M0[55], M0[13], M0[5]};
layer1_N2 layer1_N2_inst (.M0(layer1_N2_wire), .M1(M1[2:2]));

wire [5:0] layer1_N3_wire = {M0[90], M0[71], M0[70], M0[49], M0[35], M0[9]};
layer1_N3 layer1_N3_inst (.M0(layer1_N3_wire), .M1(M1[3:3]));

wire [5:0] layer1_N4_wire = {M0[123], M0[81], M0[58], M0[43], M0[32], M0[7]};
layer1_N4 layer1_N4_inst (.M0(layer1_N4_wire), .M1(M1[4:4]));

wire [5:0] layer1_N5_wire = {M0[116], M0[115], M0[98], M0[58], M0[5], M0[1]};
layer1_N5 layer1_N5_inst (.M0(layer1_N5_wire), .M1(M1[5:5]));

wire [5:0] layer1_N6_wire = {M0[122], M0[119], M0[102], M0[95], M0[45], M0[9]};
layer1_N6 layer1_N6_inst (.M0(layer1_N6_wire), .M1(M1[6:6]));

wire [5:0] layer1_N7_wire = {M0[128], M0[86], M0[73], M0[72], M0[47], M0[11]};
layer1_N7 layer1_N7_inst (.M0(layer1_N7_wire), .M1(M1[7:7]));

wire [5:0] layer1_N8_wire = {M0[106], M0[105], M0[100], M0[44], M0[30], M0[10]};
layer1_N8 layer1_N8_inst (.M0(layer1_N8_wire), .M1(M1[8:8]));

wire [5:0] layer1_N9_wire = {M0[129], M0[95], M0[73], M0[61], M0[23], M0[15]};
layer1_N9 layer1_N9_inst (.M0(layer1_N9_wire), .M1(M1[9:9]));

wire [5:0] layer1_N10_wire = {M0[86], M0[80], M0[59], M0[46], M0[26], M0[4]};
layer1_N10 layer1_N10_inst (.M0(layer1_N10_wire), .M1(M1[10:10]));

wire [5:0] layer1_N11_wire = {M0[122], M0[101], M0[92], M0[64], M0[15], M0[6]};
layer1_N11 layer1_N11_inst (.M0(layer1_N11_wire), .M1(M1[11:11]));

wire [5:0] layer1_N12_wire = {M0[117], M0[112], M0[77], M0[45], M0[29], M0[27]};
layer1_N12 layer1_N12_inst (.M0(layer1_N12_wire), .M1(M1[12:12]));

wire [5:0] layer1_N13_wire = {M0[96], M0[66], M0[37], M0[29], M0[22], M0[16]};
layer1_N13 layer1_N13_inst (.M0(layer1_N13_wire), .M1(M1[13:13]));

wire [5:0] layer1_N14_wire = {M0[84], M0[71], M0[27], M0[16], M0[9], M0[4]};
layer1_N14 layer1_N14_inst (.M0(layer1_N14_wire), .M1(M1[14:14]));

wire [5:0] layer1_N15_wire = {M0[126], M0[92], M0[91], M0[55], M0[38], M0[37]};
layer1_N15 layer1_N15_inst (.M0(layer1_N15_wire), .M1(M1[15:15]));

wire [5:0] layer1_N16_wire = {M0[91], M0[74], M0[54], M0[41], M0[40], M0[9]};
layer1_N16 layer1_N16_inst (.M0(layer1_N16_wire), .M1(M1[16:16]));

wire [5:0] layer1_N17_wire = {M0[118], M0[99], M0[86], M0[80], M0[52], M0[51]};
layer1_N17 layer1_N17_inst (.M0(layer1_N17_wire), .M1(M1[17:17]));

wire [5:0] layer1_N18_wire = {M0[126], M0[115], M0[74], M0[34], M0[12], M0[0]};
layer1_N18 layer1_N18_inst (.M0(layer1_N18_wire), .M1(M1[18:18]));

wire [5:0] layer1_N19_wire = {M0[112], M0[98], M0[80], M0[51], M0[28], M0[11]};
layer1_N19 layer1_N19_inst (.M0(layer1_N19_wire), .M1(M1[19:19]));

wire [5:0] layer1_N20_wire = {M0[84], M0[64], M0[60], M0[24], M0[10], M0[5]};
layer1_N20 layer1_N20_inst (.M0(layer1_N20_wire), .M1(M1[20:20]));

wire [5:0] layer1_N21_wire = {M0[101], M0[65], M0[40], M0[20], M0[10], M0[1]};
layer1_N21 layer1_N21_inst (.M0(layer1_N21_wire), .M1(M1[21:21]));

wire [5:0] layer1_N22_wire = {M0[124], M0[118], M0[91], M0[85], M0[75], M0[21]};
layer1_N22 layer1_N22_inst (.M0(layer1_N22_wire), .M1(M1[22:22]));

wire [5:0] layer1_N23_wire = {M0[115], M0[101], M0[93], M0[90], M0[59], M0[3]};
layer1_N23 layer1_N23_inst (.M0(layer1_N23_wire), .M1(M1[23:23]));

wire [5:0] layer1_N24_wire = {M0[112], M0[67], M0[42], M0[36], M0[35], M0[17]};
layer1_N24 layer1_N24_inst (.M0(layer1_N24_wire), .M1(M1[24:24]));

wire [5:0] layer1_N25_wire = {M0[62], M0[58], M0[53], M0[35], M0[18], M0[17]};
layer1_N25 layer1_N25_inst (.M0(layer1_N25_wire), .M1(M1[25:25]));

wire [5:0] layer1_N26_wire = {M0[105], M0[91], M0[85], M0[67], M0[32], M0[22]};
layer1_N26 layer1_N26_inst (.M0(layer1_N26_wire), .M1(M1[26:26]));

wire [5:0] layer1_N27_wire = {M0[110], M0[53], M0[46], M0[26], M0[15], M0[10]};
layer1_N27 layer1_N27_inst (.M0(layer1_N27_wire), .M1(M1[27:27]));

wire [5:0] layer1_N28_wire = {M0[100], M0[92], M0[81], M0[70], M0[4], M0[0]};
layer1_N28 layer1_N28_inst (.M0(layer1_N28_wire), .M1(M1[28:28]));

wire [5:0] layer1_N29_wire = {M0[98], M0[85], M0[50], M0[43], M0[7], M0[3]};
layer1_N29 layer1_N29_inst (.M0(layer1_N29_wire), .M1(M1[29:29]));

wire [5:0] layer1_N30_wire = {M0[117], M0[104], M0[80], M0[49], M0[18], M0[5]};
layer1_N30 layer1_N30_inst (.M0(layer1_N30_wire), .M1(M1[30:30]));

wire [5:0] layer1_N31_wire = {M0[124], M0[104], M0[102], M0[63], M0[36], M0[11]};
layer1_N31 layer1_N31_inst (.M0(layer1_N31_wire), .M1(M1[31:31]));

wire [5:0] layer1_N32_wire = {M0[82], M0[75], M0[67], M0[64], M0[39], M0[26]};
layer1_N32 layer1_N32_inst (.M0(layer1_N32_wire), .M1(M1[32:32]));

wire [5:0] layer1_N33_wire = {M0[110], M0[105], M0[96], M0[85], M0[65], M0[32]};
layer1_N33 layer1_N33_inst (.M0(layer1_N33_wire), .M1(M1[33:33]));

wire [5:0] layer1_N34_wire = {M0[117], M0[104], M0[88], M0[51], M0[19], M0[2]};
layer1_N34 layer1_N34_inst (.M0(layer1_N34_wire), .M1(M1[34:34]));

wire [5:0] layer1_N35_wire = {M0[125], M0[116], M0[68], M0[58], M0[55], M0[30]};
layer1_N35 layer1_N35_inst (.M0(layer1_N35_wire), .M1(M1[35:35]));

wire [5:0] layer1_N36_wire = {M0[129], M0[117], M0[113], M0[108], M0[66], M0[41]};
layer1_N36 layer1_N36_inst (.M0(layer1_N36_wire), .M1(M1[36:36]));

wire [5:0] layer1_N37_wire = {M0[100], M0[98], M0[25], M0[23], M0[21], M0[8]};
layer1_N37 layer1_N37_inst (.M0(layer1_N37_wire), .M1(M1[37:37]));

wire [5:0] layer1_N38_wire = {M0[122], M0[116], M0[85], M0[18], M0[10], M0[4]};
layer1_N38 layer1_N38_inst (.M0(layer1_N38_wire), .M1(M1[38:38]));

wire [5:0] layer1_N39_wire = {M0[114], M0[105], M0[76], M0[63], M0[15], M0[3]};
layer1_N39 layer1_N39_inst (.M0(layer1_N39_wire), .M1(M1[39:39]));

endmodule