module layer1 (input [289:0] M0, output [79:0] M1);

wire [11:0] layer1_N0_wire = {M0[225], M0[224], M0[219], M0[218], M0[217], M0[216], M0[177], M0[176], M0[101], M0[100], M0[47], M0[46]};
layer1_N0 layer1_N0_inst (.M0(layer1_N0_wire), .M1(M1[1:0]));

wire [11:0] layer1_N1_wire = {M0[287], M0[286], M0[267], M0[266], M0[243], M0[242], M0[199], M0[198], M0[115], M0[114], M0[41], M0[40]};
layer1_N1 layer1_N1_inst (.M0(layer1_N1_wire), .M1(M1[3:2]));

wire [11:0] layer1_N2_wire = {M0[283], M0[282], M0[279], M0[278], M0[237], M0[236], M0[207], M0[206], M0[163], M0[162], M0[43], M0[42]};
layer1_N2 layer1_N2_inst (.M0(layer1_N2_wire), .M1(M1[5:4]));

wire [11:0] layer1_N3_wire = {M0[283], M0[282], M0[255], M0[254], M0[141], M0[140], M0[123], M0[122], M0[79], M0[78], M0[7], M0[6]};
layer1_N3 layer1_N3_inst (.M0(layer1_N3_wire), .M1(M1[7:6]));

wire [11:0] layer1_N4_wire = {M0[247], M0[246], M0[199], M0[198], M0[193], M0[192], M0[143], M0[142], M0[31], M0[30], M0[3], M0[2]};
layer1_N4 layer1_N4_inst (.M0(layer1_N4_wire), .M1(M1[9:8]));

wire [11:0] layer1_N5_wire = {M0[237], M0[236], M0[209], M0[208], M0[195], M0[194], M0[51], M0[50], M0[45], M0[44], M0[31], M0[30]};
layer1_N5 layer1_N5_inst (.M0(layer1_N5_wire), .M1(M1[11:10]));

wire [11:0] layer1_N6_wire = {M0[265], M0[264], M0[235], M0[234], M0[191], M0[190], M0[123], M0[122], M0[121], M0[120], M0[17], M0[16]};
layer1_N6 layer1_N6_inst (.M0(layer1_N6_wire), .M1(M1[13:12]));

wire [11:0] layer1_N7_wire = {M0[201], M0[200], M0[179], M0[178], M0[115], M0[114], M0[97], M0[96], M0[45], M0[44], M0[21], M0[20]};
layer1_N7 layer1_N7_inst (.M0(layer1_N7_wire), .M1(M1[15:14]));

wire [11:0] layer1_N8_wire = {M0[263], M0[262], M0[245], M0[244], M0[215], M0[214], M0[199], M0[198], M0[99], M0[98], M0[85], M0[84]};
layer1_N8 layer1_N8_inst (.M0(layer1_N8_wire), .M1(M1[17:16]));

wire [11:0] layer1_N9_wire = {M0[259], M0[258], M0[249], M0[248], M0[183], M0[182], M0[139], M0[138], M0[91], M0[90], M0[7], M0[6]};
layer1_N9 layer1_N9_inst (.M0(layer1_N9_wire), .M1(M1[19:18]));

wire [11:0] layer1_N10_wire = {M0[277], M0[276], M0[275], M0[274], M0[235], M0[234], M0[113], M0[112], M0[87], M0[86], M0[45], M0[44]};
layer1_N10 layer1_N10_inst (.M0(layer1_N10_wire), .M1(M1[21:20]));

wire [11:0] layer1_N11_wire = {M0[263], M0[262], M0[251], M0[250], M0[237], M0[236], M0[161], M0[160], M0[107], M0[106], M0[27], M0[26]};
layer1_N11 layer1_N11_inst (.M0(layer1_N11_wire), .M1(M1[23:22]));

wire [11:0] layer1_N12_wire = {M0[279], M0[278], M0[229], M0[228], M0[227], M0[226], M0[97], M0[96], M0[89], M0[88], M0[73], M0[72]};
layer1_N12 layer1_N12_inst (.M0(layer1_N12_wire), .M1(M1[25:24]));

wire [11:0] layer1_N13_wire = {M0[225], M0[224], M0[149], M0[148], M0[121], M0[120], M0[47], M0[46], M0[27], M0[26], M0[25], M0[24]};
layer1_N13 layer1_N13_inst (.M0(layer1_N13_wire), .M1(M1[27:26]));

wire [11:0] layer1_N14_wire = {M0[225], M0[224], M0[219], M0[218], M0[137], M0[136], M0[87], M0[86], M0[69], M0[68], M0[67], M0[66]};
layer1_N14 layer1_N14_inst (.M0(layer1_N14_wire), .M1(M1[29:28]));

wire [11:0] layer1_N15_wire = {M0[285], M0[284], M0[175], M0[174], M0[93], M0[92], M0[19], M0[18], M0[13], M0[12], M0[9], M0[8]};
layer1_N15 layer1_N15_inst (.M0(layer1_N15_wire), .M1(M1[31:30]));

wire [11:0] layer1_N16_wire = {M0[269], M0[268], M0[225], M0[224], M0[187], M0[186], M0[137], M0[136], M0[113], M0[112], M0[7], M0[6]};
layer1_N16 layer1_N16_inst (.M0(layer1_N16_wire), .M1(M1[33:32]));

wire [11:0] layer1_N17_wire = {M0[263], M0[262], M0[245], M0[244], M0[197], M0[196], M0[193], M0[192], M0[147], M0[146], M0[109], M0[108]};
layer1_N17 layer1_N17_inst (.M0(layer1_N17_wire), .M1(M1[35:34]));

wire [11:0] layer1_N18_wire = {M0[279], M0[278], M0[249], M0[248], M0[235], M0[234], M0[225], M0[224], M0[107], M0[106], M0[91], M0[90]};
layer1_N18 layer1_N18_inst (.M0(layer1_N18_wire), .M1(M1[37:36]));

wire [11:0] layer1_N19_wire = {M0[273], M0[272], M0[229], M0[228], M0[121], M0[120], M0[111], M0[110], M0[95], M0[94], M0[75], M0[74]};
layer1_N19 layer1_N19_inst (.M0(layer1_N19_wire), .M1(M1[39:38]));

wire [11:0] layer1_N20_wire = {M0[249], M0[248], M0[229], M0[228], M0[125], M0[124], M0[65], M0[64], M0[61], M0[60], M0[41], M0[40]};
layer1_N20 layer1_N20_inst (.M0(layer1_N20_wire), .M1(M1[41:40]));

wire [11:0] layer1_N21_wire = {M0[273], M0[272], M0[255], M0[254], M0[229], M0[228], M0[139], M0[138], M0[127], M0[126], M0[5], M0[4]};
layer1_N21 layer1_N21_inst (.M0(layer1_N21_wire), .M1(M1[43:42]));

wire [11:0] layer1_N22_wire = {M0[257], M0[256], M0[251], M0[250], M0[237], M0[236], M0[197], M0[196], M0[181], M0[180], M0[127], M0[126]};
layer1_N22 layer1_N22_inst (.M0(layer1_N22_wire), .M1(M1[45:44]));

wire [11:0] layer1_N23_wire = {M0[277], M0[276], M0[177], M0[176], M0[127], M0[126], M0[75], M0[74], M0[69], M0[68], M0[39], M0[38]};
layer1_N23 layer1_N23_inst (.M0(layer1_N23_wire), .M1(M1[47:46]));

wire [11:0] layer1_N24_wire = {M0[285], M0[284], M0[185], M0[184], M0[151], M0[150], M0[87], M0[86], M0[43], M0[42], M0[25], M0[24]};
layer1_N24 layer1_N24_inst (.M0(layer1_N24_wire), .M1(M1[49:48]));

wire [11:0] layer1_N25_wire = {M0[285], M0[284], M0[165], M0[164], M0[137], M0[136], M0[127], M0[126], M0[101], M0[100], M0[35], M0[34]};
layer1_N25 layer1_N25_inst (.M0(layer1_N25_wire), .M1(M1[51:50]));

wire [11:0] layer1_N26_wire = {M0[259], M0[258], M0[253], M0[252], M0[237], M0[236], M0[175], M0[174], M0[33], M0[32], M0[19], M0[18]};
layer1_N26 layer1_N26_inst (.M0(layer1_N26_wire), .M1(M1[53:52]));

wire [11:0] layer1_N27_wire = {M0[273], M0[272], M0[271], M0[270], M0[169], M0[168], M0[107], M0[106], M0[79], M0[78], M0[57], M0[56]};
layer1_N27 layer1_N27_inst (.M0(layer1_N27_wire), .M1(M1[55:54]));

wire [11:0] layer1_N28_wire = {M0[285], M0[284], M0[189], M0[188], M0[159], M0[158], M0[133], M0[132], M0[107], M0[106], M0[87], M0[86]};
layer1_N28 layer1_N28_inst (.M0(layer1_N28_wire), .M1(M1[57:56]));

wire [11:0] layer1_N29_wire = {M0[245], M0[244], M0[209], M0[208], M0[169], M0[168], M0[139], M0[138], M0[113], M0[112], M0[83], M0[82]};
layer1_N29 layer1_N29_inst (.M0(layer1_N29_wire), .M1(M1[59:58]));

wire [11:0] layer1_N30_wire = {M0[267], M0[266], M0[103], M0[102], M0[59], M0[58], M0[43], M0[42], M0[25], M0[24], M0[1], M0[0]};
layer1_N30 layer1_N30_inst (.M0(layer1_N30_wire), .M1(M1[61:60]));

wire [11:0] layer1_N31_wire = {M0[257], M0[256], M0[251], M0[250], M0[137], M0[136], M0[103], M0[102], M0[71], M0[70], M0[25], M0[24]};
layer1_N31 layer1_N31_inst (.M0(layer1_N31_wire), .M1(M1[63:62]));

wire [11:0] layer1_N32_wire = {M0[287], M0[286], M0[277], M0[276], M0[229], M0[228], M0[189], M0[188], M0[79], M0[78], M0[41], M0[40]};
layer1_N32 layer1_N32_inst (.M0(layer1_N32_wire), .M1(M1[65:64]));

wire [11:0] layer1_N33_wire = {M0[281], M0[280], M0[267], M0[266], M0[217], M0[216], M0[189], M0[188], M0[65], M0[64], M0[19], M0[18]};
layer1_N33 layer1_N33_inst (.M0(layer1_N33_wire), .M1(M1[67:66]));

wire [11:0] layer1_N34_wire = {M0[247], M0[246], M0[229], M0[228], M0[199], M0[198], M0[191], M0[190], M0[137], M0[136], M0[9], M0[8]};
layer1_N34 layer1_N34_inst (.M0(layer1_N34_wire), .M1(M1[69:68]));

wire [11:0] layer1_N35_wire = {M0[265], M0[264], M0[255], M0[254], M0[197], M0[196], M0[193], M0[192], M0[147], M0[146], M0[69], M0[68]};
layer1_N35 layer1_N35_inst (.M0(layer1_N35_wire), .M1(M1[71:70]));

wire [11:0] layer1_N36_wire = {M0[259], M0[258], M0[221], M0[220], M0[205], M0[204], M0[187], M0[186], M0[89], M0[88], M0[17], M0[16]};
layer1_N36 layer1_N36_inst (.M0(layer1_N36_wire), .M1(M1[73:72]));

wire [11:0] layer1_N37_wire = {M0[289], M0[288], M0[253], M0[252], M0[251], M0[250], M0[207], M0[206], M0[125], M0[124], M0[49], M0[48]};
layer1_N37 layer1_N37_inst (.M0(layer1_N37_wire), .M1(M1[75:74]));

wire [11:0] layer1_N38_wire = {M0[279], M0[278], M0[227], M0[226], M0[213], M0[212], M0[191], M0[190], M0[167], M0[166], M0[83], M0[82]};
layer1_N38 layer1_N38_inst (.M0(layer1_N38_wire), .M1(M1[77:76]));

wire [11:0] layer1_N39_wire = {M0[247], M0[246], M0[181], M0[180], M0[167], M0[166], M0[163], M0[162], M0[131], M0[130], M0[105], M0[104]};
layer1_N39 layer1_N39_inst (.M0(layer1_N39_wire), .M1(M1[79:78]));

endmodule