module layer1 (input [24:0] M0, output [4:0] M1);

wire [5:0] layer1_N0_wire = {M0[16], M0[13], M0[12], M0[5], M0[3], M0[1]};
layer1_N0 layer1_N0_inst (.M0(layer1_N0_wire), .M1(M1[0:0]));

wire [5:0] layer1_N1_wire = {M0[24], M0[17], M0[15], M0[13], M0[6], M0[2]};
layer1_N1 layer1_N1_inst (.M0(layer1_N1_wire), .M1(M1[1:1]));

wire [5:0] layer1_N2_wire = {M0[24], M0[21], M0[20], M0[7], M0[6], M0[4]};
layer1_N2 layer1_N2_inst (.M0(layer1_N2_wire), .M1(M1[2:2]));

wire [5:0] layer1_N3_wire = {M0[19], M0[15], M0[14], M0[13], M0[12], M0[6]};
layer1_N3 layer1_N3_inst (.M0(layer1_N3_wire), .M1(M1[3:3]));

wire [5:0] layer1_N4_wire = {M0[21], M0[13], M0[7], M0[5], M0[3], M0[2]};
layer1_N4 layer1_N4_inst (.M0(layer1_N4_wire), .M1(M1[4:4]));

endmodule