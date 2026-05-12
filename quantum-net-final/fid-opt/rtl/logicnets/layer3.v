module layer3 (input [29:0] M0, output [1:0] M1);

wire [15:0] layer3_N0_wire = {M0[25], M0[24], M0[23], M0[22], M0[19], M0[18], M0[11], M0[10], M0[9], M0[8], M0[5], M0[4], M0[3], M0[2], M0[1], M0[0]};
layer3_N0 layer3_N0_inst (.M0(layer3_N0_wire), .M1(M1[1:0]));

endmodule