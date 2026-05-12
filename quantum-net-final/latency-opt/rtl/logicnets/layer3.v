module layer3 (input [14:0] M0, output [1:0] M1);

wire [6:0] layer3_N0_wire = {M0[13], M0[6], M0[5], M0[3], M0[2], M0[1], M0[0]};
layer3_N0 layer3_N0_inst (.M0(layer3_N0_wire), .M1(M1[1:0]));

endmodule