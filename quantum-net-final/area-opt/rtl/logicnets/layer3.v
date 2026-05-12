module layer3 (input [4:0] M0, output [0:0] M1);

wire [4:0] layer3_N0_wire = {M0[4], M0[3], M0[2], M0[1], M0[0]};
layer3_N0 layer3_N0_inst (.M0(layer3_N0_wire), .M1(M1[0:0]));

endmodule