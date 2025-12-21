module logicnet (input [47:0] M0, input clk, input rst, output[1:0] M4);
wire [47:0] M0w;
myreg #(.DataWidth(48)) layer0_reg (.data_in(M0), .clk(clk), .rst(rst), .data_out(M0w));
wire [129:0] M1;
layer0 layer0_inst (.M0(M0w), .M1(M1));
wire [129:0] M1w;
myreg #(.DataWidth(130)) layer1_reg (.data_in(M1), .clk(clk), .rst(rst), .data_out(M1w));
wire [39:0] M2;
layer1 layer1_inst (.M0(M1w), .M1(M2));
wire [39:0] M2w;
myreg #(.DataWidth(40)) layer2_reg (.data_in(M2), .clk(clk), .rst(rst), .data_out(M2w));
wire [9:0] M3;
layer2 layer2_inst (.M0(M2w), .M1(M3));
wire [9:0] M3w;
myreg #(.DataWidth(10)) layer3_reg (.data_in(M3), .clk(clk), .rst(rst), .data_out(M3w));
layer3 layer3_inst (.M0(M3w), .M1(M4));

endmodule
