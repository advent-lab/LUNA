////////////////////////////////////////////////////////////////////////////////
// THIS FILE WAS AUTOMATICALLY GENERATED FROM sum_signed.mako
// DO NOT EDIT
////////////////////////////////////////////////////////////////////////////////


module sum_signed #(
    parameter int M = 32,   // number of inputs
    parameter int W = 16    // input width
)(
    input  logic clk,
    input  logic rst,
    input  logic inp_vld,
    output logic outp_vld,

    input  logic signed [W-1:0] inp [M],
    output logic signed [W+3-1:0] sum
);

/////////////////////////////////////
// Inputs are layer 0
/////////////////////////////////////

    logic signed [W-1:0] reduce_layer00;
    assign reduce_layer00 = inp[0];
    logic signed [W-1:0] reduce_layer01;
    assign reduce_layer01 = inp[1];
    logic signed [W-1:0] reduce_layer02;
    assign reduce_layer02 = inp[2];
    logic signed [W-1:0] reduce_layer03;
    assign reduce_layer03 = inp[3];
    logic signed [W-1:0] reduce_layer04;
    assign reduce_layer04 = inp[4];
    logic signed [W-1:0] reduce_layer05;
    assign reduce_layer05 = inp[5];
    logic signed [W-1:0] reduce_layer06;
    assign reduce_layer06 = inp[6];
    logic signed [W-1:0] reduce_layer07;
    assign reduce_layer07 = inp[7];


/////////////////////////////////////
// Layer 1
/////////////////////////////////////

// Adders
    logic signed [16:0] w_reduce_layer10;
    assign w_reduce_layer10 = reduce_layer00 + reduce_layer01;
    logic signed [16:0] w_reduce_layer11;
    assign w_reduce_layer11 = reduce_layer02 + reduce_layer03;
    logic signed [16:0] w_reduce_layer12;
    assign w_reduce_layer12 = reduce_layer04 + reduce_layer05;
    logic signed [16:0] w_reduce_layer13;
    assign w_reduce_layer13 = reduce_layer06 + reduce_layer07;

// Registers
    logic signed [16:0] reduce_layer10;
    logic signed [16:0] reduce_layer11;
    logic signed [16:0] reduce_layer12;
    logic signed [16:0] reduce_layer13;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer10 <= '0;
    reduce_layer11 <= '0;
    reduce_layer12 <= '0;
    reduce_layer13 <= '0;
  end
  else begin
    reduce_layer10 <= w_reduce_layer10;
    reduce_layer11 <= w_reduce_layer11;
    reduce_layer12 <= w_reduce_layer12;
    reduce_layer13 <= w_reduce_layer13;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 2
/////////////////////////////////////

// Adders
    logic signed [17:0] w_reduce_layer20;
    assign w_reduce_layer20 = reduce_layer10 + reduce_layer11;
    logic signed [17:0] w_reduce_layer21;
    assign w_reduce_layer21 = reduce_layer12 + reduce_layer13;

// Registers
    logic signed [17:0] reduce_layer20;
    logic signed [17:0] reduce_layer21;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer20 <= '0;
    reduce_layer21 <= '0;
  end
  else begin
    reduce_layer20 <= w_reduce_layer20;
    reduce_layer21 <= w_reduce_layer21;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 3
/////////////////////////////////////

// Adders
    logic signed [18:0] w_reduce_layer30;
    assign w_reduce_layer30 = reduce_layer20 + reduce_layer21;

// Registers
    logic signed [18:0] reduce_layer30;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer30 <= '0;
  end
  else begin
    reduce_layer30 <= w_reduce_layer30;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Valid pipeline
/////////////////////////////////////

logic vld_layer0;
assign vld_layer0 = inp_vld;

    logic vld_layer1;
    logic vld_layer2;
    logic vld_layer3;

always_ff @(posedge clk) begin
  if (rst) begin
    vld_layer1 <= 0;
    vld_layer2 <= 0;
    vld_layer3 <= 0;
  end else begin
    vld_layer1 <= vld_layer0;
    vld_layer2 <= vld_layer1;
    vld_layer3 <= vld_layer2;
  end
end

assign sum      = reduce_layer30;
assign outp_vld = vld_layer3;

endmodule

