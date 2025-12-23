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
    output logic signed [W+4-1:0] sum
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
    logic signed [W-1:0] reduce_layer08;
    assign reduce_layer08 = inp[8];
    logic signed [W-1:0] reduce_layer09;
    assign reduce_layer09 = inp[9];
    logic signed [W-1:0] reduce_layer010;
    assign reduce_layer010 = inp[10];
    logic signed [W-1:0] reduce_layer011;
    assign reduce_layer011 = inp[11];
    logic signed [W-1:0] reduce_layer012;
    assign reduce_layer012 = inp[12];
    logic signed [W-1:0] reduce_layer013;
    assign reduce_layer013 = inp[13];
    logic signed [W-1:0] reduce_layer014;
    assign reduce_layer014 = inp[14];
    logic signed [W-1:0] reduce_layer015;
    assign reduce_layer015 = inp[15];


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
    logic signed [16:0] w_reduce_layer14;
    assign w_reduce_layer14 = reduce_layer08 + reduce_layer09;
    logic signed [16:0] w_reduce_layer15;
    assign w_reduce_layer15 = reduce_layer010 + reduce_layer011;
    logic signed [16:0] w_reduce_layer16;
    assign w_reduce_layer16 = reduce_layer012 + reduce_layer013;
    logic signed [16:0] w_reduce_layer17;
    assign w_reduce_layer17 = reduce_layer014 + reduce_layer015;

// Registers
    logic signed [16:0] reduce_layer10;
    logic signed [16:0] reduce_layer11;
    logic signed [16:0] reduce_layer12;
    logic signed [16:0] reduce_layer13;
    logic signed [16:0] reduce_layer14;
    logic signed [16:0] reduce_layer15;
    logic signed [16:0] reduce_layer16;
    logic signed [16:0] reduce_layer17;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer10 <= '0;
    reduce_layer11 <= '0;
    reduce_layer12 <= '0;
    reduce_layer13 <= '0;
    reduce_layer14 <= '0;
    reduce_layer15 <= '0;
    reduce_layer16 <= '0;
    reduce_layer17 <= '0;
  end
  else begin
    reduce_layer10 <= w_reduce_layer10;
    reduce_layer11 <= w_reduce_layer11;
    reduce_layer12 <= w_reduce_layer12;
    reduce_layer13 <= w_reduce_layer13;
    reduce_layer14 <= w_reduce_layer14;
    reduce_layer15 <= w_reduce_layer15;
    reduce_layer16 <= w_reduce_layer16;
    reduce_layer17 <= w_reduce_layer17;
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
    logic signed [17:0] w_reduce_layer22;
    assign w_reduce_layer22 = reduce_layer14 + reduce_layer15;
    logic signed [17:0] w_reduce_layer23;
    assign w_reduce_layer23 = reduce_layer16 + reduce_layer17;

// Registers
    logic signed [17:0] reduce_layer20;
    logic signed [17:0] reduce_layer21;
    logic signed [17:0] reduce_layer22;
    logic signed [17:0] reduce_layer23;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer20 <= '0;
    reduce_layer21 <= '0;
    reduce_layer22 <= '0;
    reduce_layer23 <= '0;
  end
  else begin
    reduce_layer20 <= w_reduce_layer20;
    reduce_layer21 <= w_reduce_layer21;
    reduce_layer22 <= w_reduce_layer22;
    reduce_layer23 <= w_reduce_layer23;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 3
/////////////////////////////////////

// Adders
    logic signed [18:0] w_reduce_layer30;
    assign w_reduce_layer30 = reduce_layer20 + reduce_layer21;
    logic signed [18:0] w_reduce_layer31;
    assign w_reduce_layer31 = reduce_layer22 + reduce_layer23;

// Registers
    logic signed [18:0] reduce_layer30;
    logic signed [18:0] reduce_layer31;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer30 <= '0;
    reduce_layer31 <= '0;
  end
  else begin
    reduce_layer30 <= w_reduce_layer30;
    reduce_layer31 <= w_reduce_layer31;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 4
/////////////////////////////////////

// Adders
    logic signed [19:0] w_reduce_layer40;
    assign w_reduce_layer40 = reduce_layer30 + reduce_layer31;

// Registers
    logic signed [19:0] reduce_layer40;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer40 <= '0;
  end
  else begin
    reduce_layer40 <= w_reduce_layer40;
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
    logic vld_layer4;

always_ff @(posedge clk) begin
  if (rst) begin
    vld_layer1 <= 0;
    vld_layer2 <= 0;
    vld_layer3 <= 0;
    vld_layer4 <= 0;
  end else begin
    vld_layer1 <= vld_layer0;
    vld_layer2 <= vld_layer1;
    vld_layer3 <= vld_layer2;
    vld_layer4 <= vld_layer3;
  end
end

assign sum      = reduce_layer40;
assign outp_vld = vld_layer4;

endmodule

