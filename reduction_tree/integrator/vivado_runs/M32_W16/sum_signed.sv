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
    output logic signed [W+5-1:0] sum
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
    logic signed [W-1:0] reduce_layer016;
    assign reduce_layer016 = inp[16];
    logic signed [W-1:0] reduce_layer017;
    assign reduce_layer017 = inp[17];
    logic signed [W-1:0] reduce_layer018;
    assign reduce_layer018 = inp[18];
    logic signed [W-1:0] reduce_layer019;
    assign reduce_layer019 = inp[19];
    logic signed [W-1:0] reduce_layer020;
    assign reduce_layer020 = inp[20];
    logic signed [W-1:0] reduce_layer021;
    assign reduce_layer021 = inp[21];
    logic signed [W-1:0] reduce_layer022;
    assign reduce_layer022 = inp[22];
    logic signed [W-1:0] reduce_layer023;
    assign reduce_layer023 = inp[23];
    logic signed [W-1:0] reduce_layer024;
    assign reduce_layer024 = inp[24];
    logic signed [W-1:0] reduce_layer025;
    assign reduce_layer025 = inp[25];
    logic signed [W-1:0] reduce_layer026;
    assign reduce_layer026 = inp[26];
    logic signed [W-1:0] reduce_layer027;
    assign reduce_layer027 = inp[27];
    logic signed [W-1:0] reduce_layer028;
    assign reduce_layer028 = inp[28];
    logic signed [W-1:0] reduce_layer029;
    assign reduce_layer029 = inp[29];
    logic signed [W-1:0] reduce_layer030;
    assign reduce_layer030 = inp[30];
    logic signed [W-1:0] reduce_layer031;
    assign reduce_layer031 = inp[31];


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
    logic signed [16:0] w_reduce_layer18;
    assign w_reduce_layer18 = reduce_layer016 + reduce_layer017;
    logic signed [16:0] w_reduce_layer19;
    assign w_reduce_layer19 = reduce_layer018 + reduce_layer019;
    logic signed [16:0] w_reduce_layer110;
    assign w_reduce_layer110 = reduce_layer020 + reduce_layer021;
    logic signed [16:0] w_reduce_layer111;
    assign w_reduce_layer111 = reduce_layer022 + reduce_layer023;
    logic signed [16:0] w_reduce_layer112;
    assign w_reduce_layer112 = reduce_layer024 + reduce_layer025;
    logic signed [16:0] w_reduce_layer113;
    assign w_reduce_layer113 = reduce_layer026 + reduce_layer027;
    logic signed [16:0] w_reduce_layer114;
    assign w_reduce_layer114 = reduce_layer028 + reduce_layer029;
    logic signed [16:0] w_reduce_layer115;
    assign w_reduce_layer115 = reduce_layer030 + reduce_layer031;

// Registers
    logic signed [16:0] reduce_layer10;
    logic signed [16:0] reduce_layer11;
    logic signed [16:0] reduce_layer12;
    logic signed [16:0] reduce_layer13;
    logic signed [16:0] reduce_layer14;
    logic signed [16:0] reduce_layer15;
    logic signed [16:0] reduce_layer16;
    logic signed [16:0] reduce_layer17;
    logic signed [16:0] reduce_layer18;
    logic signed [16:0] reduce_layer19;
    logic signed [16:0] reduce_layer110;
    logic signed [16:0] reduce_layer111;
    logic signed [16:0] reduce_layer112;
    logic signed [16:0] reduce_layer113;
    logic signed [16:0] reduce_layer114;
    logic signed [16:0] reduce_layer115;

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
    reduce_layer18 <= '0;
    reduce_layer19 <= '0;
    reduce_layer110 <= '0;
    reduce_layer111 <= '0;
    reduce_layer112 <= '0;
    reduce_layer113 <= '0;
    reduce_layer114 <= '0;
    reduce_layer115 <= '0;
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
    reduce_layer18 <= w_reduce_layer18;
    reduce_layer19 <= w_reduce_layer19;
    reduce_layer110 <= w_reduce_layer110;
    reduce_layer111 <= w_reduce_layer111;
    reduce_layer112 <= w_reduce_layer112;
    reduce_layer113 <= w_reduce_layer113;
    reduce_layer114 <= w_reduce_layer114;
    reduce_layer115 <= w_reduce_layer115;
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
    logic signed [17:0] w_reduce_layer24;
    assign w_reduce_layer24 = reduce_layer18 + reduce_layer19;
    logic signed [17:0] w_reduce_layer25;
    assign w_reduce_layer25 = reduce_layer110 + reduce_layer111;
    logic signed [17:0] w_reduce_layer26;
    assign w_reduce_layer26 = reduce_layer112 + reduce_layer113;
    logic signed [17:0] w_reduce_layer27;
    assign w_reduce_layer27 = reduce_layer114 + reduce_layer115;

// Registers
    logic signed [17:0] reduce_layer20;
    logic signed [17:0] reduce_layer21;
    logic signed [17:0] reduce_layer22;
    logic signed [17:0] reduce_layer23;
    logic signed [17:0] reduce_layer24;
    logic signed [17:0] reduce_layer25;
    logic signed [17:0] reduce_layer26;
    logic signed [17:0] reduce_layer27;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer20 <= '0;
    reduce_layer21 <= '0;
    reduce_layer22 <= '0;
    reduce_layer23 <= '0;
    reduce_layer24 <= '0;
    reduce_layer25 <= '0;
    reduce_layer26 <= '0;
    reduce_layer27 <= '0;
  end
  else begin
    reduce_layer20 <= w_reduce_layer20;
    reduce_layer21 <= w_reduce_layer21;
    reduce_layer22 <= w_reduce_layer22;
    reduce_layer23 <= w_reduce_layer23;
    reduce_layer24 <= w_reduce_layer24;
    reduce_layer25 <= w_reduce_layer25;
    reduce_layer26 <= w_reduce_layer26;
    reduce_layer27 <= w_reduce_layer27;
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
    logic signed [18:0] w_reduce_layer32;
    assign w_reduce_layer32 = reduce_layer24 + reduce_layer25;
    logic signed [18:0] w_reduce_layer33;
    assign w_reduce_layer33 = reduce_layer26 + reduce_layer27;

// Registers
    logic signed [18:0] reduce_layer30;
    logic signed [18:0] reduce_layer31;
    logic signed [18:0] reduce_layer32;
    logic signed [18:0] reduce_layer33;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer30 <= '0;
    reduce_layer31 <= '0;
    reduce_layer32 <= '0;
    reduce_layer33 <= '0;
  end
  else begin
    reduce_layer30 <= w_reduce_layer30;
    reduce_layer31 <= w_reduce_layer31;
    reduce_layer32 <= w_reduce_layer32;
    reduce_layer33 <= w_reduce_layer33;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 4
/////////////////////////////////////

// Adders
    logic signed [19:0] w_reduce_layer40;
    assign w_reduce_layer40 = reduce_layer30 + reduce_layer31;
    logic signed [19:0] w_reduce_layer41;
    assign w_reduce_layer41 = reduce_layer32 + reduce_layer33;

// Registers
    logic signed [19:0] reduce_layer40;
    logic signed [19:0] reduce_layer41;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer40 <= '0;
    reduce_layer41 <= '0;
  end
  else begin
    reduce_layer40 <= w_reduce_layer40;
    reduce_layer41 <= w_reduce_layer41;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 5
/////////////////////////////////////

// Adders
    logic signed [20:0] w_reduce_layer50;
    assign w_reduce_layer50 = reduce_layer40 + reduce_layer41;

// Registers
    logic signed [20:0] reduce_layer50;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer50 <= '0;
  end
  else begin
    reduce_layer50 <= w_reduce_layer50;
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
    logic vld_layer5;

always_ff @(posedge clk) begin
  if (rst) begin
    vld_layer1 <= 0;
    vld_layer2 <= 0;
    vld_layer3 <= 0;
    vld_layer4 <= 0;
    vld_layer5 <= 0;
  end else begin
    vld_layer1 <= vld_layer0;
    vld_layer2 <= vld_layer1;
    vld_layer3 <= vld_layer2;
    vld_layer4 <= vld_layer3;
    vld_layer5 <= vld_layer4;
  end
end

assign sum      = reduce_layer50;
assign outp_vld = vld_layer5;

endmodule

