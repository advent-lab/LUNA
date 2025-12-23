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
    output logic signed [W+6-1:0] sum
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
    logic signed [W-1:0] reduce_layer032;
    assign reduce_layer032 = inp[32];
    logic signed [W-1:0] reduce_layer033;
    assign reduce_layer033 = inp[33];
    logic signed [W-1:0] reduce_layer034;
    assign reduce_layer034 = inp[34];
    logic signed [W-1:0] reduce_layer035;
    assign reduce_layer035 = inp[35];
    logic signed [W-1:0] reduce_layer036;
    assign reduce_layer036 = inp[36];
    logic signed [W-1:0] reduce_layer037;
    assign reduce_layer037 = inp[37];
    logic signed [W-1:0] reduce_layer038;
    assign reduce_layer038 = inp[38];
    logic signed [W-1:0] reduce_layer039;
    assign reduce_layer039 = inp[39];
    logic signed [W-1:0] reduce_layer040;
    assign reduce_layer040 = inp[40];
    logic signed [W-1:0] reduce_layer041;
    assign reduce_layer041 = inp[41];
    logic signed [W-1:0] reduce_layer042;
    assign reduce_layer042 = inp[42];
    logic signed [W-1:0] reduce_layer043;
    assign reduce_layer043 = inp[43];
    logic signed [W-1:0] reduce_layer044;
    assign reduce_layer044 = inp[44];
    logic signed [W-1:0] reduce_layer045;
    assign reduce_layer045 = inp[45];
    logic signed [W-1:0] reduce_layer046;
    assign reduce_layer046 = inp[46];
    logic signed [W-1:0] reduce_layer047;
    assign reduce_layer047 = inp[47];
    logic signed [W-1:0] reduce_layer048;
    assign reduce_layer048 = inp[48];
    logic signed [W-1:0] reduce_layer049;
    assign reduce_layer049 = inp[49];
    logic signed [W-1:0] reduce_layer050;
    assign reduce_layer050 = inp[50];
    logic signed [W-1:0] reduce_layer051;
    assign reduce_layer051 = inp[51];
    logic signed [W-1:0] reduce_layer052;
    assign reduce_layer052 = inp[52];
    logic signed [W-1:0] reduce_layer053;
    assign reduce_layer053 = inp[53];
    logic signed [W-1:0] reduce_layer054;
    assign reduce_layer054 = inp[54];
    logic signed [W-1:0] reduce_layer055;
    assign reduce_layer055 = inp[55];
    logic signed [W-1:0] reduce_layer056;
    assign reduce_layer056 = inp[56];
    logic signed [W-1:0] reduce_layer057;
    assign reduce_layer057 = inp[57];
    logic signed [W-1:0] reduce_layer058;
    assign reduce_layer058 = inp[58];
    logic signed [W-1:0] reduce_layer059;
    assign reduce_layer059 = inp[59];
    logic signed [W-1:0] reduce_layer060;
    assign reduce_layer060 = inp[60];
    logic signed [W-1:0] reduce_layer061;
    assign reduce_layer061 = inp[61];
    logic signed [W-1:0] reduce_layer062;
    assign reduce_layer062 = inp[62];
    logic signed [W-1:0] reduce_layer063;
    assign reduce_layer063 = inp[63];


/////////////////////////////////////
// Layer 1
/////////////////////////////////////

// Adders
    logic signed [8:0] w_reduce_layer10;
    assign w_reduce_layer10 = reduce_layer00 + reduce_layer01;
    logic signed [8:0] w_reduce_layer11;
    assign w_reduce_layer11 = reduce_layer02 + reduce_layer03;
    logic signed [8:0] w_reduce_layer12;
    assign w_reduce_layer12 = reduce_layer04 + reduce_layer05;
    logic signed [8:0] w_reduce_layer13;
    assign w_reduce_layer13 = reduce_layer06 + reduce_layer07;
    logic signed [8:0] w_reduce_layer14;
    assign w_reduce_layer14 = reduce_layer08 + reduce_layer09;
    logic signed [8:0] w_reduce_layer15;
    assign w_reduce_layer15 = reduce_layer010 + reduce_layer011;
    logic signed [8:0] w_reduce_layer16;
    assign w_reduce_layer16 = reduce_layer012 + reduce_layer013;
    logic signed [8:0] w_reduce_layer17;
    assign w_reduce_layer17 = reduce_layer014 + reduce_layer015;
    logic signed [8:0] w_reduce_layer18;
    assign w_reduce_layer18 = reduce_layer016 + reduce_layer017;
    logic signed [8:0] w_reduce_layer19;
    assign w_reduce_layer19 = reduce_layer018 + reduce_layer019;
    logic signed [8:0] w_reduce_layer110;
    assign w_reduce_layer110 = reduce_layer020 + reduce_layer021;
    logic signed [8:0] w_reduce_layer111;
    assign w_reduce_layer111 = reduce_layer022 + reduce_layer023;
    logic signed [8:0] w_reduce_layer112;
    assign w_reduce_layer112 = reduce_layer024 + reduce_layer025;
    logic signed [8:0] w_reduce_layer113;
    assign w_reduce_layer113 = reduce_layer026 + reduce_layer027;
    logic signed [8:0] w_reduce_layer114;
    assign w_reduce_layer114 = reduce_layer028 + reduce_layer029;
    logic signed [8:0] w_reduce_layer115;
    assign w_reduce_layer115 = reduce_layer030 + reduce_layer031;
    logic signed [8:0] w_reduce_layer116;
    assign w_reduce_layer116 = reduce_layer032 + reduce_layer033;
    logic signed [8:0] w_reduce_layer117;
    assign w_reduce_layer117 = reduce_layer034 + reduce_layer035;
    logic signed [8:0] w_reduce_layer118;
    assign w_reduce_layer118 = reduce_layer036 + reduce_layer037;
    logic signed [8:0] w_reduce_layer119;
    assign w_reduce_layer119 = reduce_layer038 + reduce_layer039;
    logic signed [8:0] w_reduce_layer120;
    assign w_reduce_layer120 = reduce_layer040 + reduce_layer041;
    logic signed [8:0] w_reduce_layer121;
    assign w_reduce_layer121 = reduce_layer042 + reduce_layer043;
    logic signed [8:0] w_reduce_layer122;
    assign w_reduce_layer122 = reduce_layer044 + reduce_layer045;
    logic signed [8:0] w_reduce_layer123;
    assign w_reduce_layer123 = reduce_layer046 + reduce_layer047;
    logic signed [8:0] w_reduce_layer124;
    assign w_reduce_layer124 = reduce_layer048 + reduce_layer049;
    logic signed [8:0] w_reduce_layer125;
    assign w_reduce_layer125 = reduce_layer050 + reduce_layer051;
    logic signed [8:0] w_reduce_layer126;
    assign w_reduce_layer126 = reduce_layer052 + reduce_layer053;
    logic signed [8:0] w_reduce_layer127;
    assign w_reduce_layer127 = reduce_layer054 + reduce_layer055;
    logic signed [8:0] w_reduce_layer128;
    assign w_reduce_layer128 = reduce_layer056 + reduce_layer057;
    logic signed [8:0] w_reduce_layer129;
    assign w_reduce_layer129 = reduce_layer058 + reduce_layer059;
    logic signed [8:0] w_reduce_layer130;
    assign w_reduce_layer130 = reduce_layer060 + reduce_layer061;
    logic signed [8:0] w_reduce_layer131;
    assign w_reduce_layer131 = reduce_layer062 + reduce_layer063;

// Registers
    logic signed [8:0] reduce_layer10;
    logic signed [8:0] reduce_layer11;
    logic signed [8:0] reduce_layer12;
    logic signed [8:0] reduce_layer13;
    logic signed [8:0] reduce_layer14;
    logic signed [8:0] reduce_layer15;
    logic signed [8:0] reduce_layer16;
    logic signed [8:0] reduce_layer17;
    logic signed [8:0] reduce_layer18;
    logic signed [8:0] reduce_layer19;
    logic signed [8:0] reduce_layer110;
    logic signed [8:0] reduce_layer111;
    logic signed [8:0] reduce_layer112;
    logic signed [8:0] reduce_layer113;
    logic signed [8:0] reduce_layer114;
    logic signed [8:0] reduce_layer115;
    logic signed [8:0] reduce_layer116;
    logic signed [8:0] reduce_layer117;
    logic signed [8:0] reduce_layer118;
    logic signed [8:0] reduce_layer119;
    logic signed [8:0] reduce_layer120;
    logic signed [8:0] reduce_layer121;
    logic signed [8:0] reduce_layer122;
    logic signed [8:0] reduce_layer123;
    logic signed [8:0] reduce_layer124;
    logic signed [8:0] reduce_layer125;
    logic signed [8:0] reduce_layer126;
    logic signed [8:0] reduce_layer127;
    logic signed [8:0] reduce_layer128;
    logic signed [8:0] reduce_layer129;
    logic signed [8:0] reduce_layer130;
    logic signed [8:0] reduce_layer131;

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
    reduce_layer116 <= '0;
    reduce_layer117 <= '0;
    reduce_layer118 <= '0;
    reduce_layer119 <= '0;
    reduce_layer120 <= '0;
    reduce_layer121 <= '0;
    reduce_layer122 <= '0;
    reduce_layer123 <= '0;
    reduce_layer124 <= '0;
    reduce_layer125 <= '0;
    reduce_layer126 <= '0;
    reduce_layer127 <= '0;
    reduce_layer128 <= '0;
    reduce_layer129 <= '0;
    reduce_layer130 <= '0;
    reduce_layer131 <= '0;
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
    reduce_layer116 <= w_reduce_layer116;
    reduce_layer117 <= w_reduce_layer117;
    reduce_layer118 <= w_reduce_layer118;
    reduce_layer119 <= w_reduce_layer119;
    reduce_layer120 <= w_reduce_layer120;
    reduce_layer121 <= w_reduce_layer121;
    reduce_layer122 <= w_reduce_layer122;
    reduce_layer123 <= w_reduce_layer123;
    reduce_layer124 <= w_reduce_layer124;
    reduce_layer125 <= w_reduce_layer125;
    reduce_layer126 <= w_reduce_layer126;
    reduce_layer127 <= w_reduce_layer127;
    reduce_layer128 <= w_reduce_layer128;
    reduce_layer129 <= w_reduce_layer129;
    reduce_layer130 <= w_reduce_layer130;
    reduce_layer131 <= w_reduce_layer131;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 2
/////////////////////////////////////

// Adders
    logic signed [9:0] w_reduce_layer20;
    assign w_reduce_layer20 = reduce_layer10 + reduce_layer11;
    logic signed [9:0] w_reduce_layer21;
    assign w_reduce_layer21 = reduce_layer12 + reduce_layer13;
    logic signed [9:0] w_reduce_layer22;
    assign w_reduce_layer22 = reduce_layer14 + reduce_layer15;
    logic signed [9:0] w_reduce_layer23;
    assign w_reduce_layer23 = reduce_layer16 + reduce_layer17;
    logic signed [9:0] w_reduce_layer24;
    assign w_reduce_layer24 = reduce_layer18 + reduce_layer19;
    logic signed [9:0] w_reduce_layer25;
    assign w_reduce_layer25 = reduce_layer110 + reduce_layer111;
    logic signed [9:0] w_reduce_layer26;
    assign w_reduce_layer26 = reduce_layer112 + reduce_layer113;
    logic signed [9:0] w_reduce_layer27;
    assign w_reduce_layer27 = reduce_layer114 + reduce_layer115;
    logic signed [9:0] w_reduce_layer28;
    assign w_reduce_layer28 = reduce_layer116 + reduce_layer117;
    logic signed [9:0] w_reduce_layer29;
    assign w_reduce_layer29 = reduce_layer118 + reduce_layer119;
    logic signed [9:0] w_reduce_layer210;
    assign w_reduce_layer210 = reduce_layer120 + reduce_layer121;
    logic signed [9:0] w_reduce_layer211;
    assign w_reduce_layer211 = reduce_layer122 + reduce_layer123;
    logic signed [9:0] w_reduce_layer212;
    assign w_reduce_layer212 = reduce_layer124 + reduce_layer125;
    logic signed [9:0] w_reduce_layer213;
    assign w_reduce_layer213 = reduce_layer126 + reduce_layer127;
    logic signed [9:0] w_reduce_layer214;
    assign w_reduce_layer214 = reduce_layer128 + reduce_layer129;
    logic signed [9:0] w_reduce_layer215;
    assign w_reduce_layer215 = reduce_layer130 + reduce_layer131;

// Registers
    logic signed [9:0] reduce_layer20;
    logic signed [9:0] reduce_layer21;
    logic signed [9:0] reduce_layer22;
    logic signed [9:0] reduce_layer23;
    logic signed [9:0] reduce_layer24;
    logic signed [9:0] reduce_layer25;
    logic signed [9:0] reduce_layer26;
    logic signed [9:0] reduce_layer27;
    logic signed [9:0] reduce_layer28;
    logic signed [9:0] reduce_layer29;
    logic signed [9:0] reduce_layer210;
    logic signed [9:0] reduce_layer211;
    logic signed [9:0] reduce_layer212;
    logic signed [9:0] reduce_layer213;
    logic signed [9:0] reduce_layer214;
    logic signed [9:0] reduce_layer215;

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
    reduce_layer28 <= '0;
    reduce_layer29 <= '0;
    reduce_layer210 <= '0;
    reduce_layer211 <= '0;
    reduce_layer212 <= '0;
    reduce_layer213 <= '0;
    reduce_layer214 <= '0;
    reduce_layer215 <= '0;
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
    reduce_layer28 <= w_reduce_layer28;
    reduce_layer29 <= w_reduce_layer29;
    reduce_layer210 <= w_reduce_layer210;
    reduce_layer211 <= w_reduce_layer211;
    reduce_layer212 <= w_reduce_layer212;
    reduce_layer213 <= w_reduce_layer213;
    reduce_layer214 <= w_reduce_layer214;
    reduce_layer215 <= w_reduce_layer215;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 3
/////////////////////////////////////

// Adders
    logic signed [10:0] w_reduce_layer30;
    assign w_reduce_layer30 = reduce_layer20 + reduce_layer21;
    logic signed [10:0] w_reduce_layer31;
    assign w_reduce_layer31 = reduce_layer22 + reduce_layer23;
    logic signed [10:0] w_reduce_layer32;
    assign w_reduce_layer32 = reduce_layer24 + reduce_layer25;
    logic signed [10:0] w_reduce_layer33;
    assign w_reduce_layer33 = reduce_layer26 + reduce_layer27;
    logic signed [10:0] w_reduce_layer34;
    assign w_reduce_layer34 = reduce_layer28 + reduce_layer29;
    logic signed [10:0] w_reduce_layer35;
    assign w_reduce_layer35 = reduce_layer210 + reduce_layer211;
    logic signed [10:0] w_reduce_layer36;
    assign w_reduce_layer36 = reduce_layer212 + reduce_layer213;
    logic signed [10:0] w_reduce_layer37;
    assign w_reduce_layer37 = reduce_layer214 + reduce_layer215;

// Registers
    logic signed [10:0] reduce_layer30;
    logic signed [10:0] reduce_layer31;
    logic signed [10:0] reduce_layer32;
    logic signed [10:0] reduce_layer33;
    logic signed [10:0] reduce_layer34;
    logic signed [10:0] reduce_layer35;
    logic signed [10:0] reduce_layer36;
    logic signed [10:0] reduce_layer37;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer30 <= '0;
    reduce_layer31 <= '0;
    reduce_layer32 <= '0;
    reduce_layer33 <= '0;
    reduce_layer34 <= '0;
    reduce_layer35 <= '0;
    reduce_layer36 <= '0;
    reduce_layer37 <= '0;
  end
  else begin
    reduce_layer30 <= w_reduce_layer30;
    reduce_layer31 <= w_reduce_layer31;
    reduce_layer32 <= w_reduce_layer32;
    reduce_layer33 <= w_reduce_layer33;
    reduce_layer34 <= w_reduce_layer34;
    reduce_layer35 <= w_reduce_layer35;
    reduce_layer36 <= w_reduce_layer36;
    reduce_layer37 <= w_reduce_layer37;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 4
/////////////////////////////////////

// Adders
    logic signed [11:0] w_reduce_layer40;
    assign w_reduce_layer40 = reduce_layer30 + reduce_layer31;
    logic signed [11:0] w_reduce_layer41;
    assign w_reduce_layer41 = reduce_layer32 + reduce_layer33;
    logic signed [11:0] w_reduce_layer42;
    assign w_reduce_layer42 = reduce_layer34 + reduce_layer35;
    logic signed [11:0] w_reduce_layer43;
    assign w_reduce_layer43 = reduce_layer36 + reduce_layer37;

// Registers
    logic signed [11:0] reduce_layer40;
    logic signed [11:0] reduce_layer41;
    logic signed [11:0] reduce_layer42;
    logic signed [11:0] reduce_layer43;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer40 <= '0;
    reduce_layer41 <= '0;
    reduce_layer42 <= '0;
    reduce_layer43 <= '0;
  end
  else begin
    reduce_layer40 <= w_reduce_layer40;
    reduce_layer41 <= w_reduce_layer41;
    reduce_layer42 <= w_reduce_layer42;
    reduce_layer43 <= w_reduce_layer43;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 5
/////////////////////////////////////

// Adders
    logic signed [12:0] w_reduce_layer50;
    assign w_reduce_layer50 = reduce_layer40 + reduce_layer41;
    logic signed [12:0] w_reduce_layer51;
    assign w_reduce_layer51 = reduce_layer42 + reduce_layer43;

// Registers
    logic signed [12:0] reduce_layer50;
    logic signed [12:0] reduce_layer51;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer50 <= '0;
    reduce_layer51 <= '0;
  end
  else begin
    reduce_layer50 <= w_reduce_layer50;
    reduce_layer51 <= w_reduce_layer51;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 6
/////////////////////////////////////

// Adders
    logic signed [13:0] w_reduce_layer60;
    assign w_reduce_layer60 = reduce_layer50 + reduce_layer51;

// Registers
    logic signed [13:0] reduce_layer60;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer60 <= '0;
  end
  else begin
    reduce_layer60 <= w_reduce_layer60;
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
    logic vld_layer6;

always_ff @(posedge clk) begin
  if (rst) begin
    vld_layer1 <= 0;
    vld_layer2 <= 0;
    vld_layer3 <= 0;
    vld_layer4 <= 0;
    vld_layer5 <= 0;
    vld_layer6 <= 0;
  end else begin
    vld_layer1 <= vld_layer0;
    vld_layer2 <= vld_layer1;
    vld_layer3 <= vld_layer2;
    vld_layer4 <= vld_layer3;
    vld_layer5 <= vld_layer4;
    vld_layer6 <= vld_layer5;
  end
end

assign sum      = reduce_layer60;
assign outp_vld = vld_layer6;

endmodule

