////////////////////////////////////////////////////////////////////////////////
// THIS FILE WAS AUTOMATICALLY GENERATED FROM sum_signed.mako
// DO NOT EDIT
////////////////////////////////////////////////////////////////////////////////


module sum_signed #(
    parameter int M = 150,   // number of inputs
    parameter int W = 5   // input width
)(
    input  logic clk,
    input  logic rst,
    input  logic inp_vld,
    output logic outp_vld,

    input  logic signed [W-1:0] inp [M],
    output logic signed [W+8-1:0] sum
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
    logic signed [W-1:0] reduce_layer064;
    assign reduce_layer064 = inp[64];
    logic signed [W-1:0] reduce_layer065;
    assign reduce_layer065 = inp[65];
    logic signed [W-1:0] reduce_layer066;
    assign reduce_layer066 = inp[66];
    logic signed [W-1:0] reduce_layer067;
    assign reduce_layer067 = inp[67];
    logic signed [W-1:0] reduce_layer068;
    assign reduce_layer068 = inp[68];
    logic signed [W-1:0] reduce_layer069;
    assign reduce_layer069 = inp[69];
    logic signed [W-1:0] reduce_layer070;
    assign reduce_layer070 = inp[70];
    logic signed [W-1:0] reduce_layer071;
    assign reduce_layer071 = inp[71];
    logic signed [W-1:0] reduce_layer072;
    assign reduce_layer072 = inp[72];
    logic signed [W-1:0] reduce_layer073;
    assign reduce_layer073 = inp[73];
    logic signed [W-1:0] reduce_layer074;
    assign reduce_layer074 = inp[74];
    logic signed [W-1:0] reduce_layer075;
    assign reduce_layer075 = inp[75];
    logic signed [W-1:0] reduce_layer076;
    assign reduce_layer076 = inp[76];
    logic signed [W-1:0] reduce_layer077;
    assign reduce_layer077 = inp[77];
    logic signed [W-1:0] reduce_layer078;
    assign reduce_layer078 = inp[78];
    logic signed [W-1:0] reduce_layer079;
    assign reduce_layer079 = inp[79];
    logic signed [W-1:0] reduce_layer080;
    assign reduce_layer080 = inp[80];
    logic signed [W-1:0] reduce_layer081;
    assign reduce_layer081 = inp[81];
    logic signed [W-1:0] reduce_layer082;
    assign reduce_layer082 = inp[82];
    logic signed [W-1:0] reduce_layer083;
    assign reduce_layer083 = inp[83];
    logic signed [W-1:0] reduce_layer084;
    assign reduce_layer084 = inp[84];
    logic signed [W-1:0] reduce_layer085;
    assign reduce_layer085 = inp[85];
    logic signed [W-1:0] reduce_layer086;
    assign reduce_layer086 = inp[86];
    logic signed [W-1:0] reduce_layer087;
    assign reduce_layer087 = inp[87];
    logic signed [W-1:0] reduce_layer088;
    assign reduce_layer088 = inp[88];
    logic signed [W-1:0] reduce_layer089;
    assign reduce_layer089 = inp[89];
    logic signed [W-1:0] reduce_layer090;
    assign reduce_layer090 = inp[90];
    logic signed [W-1:0] reduce_layer091;
    assign reduce_layer091 = inp[91];
    logic signed [W-1:0] reduce_layer092;
    assign reduce_layer092 = inp[92];
    logic signed [W-1:0] reduce_layer093;
    assign reduce_layer093 = inp[93];
    logic signed [W-1:0] reduce_layer094;
    assign reduce_layer094 = inp[94];
    logic signed [W-1:0] reduce_layer095;
    assign reduce_layer095 = inp[95];
    logic signed [W-1:0] reduce_layer096;
    assign reduce_layer096 = inp[96];
    logic signed [W-1:0] reduce_layer097;
    assign reduce_layer097 = inp[97];
    logic signed [W-1:0] reduce_layer098;
    assign reduce_layer098 = inp[98];
    logic signed [W-1:0] reduce_layer099;
    assign reduce_layer099 = inp[99];
    logic signed [W-1:0] reduce_layer0100;
    assign reduce_layer0100 = inp[100];
    logic signed [W-1:0] reduce_layer0101;
    assign reduce_layer0101 = inp[101];
    logic signed [W-1:0] reduce_layer0102;
    assign reduce_layer0102 = inp[102];
    logic signed [W-1:0] reduce_layer0103;
    assign reduce_layer0103 = inp[103];
    logic signed [W-1:0] reduce_layer0104;
    assign reduce_layer0104 = inp[104];
    logic signed [W-1:0] reduce_layer0105;
    assign reduce_layer0105 = inp[105];
    logic signed [W-1:0] reduce_layer0106;
    assign reduce_layer0106 = inp[106];
    logic signed [W-1:0] reduce_layer0107;
    assign reduce_layer0107 = inp[107];
    logic signed [W-1:0] reduce_layer0108;
    assign reduce_layer0108 = inp[108];
    logic signed [W-1:0] reduce_layer0109;
    assign reduce_layer0109 = inp[109];
    logic signed [W-1:0] reduce_layer0110;
    assign reduce_layer0110 = inp[110];
    logic signed [W-1:0] reduce_layer0111;
    assign reduce_layer0111 = inp[111];
    logic signed [W-1:0] reduce_layer0112;
    assign reduce_layer0112 = inp[112];
    logic signed [W-1:0] reduce_layer0113;
    assign reduce_layer0113 = inp[113];
    logic signed [W-1:0] reduce_layer0114;
    assign reduce_layer0114 = inp[114];
    logic signed [W-1:0] reduce_layer0115;
    assign reduce_layer0115 = inp[115];
    logic signed [W-1:0] reduce_layer0116;
    assign reduce_layer0116 = inp[116];
    logic signed [W-1:0] reduce_layer0117;
    assign reduce_layer0117 = inp[117];
    logic signed [W-1:0] reduce_layer0118;
    assign reduce_layer0118 = inp[118];
    logic signed [W-1:0] reduce_layer0119;
    assign reduce_layer0119 = inp[119];
    logic signed [W-1:0] reduce_layer0120;
    assign reduce_layer0120 = inp[120];
    logic signed [W-1:0] reduce_layer0121;
    assign reduce_layer0121 = inp[121];
    logic signed [W-1:0] reduce_layer0122;
    assign reduce_layer0122 = inp[122];
    logic signed [W-1:0] reduce_layer0123;
    assign reduce_layer0123 = inp[123];
    logic signed [W-1:0] reduce_layer0124;
    assign reduce_layer0124 = inp[124];
    logic signed [W-1:0] reduce_layer0125;
    assign reduce_layer0125 = inp[125];
    logic signed [W-1:0] reduce_layer0126;
    assign reduce_layer0126 = inp[126];
    logic signed [W-1:0] reduce_layer0127;
    assign reduce_layer0127 = inp[127];
    logic signed [W-1:0] reduce_layer0128;
    assign reduce_layer0128 = inp[128];
    logic signed [W-1:0] reduce_layer0129;
    assign reduce_layer0129 = inp[129];
    logic signed [W-1:0] reduce_layer0130;
    assign reduce_layer0130 = inp[130];
    logic signed [W-1:0] reduce_layer0131;
    assign reduce_layer0131 = inp[131];
    logic signed [W-1:0] reduce_layer0132;
    assign reduce_layer0132 = inp[132];
    logic signed [W-1:0] reduce_layer0133;
    assign reduce_layer0133 = inp[133];
    logic signed [W-1:0] reduce_layer0134;
    assign reduce_layer0134 = inp[134];
    logic signed [W-1:0] reduce_layer0135;
    assign reduce_layer0135 = inp[135];
    logic signed [W-1:0] reduce_layer0136;
    assign reduce_layer0136 = inp[136];
    logic signed [W-1:0] reduce_layer0137;
    assign reduce_layer0137 = inp[137];
    logic signed [W-1:0] reduce_layer0138;
    assign reduce_layer0138 = inp[138];
    logic signed [W-1:0] reduce_layer0139;
    assign reduce_layer0139 = inp[139];
    logic signed [W-1:0] reduce_layer0140;
    assign reduce_layer0140 = inp[140];
    logic signed [W-1:0] reduce_layer0141;
    assign reduce_layer0141 = inp[141];
    logic signed [W-1:0] reduce_layer0142;
    assign reduce_layer0142 = inp[142];
    logic signed [W-1:0] reduce_layer0143;
    assign reduce_layer0143 = inp[143];
    logic signed [W-1:0] reduce_layer0144;
    assign reduce_layer0144 = inp[144];
    logic signed [W-1:0] reduce_layer0145;
    assign reduce_layer0145 = inp[145];
    logic signed [W-1:0] reduce_layer0146;
    assign reduce_layer0146 = inp[146];
    logic signed [W-1:0] reduce_layer0147;
    assign reduce_layer0147 = inp[147];
    logic signed [W-1:0] reduce_layer0148;
    assign reduce_layer0148 = inp[148];
    logic signed [W-1:0] reduce_layer0149;
    assign reduce_layer0149 = inp[149];


/////////////////////////////////////
// Layer 1
/////////////////////////////////////

// Adders
    logic signed [5:0] w_reduce_layer10;
    assign w_reduce_layer10 = reduce_layer00 + reduce_layer01;
    logic signed [5:0] w_reduce_layer11;
    assign w_reduce_layer11 = reduce_layer02 + reduce_layer03;
    logic signed [5:0] w_reduce_layer12;
    assign w_reduce_layer12 = reduce_layer04 + reduce_layer05;
    logic signed [5:0] w_reduce_layer13;
    assign w_reduce_layer13 = reduce_layer06 + reduce_layer07;
    logic signed [5:0] w_reduce_layer14;
    assign w_reduce_layer14 = reduce_layer08 + reduce_layer09;
    logic signed [5:0] w_reduce_layer15;
    assign w_reduce_layer15 = reduce_layer010 + reduce_layer011;
    logic signed [5:0] w_reduce_layer16;
    assign w_reduce_layer16 = reduce_layer012 + reduce_layer013;
    logic signed [5:0] w_reduce_layer17;
    assign w_reduce_layer17 = reduce_layer014 + reduce_layer015;
    logic signed [5:0] w_reduce_layer18;
    assign w_reduce_layer18 = reduce_layer016 + reduce_layer017;
    logic signed [5:0] w_reduce_layer19;
    assign w_reduce_layer19 = reduce_layer018 + reduce_layer019;
    logic signed [5:0] w_reduce_layer110;
    assign w_reduce_layer110 = reduce_layer020 + reduce_layer021;
    logic signed [5:0] w_reduce_layer111;
    assign w_reduce_layer111 = reduce_layer022 + reduce_layer023;
    logic signed [5:0] w_reduce_layer112;
    assign w_reduce_layer112 = reduce_layer024 + reduce_layer025;
    logic signed [5:0] w_reduce_layer113;
    assign w_reduce_layer113 = reduce_layer026 + reduce_layer027;
    logic signed [5:0] w_reduce_layer114;
    assign w_reduce_layer114 = reduce_layer028 + reduce_layer029;
    logic signed [5:0] w_reduce_layer115;
    assign w_reduce_layer115 = reduce_layer030 + reduce_layer031;
    logic signed [5:0] w_reduce_layer116;
    assign w_reduce_layer116 = reduce_layer032 + reduce_layer033;
    logic signed [5:0] w_reduce_layer117;
    assign w_reduce_layer117 = reduce_layer034 + reduce_layer035;
    logic signed [5:0] w_reduce_layer118;
    assign w_reduce_layer118 = reduce_layer036 + reduce_layer037;
    logic signed [5:0] w_reduce_layer119;
    assign w_reduce_layer119 = reduce_layer038 + reduce_layer039;
    logic signed [5:0] w_reduce_layer120;
    assign w_reduce_layer120 = reduce_layer040 + reduce_layer041;
    logic signed [5:0] w_reduce_layer121;
    assign w_reduce_layer121 = reduce_layer042 + reduce_layer043;
    logic signed [5:0] w_reduce_layer122;
    assign w_reduce_layer122 = reduce_layer044 + reduce_layer045;
    logic signed [5:0] w_reduce_layer123;
    assign w_reduce_layer123 = reduce_layer046 + reduce_layer047;
    logic signed [5:0] w_reduce_layer124;
    assign w_reduce_layer124 = reduce_layer048 + reduce_layer049;
    logic signed [5:0] w_reduce_layer125;
    assign w_reduce_layer125 = reduce_layer050 + reduce_layer051;
    logic signed [5:0] w_reduce_layer126;
    assign w_reduce_layer126 = reduce_layer052 + reduce_layer053;
    logic signed [5:0] w_reduce_layer127;
    assign w_reduce_layer127 = reduce_layer054 + reduce_layer055;
    logic signed [5:0] w_reduce_layer128;
    assign w_reduce_layer128 = reduce_layer056 + reduce_layer057;
    logic signed [5:0] w_reduce_layer129;
    assign w_reduce_layer129 = reduce_layer058 + reduce_layer059;
    logic signed [5:0] w_reduce_layer130;
    assign w_reduce_layer130 = reduce_layer060 + reduce_layer061;
    logic signed [5:0] w_reduce_layer131;
    assign w_reduce_layer131 = reduce_layer062 + reduce_layer063;
    logic signed [5:0] w_reduce_layer132;
    assign w_reduce_layer132 = reduce_layer064 + reduce_layer065;
    logic signed [5:0] w_reduce_layer133;
    assign w_reduce_layer133 = reduce_layer066 + reduce_layer067;
    logic signed [5:0] w_reduce_layer134;
    assign w_reduce_layer134 = reduce_layer068 + reduce_layer069;
    logic signed [5:0] w_reduce_layer135;
    assign w_reduce_layer135 = reduce_layer070 + reduce_layer071;
    logic signed [5:0] w_reduce_layer136;
    assign w_reduce_layer136 = reduce_layer072 + reduce_layer073;
    logic signed [5:0] w_reduce_layer137;
    assign w_reduce_layer137 = reduce_layer074 + reduce_layer075;
    logic signed [5:0] w_reduce_layer138;
    assign w_reduce_layer138 = reduce_layer076 + reduce_layer077;
    logic signed [5:0] w_reduce_layer139;
    assign w_reduce_layer139 = reduce_layer078 + reduce_layer079;
    logic signed [5:0] w_reduce_layer140;
    assign w_reduce_layer140 = reduce_layer080 + reduce_layer081;
    logic signed [5:0] w_reduce_layer141;
    assign w_reduce_layer141 = reduce_layer082 + reduce_layer083;
    logic signed [5:0] w_reduce_layer142;
    assign w_reduce_layer142 = reduce_layer084 + reduce_layer085;
    logic signed [5:0] w_reduce_layer143;
    assign w_reduce_layer143 = reduce_layer086 + reduce_layer087;
    logic signed [5:0] w_reduce_layer144;
    assign w_reduce_layer144 = reduce_layer088 + reduce_layer089;
    logic signed [5:0] w_reduce_layer145;
    assign w_reduce_layer145 = reduce_layer090 + reduce_layer091;
    logic signed [5:0] w_reduce_layer146;
    assign w_reduce_layer146 = reduce_layer092 + reduce_layer093;
    logic signed [5:0] w_reduce_layer147;
    assign w_reduce_layer147 = reduce_layer094 + reduce_layer095;
    logic signed [5:0] w_reduce_layer148;
    assign w_reduce_layer148 = reduce_layer096 + reduce_layer097;
    logic signed [5:0] w_reduce_layer149;
    assign w_reduce_layer149 = reduce_layer098 + reduce_layer099;
    logic signed [5:0] w_reduce_layer150;
    assign w_reduce_layer150 = reduce_layer0100 + reduce_layer0101;
    logic signed [5:0] w_reduce_layer151;
    assign w_reduce_layer151 = reduce_layer0102 + reduce_layer0103;
    logic signed [5:0] w_reduce_layer152;
    assign w_reduce_layer152 = reduce_layer0104 + reduce_layer0105;
    logic signed [5:0] w_reduce_layer153;
    assign w_reduce_layer153 = reduce_layer0106 + reduce_layer0107;
    logic signed [5:0] w_reduce_layer154;
    assign w_reduce_layer154 = reduce_layer0108 + reduce_layer0109;
    logic signed [5:0] w_reduce_layer155;
    assign w_reduce_layer155 = reduce_layer0110 + reduce_layer0111;
    logic signed [5:0] w_reduce_layer156;
    assign w_reduce_layer156 = reduce_layer0112 + reduce_layer0113;
    logic signed [5:0] w_reduce_layer157;
    assign w_reduce_layer157 = reduce_layer0114 + reduce_layer0115;
    logic signed [5:0] w_reduce_layer158;
    assign w_reduce_layer158 = reduce_layer0116 + reduce_layer0117;
    logic signed [5:0] w_reduce_layer159;
    assign w_reduce_layer159 = reduce_layer0118 + reduce_layer0119;
    logic signed [5:0] w_reduce_layer160;
    assign w_reduce_layer160 = reduce_layer0120 + reduce_layer0121;
    logic signed [5:0] w_reduce_layer161;
    assign w_reduce_layer161 = reduce_layer0122 + reduce_layer0123;
    logic signed [5:0] w_reduce_layer162;
    assign w_reduce_layer162 = reduce_layer0124 + reduce_layer0125;
    logic signed [5:0] w_reduce_layer163;
    assign w_reduce_layer163 = reduce_layer0126 + reduce_layer0127;
    logic signed [5:0] w_reduce_layer164;
    assign w_reduce_layer164 = reduce_layer0128 + reduce_layer0129;
    logic signed [5:0] w_reduce_layer165;
    assign w_reduce_layer165 = reduce_layer0130 + reduce_layer0131;
    logic signed [5:0] w_reduce_layer166;
    assign w_reduce_layer166 = reduce_layer0132 + reduce_layer0133;
    logic signed [5:0] w_reduce_layer167;
    assign w_reduce_layer167 = reduce_layer0134 + reduce_layer0135;
    logic signed [5:0] w_reduce_layer168;
    assign w_reduce_layer168 = reduce_layer0136 + reduce_layer0137;
    logic signed [5:0] w_reduce_layer169;
    assign w_reduce_layer169 = reduce_layer0138 + reduce_layer0139;
    logic signed [5:0] w_reduce_layer170;
    assign w_reduce_layer170 = reduce_layer0140 + reduce_layer0141;
    logic signed [5:0] w_reduce_layer171;
    assign w_reduce_layer171 = reduce_layer0142 + reduce_layer0143;
    logic signed [5:0] w_reduce_layer172;
    assign w_reduce_layer172 = reduce_layer0144 + reduce_layer0145;
    logic signed [5:0] w_reduce_layer173;
    assign w_reduce_layer173 = reduce_layer0146 + reduce_layer0147;
    logic signed [5:0] w_reduce_layer174;
    assign w_reduce_layer174 = reduce_layer0148 + reduce_layer0149;

// Registers
    logic signed [5:0] reduce_layer10;
    logic signed [5:0] reduce_layer11;
    logic signed [5:0] reduce_layer12;
    logic signed [5:0] reduce_layer13;
    logic signed [5:0] reduce_layer14;
    logic signed [5:0] reduce_layer15;
    logic signed [5:0] reduce_layer16;
    logic signed [5:0] reduce_layer17;
    logic signed [5:0] reduce_layer18;
    logic signed [5:0] reduce_layer19;
    logic signed [5:0] reduce_layer110;
    logic signed [5:0] reduce_layer111;
    logic signed [5:0] reduce_layer112;
    logic signed [5:0] reduce_layer113;
    logic signed [5:0] reduce_layer114;
    logic signed [5:0] reduce_layer115;
    logic signed [5:0] reduce_layer116;
    logic signed [5:0] reduce_layer117;
    logic signed [5:0] reduce_layer118;
    logic signed [5:0] reduce_layer119;
    logic signed [5:0] reduce_layer120;
    logic signed [5:0] reduce_layer121;
    logic signed [5:0] reduce_layer122;
    logic signed [5:0] reduce_layer123;
    logic signed [5:0] reduce_layer124;
    logic signed [5:0] reduce_layer125;
    logic signed [5:0] reduce_layer126;
    logic signed [5:0] reduce_layer127;
    logic signed [5:0] reduce_layer128;
    logic signed [5:0] reduce_layer129;
    logic signed [5:0] reduce_layer130;
    logic signed [5:0] reduce_layer131;
    logic signed [5:0] reduce_layer132;
    logic signed [5:0] reduce_layer133;
    logic signed [5:0] reduce_layer134;
    logic signed [5:0] reduce_layer135;
    logic signed [5:0] reduce_layer136;
    logic signed [5:0] reduce_layer137;
    logic signed [5:0] reduce_layer138;
    logic signed [5:0] reduce_layer139;
    logic signed [5:0] reduce_layer140;
    logic signed [5:0] reduce_layer141;
    logic signed [5:0] reduce_layer142;
    logic signed [5:0] reduce_layer143;
    logic signed [5:0] reduce_layer144;
    logic signed [5:0] reduce_layer145;
    logic signed [5:0] reduce_layer146;
    logic signed [5:0] reduce_layer147;
    logic signed [5:0] reduce_layer148;
    logic signed [5:0] reduce_layer149;
    logic signed [5:0] reduce_layer150;
    logic signed [5:0] reduce_layer151;
    logic signed [5:0] reduce_layer152;
    logic signed [5:0] reduce_layer153;
    logic signed [5:0] reduce_layer154;
    logic signed [5:0] reduce_layer155;
    logic signed [5:0] reduce_layer156;
    logic signed [5:0] reduce_layer157;
    logic signed [5:0] reduce_layer158;
    logic signed [5:0] reduce_layer159;
    logic signed [5:0] reduce_layer160;
    logic signed [5:0] reduce_layer161;
    logic signed [5:0] reduce_layer162;
    logic signed [5:0] reduce_layer163;
    logic signed [5:0] reduce_layer164;
    logic signed [5:0] reduce_layer165;
    logic signed [5:0] reduce_layer166;
    logic signed [5:0] reduce_layer167;
    logic signed [5:0] reduce_layer168;
    logic signed [5:0] reduce_layer169;
    logic signed [5:0] reduce_layer170;
    logic signed [5:0] reduce_layer171;
    logic signed [5:0] reduce_layer172;
    logic signed [5:0] reduce_layer173;
    logic signed [5:0] reduce_layer174;

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
    reduce_layer132 <= '0;
    reduce_layer133 <= '0;
    reduce_layer134 <= '0;
    reduce_layer135 <= '0;
    reduce_layer136 <= '0;
    reduce_layer137 <= '0;
    reduce_layer138 <= '0;
    reduce_layer139 <= '0;
    reduce_layer140 <= '0;
    reduce_layer141 <= '0;
    reduce_layer142 <= '0;
    reduce_layer143 <= '0;
    reduce_layer144 <= '0;
    reduce_layer145 <= '0;
    reduce_layer146 <= '0;
    reduce_layer147 <= '0;
    reduce_layer148 <= '0;
    reduce_layer149 <= '0;
    reduce_layer150 <= '0;
    reduce_layer151 <= '0;
    reduce_layer152 <= '0;
    reduce_layer153 <= '0;
    reduce_layer154 <= '0;
    reduce_layer155 <= '0;
    reduce_layer156 <= '0;
    reduce_layer157 <= '0;
    reduce_layer158 <= '0;
    reduce_layer159 <= '0;
    reduce_layer160 <= '0;
    reduce_layer161 <= '0;
    reduce_layer162 <= '0;
    reduce_layer163 <= '0;
    reduce_layer164 <= '0;
    reduce_layer165 <= '0;
    reduce_layer166 <= '0;
    reduce_layer167 <= '0;
    reduce_layer168 <= '0;
    reduce_layer169 <= '0;
    reduce_layer170 <= '0;
    reduce_layer171 <= '0;
    reduce_layer172 <= '0;
    reduce_layer173 <= '0;
    reduce_layer174 <= '0;
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
    reduce_layer132 <= w_reduce_layer132;
    reduce_layer133 <= w_reduce_layer133;
    reduce_layer134 <= w_reduce_layer134;
    reduce_layer135 <= w_reduce_layer135;
    reduce_layer136 <= w_reduce_layer136;
    reduce_layer137 <= w_reduce_layer137;
    reduce_layer138 <= w_reduce_layer138;
    reduce_layer139 <= w_reduce_layer139;
    reduce_layer140 <= w_reduce_layer140;
    reduce_layer141 <= w_reduce_layer141;
    reduce_layer142 <= w_reduce_layer142;
    reduce_layer143 <= w_reduce_layer143;
    reduce_layer144 <= w_reduce_layer144;
    reduce_layer145 <= w_reduce_layer145;
    reduce_layer146 <= w_reduce_layer146;
    reduce_layer147 <= w_reduce_layer147;
    reduce_layer148 <= w_reduce_layer148;
    reduce_layer149 <= w_reduce_layer149;
    reduce_layer150 <= w_reduce_layer150;
    reduce_layer151 <= w_reduce_layer151;
    reduce_layer152 <= w_reduce_layer152;
    reduce_layer153 <= w_reduce_layer153;
    reduce_layer154 <= w_reduce_layer154;
    reduce_layer155 <= w_reduce_layer155;
    reduce_layer156 <= w_reduce_layer156;
    reduce_layer157 <= w_reduce_layer157;
    reduce_layer158 <= w_reduce_layer158;
    reduce_layer159 <= w_reduce_layer159;
    reduce_layer160 <= w_reduce_layer160;
    reduce_layer161 <= w_reduce_layer161;
    reduce_layer162 <= w_reduce_layer162;
    reduce_layer163 <= w_reduce_layer163;
    reduce_layer164 <= w_reduce_layer164;
    reduce_layer165 <= w_reduce_layer165;
    reduce_layer166 <= w_reduce_layer166;
    reduce_layer167 <= w_reduce_layer167;
    reduce_layer168 <= w_reduce_layer168;
    reduce_layer169 <= w_reduce_layer169;
    reduce_layer170 <= w_reduce_layer170;
    reduce_layer171 <= w_reduce_layer171;
    reduce_layer172 <= w_reduce_layer172;
    reduce_layer173 <= w_reduce_layer173;
    reduce_layer174 <= w_reduce_layer174;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 2
/////////////////////////////////////

// Adders
    logic signed [6:0] w_reduce_layer20;
    assign w_reduce_layer20 = reduce_layer10 + reduce_layer11;
    logic signed [6:0] w_reduce_layer21;
    assign w_reduce_layer21 = reduce_layer12 + reduce_layer13;
    logic signed [6:0] w_reduce_layer22;
    assign w_reduce_layer22 = reduce_layer14 + reduce_layer15;
    logic signed [6:0] w_reduce_layer23;
    assign w_reduce_layer23 = reduce_layer16 + reduce_layer17;
    logic signed [6:0] w_reduce_layer24;
    assign w_reduce_layer24 = reduce_layer18 + reduce_layer19;
    logic signed [6:0] w_reduce_layer25;
    assign w_reduce_layer25 = reduce_layer110 + reduce_layer111;
    logic signed [6:0] w_reduce_layer26;
    assign w_reduce_layer26 = reduce_layer112 + reduce_layer113;
    logic signed [6:0] w_reduce_layer27;
    assign w_reduce_layer27 = reduce_layer114 + reduce_layer115;
    logic signed [6:0] w_reduce_layer28;
    assign w_reduce_layer28 = reduce_layer116 + reduce_layer117;
    logic signed [6:0] w_reduce_layer29;
    assign w_reduce_layer29 = reduce_layer118 + reduce_layer119;
    logic signed [6:0] w_reduce_layer210;
    assign w_reduce_layer210 = reduce_layer120 + reduce_layer121;
    logic signed [6:0] w_reduce_layer211;
    assign w_reduce_layer211 = reduce_layer122 + reduce_layer123;
    logic signed [6:0] w_reduce_layer212;
    assign w_reduce_layer212 = reduce_layer124 + reduce_layer125;
    logic signed [6:0] w_reduce_layer213;
    assign w_reduce_layer213 = reduce_layer126 + reduce_layer127;
    logic signed [6:0] w_reduce_layer214;
    assign w_reduce_layer214 = reduce_layer128 + reduce_layer129;
    logic signed [6:0] w_reduce_layer215;
    assign w_reduce_layer215 = reduce_layer130 + reduce_layer131;
    logic signed [6:0] w_reduce_layer216;
    assign w_reduce_layer216 = reduce_layer132 + reduce_layer133;
    logic signed [6:0] w_reduce_layer217;
    assign w_reduce_layer217 = reduce_layer134 + reduce_layer135;
    logic signed [6:0] w_reduce_layer218;
    assign w_reduce_layer218 = reduce_layer136 + reduce_layer137;
    logic signed [6:0] w_reduce_layer219;
    assign w_reduce_layer219 = reduce_layer138 + reduce_layer139;
    logic signed [6:0] w_reduce_layer220;
    assign w_reduce_layer220 = reduce_layer140 + reduce_layer141;
    logic signed [6:0] w_reduce_layer221;
    assign w_reduce_layer221 = reduce_layer142 + reduce_layer143;
    logic signed [6:0] w_reduce_layer222;
    assign w_reduce_layer222 = reduce_layer144 + reduce_layer145;
    logic signed [6:0] w_reduce_layer223;
    assign w_reduce_layer223 = reduce_layer146 + reduce_layer147;
    logic signed [6:0] w_reduce_layer224;
    assign w_reduce_layer224 = reduce_layer148 + reduce_layer149;
    logic signed [6:0] w_reduce_layer225;
    assign w_reduce_layer225 = reduce_layer150 + reduce_layer151;
    logic signed [6:0] w_reduce_layer226;
    assign w_reduce_layer226 = reduce_layer152 + reduce_layer153;
    logic signed [6:0] w_reduce_layer227;
    assign w_reduce_layer227 = reduce_layer154 + reduce_layer155;
    logic signed [6:0] w_reduce_layer228;
    assign w_reduce_layer228 = reduce_layer156 + reduce_layer157;
    logic signed [6:0] w_reduce_layer229;
    assign w_reduce_layer229 = reduce_layer158 + reduce_layer159;
    logic signed [6:0] w_reduce_layer230;
    assign w_reduce_layer230 = reduce_layer160 + reduce_layer161;
    logic signed [6:0] w_reduce_layer231;
    assign w_reduce_layer231 = reduce_layer162 + reduce_layer163;
    logic signed [6:0] w_reduce_layer232;
    assign w_reduce_layer232 = reduce_layer164 + reduce_layer165;
    logic signed [6:0] w_reduce_layer233;
    assign w_reduce_layer233 = reduce_layer166 + reduce_layer167;
    logic signed [6:0] w_reduce_layer234;
    assign w_reduce_layer234 = reduce_layer168 + reduce_layer169;
    logic signed [6:0] w_reduce_layer235;
    assign w_reduce_layer235 = reduce_layer170 + reduce_layer171;
    logic signed [6:0] w_reduce_layer236;
    assign w_reduce_layer236 = reduce_layer172 + reduce_layer173;

// Registers
    logic signed [6:0] reduce_layer20;
    logic signed [6:0] reduce_layer21;
    logic signed [6:0] reduce_layer22;
    logic signed [6:0] reduce_layer23;
    logic signed [6:0] reduce_layer24;
    logic signed [6:0] reduce_layer25;
    logic signed [6:0] reduce_layer26;
    logic signed [6:0] reduce_layer27;
    logic signed [6:0] reduce_layer28;
    logic signed [6:0] reduce_layer29;
    logic signed [6:0] reduce_layer210;
    logic signed [6:0] reduce_layer211;
    logic signed [6:0] reduce_layer212;
    logic signed [6:0] reduce_layer213;
    logic signed [6:0] reduce_layer214;
    logic signed [6:0] reduce_layer215;
    logic signed [6:0] reduce_layer216;
    logic signed [6:0] reduce_layer217;
    logic signed [6:0] reduce_layer218;
    logic signed [6:0] reduce_layer219;
    logic signed [6:0] reduce_layer220;
    logic signed [6:0] reduce_layer221;
    logic signed [6:0] reduce_layer222;
    logic signed [6:0] reduce_layer223;
    logic signed [6:0] reduce_layer224;
    logic signed [6:0] reduce_layer225;
    logic signed [6:0] reduce_layer226;
    logic signed [6:0] reduce_layer227;
    logic signed [6:0] reduce_layer228;
    logic signed [6:0] reduce_layer229;
    logic signed [6:0] reduce_layer230;
    logic signed [6:0] reduce_layer231;
    logic signed [6:0] reduce_layer232;
    logic signed [6:0] reduce_layer233;
    logic signed [6:0] reduce_layer234;
    logic signed [6:0] reduce_layer235;
    logic signed [6:0] reduce_layer236;

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
    reduce_layer216 <= '0;
    reduce_layer217 <= '0;
    reduce_layer218 <= '0;
    reduce_layer219 <= '0;
    reduce_layer220 <= '0;
    reduce_layer221 <= '0;
    reduce_layer222 <= '0;
    reduce_layer223 <= '0;
    reduce_layer224 <= '0;
    reduce_layer225 <= '0;
    reduce_layer226 <= '0;
    reduce_layer227 <= '0;
    reduce_layer228 <= '0;
    reduce_layer229 <= '0;
    reduce_layer230 <= '0;
    reduce_layer231 <= '0;
    reduce_layer232 <= '0;
    reduce_layer233 <= '0;
    reduce_layer234 <= '0;
    reduce_layer235 <= '0;
    reduce_layer236 <= '0;
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
    reduce_layer216 <= w_reduce_layer216;
    reduce_layer217 <= w_reduce_layer217;
    reduce_layer218 <= w_reduce_layer218;
    reduce_layer219 <= w_reduce_layer219;
    reduce_layer220 <= w_reduce_layer220;
    reduce_layer221 <= w_reduce_layer221;
    reduce_layer222 <= w_reduce_layer222;
    reduce_layer223 <= w_reduce_layer223;
    reduce_layer224 <= w_reduce_layer224;
    reduce_layer225 <= w_reduce_layer225;
    reduce_layer226 <= w_reduce_layer226;
    reduce_layer227 <= w_reduce_layer227;
    reduce_layer228 <= w_reduce_layer228;
    reduce_layer229 <= w_reduce_layer229;
    reduce_layer230 <= w_reduce_layer230;
    reduce_layer231 <= w_reduce_layer231;
    reduce_layer232 <= w_reduce_layer232;
    reduce_layer233 <= w_reduce_layer233;
    reduce_layer234 <= w_reduce_layer234;
    reduce_layer235 <= w_reduce_layer235;
    reduce_layer236 <= w_reduce_layer236;
  end
end

// Odd input passthrough
    logic signed [6:0] w_reduce_layer237;
    logic signed [6:0] reduce_layer237;
    assign w_reduce_layer237 = reduce_layer174;
    always_ff @(posedge clk) begin
      if (rst)
        reduce_layer237 <= '0;
      else
        reduce_layer237 <= w_reduce_layer237;
    end


/////////////////////////////////////
// Layer 3
/////////////////////////////////////

// Adders
    logic signed [7:0] w_reduce_layer30;
    assign w_reduce_layer30 = reduce_layer20 + reduce_layer21;
    logic signed [7:0] w_reduce_layer31;
    assign w_reduce_layer31 = reduce_layer22 + reduce_layer23;
    logic signed [7:0] w_reduce_layer32;
    assign w_reduce_layer32 = reduce_layer24 + reduce_layer25;
    logic signed [7:0] w_reduce_layer33;
    assign w_reduce_layer33 = reduce_layer26 + reduce_layer27;
    logic signed [7:0] w_reduce_layer34;
    assign w_reduce_layer34 = reduce_layer28 + reduce_layer29;
    logic signed [7:0] w_reduce_layer35;
    assign w_reduce_layer35 = reduce_layer210 + reduce_layer211;
    logic signed [7:0] w_reduce_layer36;
    assign w_reduce_layer36 = reduce_layer212 + reduce_layer213;
    logic signed [7:0] w_reduce_layer37;
    assign w_reduce_layer37 = reduce_layer214 + reduce_layer215;
    logic signed [7:0] w_reduce_layer38;
    assign w_reduce_layer38 = reduce_layer216 + reduce_layer217;
    logic signed [7:0] w_reduce_layer39;
    assign w_reduce_layer39 = reduce_layer218 + reduce_layer219;
    logic signed [7:0] w_reduce_layer310;
    assign w_reduce_layer310 = reduce_layer220 + reduce_layer221;
    logic signed [7:0] w_reduce_layer311;
    assign w_reduce_layer311 = reduce_layer222 + reduce_layer223;
    logic signed [7:0] w_reduce_layer312;
    assign w_reduce_layer312 = reduce_layer224 + reduce_layer225;
    logic signed [7:0] w_reduce_layer313;
    assign w_reduce_layer313 = reduce_layer226 + reduce_layer227;
    logic signed [7:0] w_reduce_layer314;
    assign w_reduce_layer314 = reduce_layer228 + reduce_layer229;
    logic signed [7:0] w_reduce_layer315;
    assign w_reduce_layer315 = reduce_layer230 + reduce_layer231;
    logic signed [7:0] w_reduce_layer316;
    assign w_reduce_layer316 = reduce_layer232 + reduce_layer233;
    logic signed [7:0] w_reduce_layer317;
    assign w_reduce_layer317 = reduce_layer234 + reduce_layer235;
    logic signed [7:0] w_reduce_layer318;
    assign w_reduce_layer318 = reduce_layer236 + reduce_layer237;

// Registers
    logic signed [7:0] reduce_layer30;
    logic signed [7:0] reduce_layer31;
    logic signed [7:0] reduce_layer32;
    logic signed [7:0] reduce_layer33;
    logic signed [7:0] reduce_layer34;
    logic signed [7:0] reduce_layer35;
    logic signed [7:0] reduce_layer36;
    logic signed [7:0] reduce_layer37;
    logic signed [7:0] reduce_layer38;
    logic signed [7:0] reduce_layer39;
    logic signed [7:0] reduce_layer310;
    logic signed [7:0] reduce_layer311;
    logic signed [7:0] reduce_layer312;
    logic signed [7:0] reduce_layer313;
    logic signed [7:0] reduce_layer314;
    logic signed [7:0] reduce_layer315;
    logic signed [7:0] reduce_layer316;
    logic signed [7:0] reduce_layer317;
    logic signed [7:0] reduce_layer318;

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
    reduce_layer38 <= '0;
    reduce_layer39 <= '0;
    reduce_layer310 <= '0;
    reduce_layer311 <= '0;
    reduce_layer312 <= '0;
    reduce_layer313 <= '0;
    reduce_layer314 <= '0;
    reduce_layer315 <= '0;
    reduce_layer316 <= '0;
    reduce_layer317 <= '0;
    reduce_layer318 <= '0;
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
    reduce_layer38 <= w_reduce_layer38;
    reduce_layer39 <= w_reduce_layer39;
    reduce_layer310 <= w_reduce_layer310;
    reduce_layer311 <= w_reduce_layer311;
    reduce_layer312 <= w_reduce_layer312;
    reduce_layer313 <= w_reduce_layer313;
    reduce_layer314 <= w_reduce_layer314;
    reduce_layer315 <= w_reduce_layer315;
    reduce_layer316 <= w_reduce_layer316;
    reduce_layer317 <= w_reduce_layer317;
    reduce_layer318 <= w_reduce_layer318;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 4
/////////////////////////////////////

// Adders
    logic signed [8:0] w_reduce_layer40;
    assign w_reduce_layer40 = reduce_layer30 + reduce_layer31;
    logic signed [8:0] w_reduce_layer41;
    assign w_reduce_layer41 = reduce_layer32 + reduce_layer33;
    logic signed [8:0] w_reduce_layer42;
    assign w_reduce_layer42 = reduce_layer34 + reduce_layer35;
    logic signed [8:0] w_reduce_layer43;
    assign w_reduce_layer43 = reduce_layer36 + reduce_layer37;
    logic signed [8:0] w_reduce_layer44;
    assign w_reduce_layer44 = reduce_layer38 + reduce_layer39;
    logic signed [8:0] w_reduce_layer45;
    assign w_reduce_layer45 = reduce_layer310 + reduce_layer311;
    logic signed [8:0] w_reduce_layer46;
    assign w_reduce_layer46 = reduce_layer312 + reduce_layer313;
    logic signed [8:0] w_reduce_layer47;
    assign w_reduce_layer47 = reduce_layer314 + reduce_layer315;
    logic signed [8:0] w_reduce_layer48;
    assign w_reduce_layer48 = reduce_layer316 + reduce_layer317;

// Registers
    logic signed [8:0] reduce_layer40;
    logic signed [8:0] reduce_layer41;
    logic signed [8:0] reduce_layer42;
    logic signed [8:0] reduce_layer43;
    logic signed [8:0] reduce_layer44;
    logic signed [8:0] reduce_layer45;
    logic signed [8:0] reduce_layer46;
    logic signed [8:0] reduce_layer47;
    logic signed [8:0] reduce_layer48;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer40 <= '0;
    reduce_layer41 <= '0;
    reduce_layer42 <= '0;
    reduce_layer43 <= '0;
    reduce_layer44 <= '0;
    reduce_layer45 <= '0;
    reduce_layer46 <= '0;
    reduce_layer47 <= '0;
    reduce_layer48 <= '0;
  end
  else begin
    reduce_layer40 <= w_reduce_layer40;
    reduce_layer41 <= w_reduce_layer41;
    reduce_layer42 <= w_reduce_layer42;
    reduce_layer43 <= w_reduce_layer43;
    reduce_layer44 <= w_reduce_layer44;
    reduce_layer45 <= w_reduce_layer45;
    reduce_layer46 <= w_reduce_layer46;
    reduce_layer47 <= w_reduce_layer47;
    reduce_layer48 <= w_reduce_layer48;
  end
end

// Odd input passthrough
    logic signed [8:0] w_reduce_layer49;
    logic signed [8:0] reduce_layer49;
    assign w_reduce_layer49 = reduce_layer318;
    always_ff @(posedge clk) begin
      if (rst)
        reduce_layer49 <= '0;
      else
        reduce_layer49 <= w_reduce_layer49;
    end


/////////////////////////////////////
// Layer 5
/////////////////////////////////////

// Adders
    logic signed [9:0] w_reduce_layer50;
    assign w_reduce_layer50 = reduce_layer40 + reduce_layer41;
    logic signed [9:0] w_reduce_layer51;
    assign w_reduce_layer51 = reduce_layer42 + reduce_layer43;
    logic signed [9:0] w_reduce_layer52;
    assign w_reduce_layer52 = reduce_layer44 + reduce_layer45;
    logic signed [9:0] w_reduce_layer53;
    assign w_reduce_layer53 = reduce_layer46 + reduce_layer47;
    logic signed [9:0] w_reduce_layer54;
    assign w_reduce_layer54 = reduce_layer48 + reduce_layer49;

// Registers
    logic signed [9:0] reduce_layer50;
    logic signed [9:0] reduce_layer51;
    logic signed [9:0] reduce_layer52;
    logic signed [9:0] reduce_layer53;
    logic signed [9:0] reduce_layer54;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer50 <= '0;
    reduce_layer51 <= '0;
    reduce_layer52 <= '0;
    reduce_layer53 <= '0;
    reduce_layer54 <= '0;
  end
  else begin
    reduce_layer50 <= w_reduce_layer50;
    reduce_layer51 <= w_reduce_layer51;
    reduce_layer52 <= w_reduce_layer52;
    reduce_layer53 <= w_reduce_layer53;
    reduce_layer54 <= w_reduce_layer54;
  end
end

// Odd input passthrough


/////////////////////////////////////
// Layer 6
/////////////////////////////////////

// Adders
    logic signed [10:0] w_reduce_layer60;
    assign w_reduce_layer60 = reduce_layer50 + reduce_layer51;
    logic signed [10:0] w_reduce_layer61;
    assign w_reduce_layer61 = reduce_layer52 + reduce_layer53;

// Registers
    logic signed [10:0] reduce_layer60;
    logic signed [10:0] reduce_layer61;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer60 <= '0;
    reduce_layer61 <= '0;
  end
  else begin
    reduce_layer60 <= w_reduce_layer60;
    reduce_layer61 <= w_reduce_layer61;
  end
end

// Odd input passthrough
    logic signed [10:0] w_reduce_layer62;
    logic signed [10:0] reduce_layer62;
    assign w_reduce_layer62 = reduce_layer54;
    always_ff @(posedge clk) begin
      if (rst)
        reduce_layer62 <= '0;
      else
        reduce_layer62 <= w_reduce_layer62;
    end


/////////////////////////////////////
// Layer 7
/////////////////////////////////////

// Adders
    logic signed [11:0] w_reduce_layer70;
    assign w_reduce_layer70 = reduce_layer60 + reduce_layer61;

// Registers
    logic signed [11:0] reduce_layer70;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer70 <= '0;
  end
  else begin
    reduce_layer70 <= w_reduce_layer70;
  end
end

// Odd input passthrough
    logic signed [11:0] w_reduce_layer71;
    logic signed [11:0] reduce_layer71;
    assign w_reduce_layer71 = reduce_layer62;
    always_ff @(posedge clk) begin
      if (rst)
        reduce_layer71 <= '0;
      else
        reduce_layer71 <= w_reduce_layer71;
    end


/////////////////////////////////////
// Layer 8
/////////////////////////////////////

// Adders
    logic signed [12:0] w_reduce_layer80;
    assign w_reduce_layer80 = reduce_layer70 + reduce_layer71;

// Registers
    logic signed [12:0] reduce_layer80;

always_ff @(posedge clk) begin
  if (rst) begin
    reduce_layer80 <= '0;
  end
  else begin
    reduce_layer80 <= w_reduce_layer80;
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
    logic vld_layer7;
    logic vld_layer8;

always_ff @(posedge clk) begin
  if (rst) begin
    vld_layer1 <= 0;
    vld_layer2 <= 0;
    vld_layer3 <= 0;
    vld_layer4 <= 0;
    vld_layer5 <= 0;
    vld_layer6 <= 0;
    vld_layer7 <= 0;
    vld_layer8 <= 0;
  end else begin
    vld_layer1 <= vld_layer0;
    vld_layer2 <= vld_layer1;
    vld_layer3 <= vld_layer2;
    vld_layer4 <= vld_layer3;
    vld_layer5 <= vld_layer4;
    vld_layer6 <= vld_layer5;
    vld_layer7 <= vld_layer6;
    vld_layer8 <= vld_layer7;
  end
end

assign sum      = reduce_layer80;
assign outp_vld = vld_layer8;

endmodule

