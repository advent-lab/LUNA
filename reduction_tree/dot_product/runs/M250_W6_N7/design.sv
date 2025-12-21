

module dotprod #(
    parameter M = 250,
    parameter W = 6,
    parameter N = 7
) (
    input  logic clk,
    input  logic rst,
    input  logic [M*W-1:0] data_in,
    input  logic [M*N-1:0] weights,
    input  logic vld_in,
    output logic [W+N+$clog2(M)-1:0] result,
    output logic vld_out
);

    // ------------------------------------------------------------------------
    // Multiply stage: M parallel multipliers
    // ------------------------------------------------------------------------
    logic [W+N-1:0] prods [0:M-1];
    genvar i;
    generate
        for (i = 0; i < M; i++) begin
            assign prods[i] = data_in[i*W +: W] * weights[i*N +: N];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Pipelined adder tree
    // ------------------------------------------------------------------------
    localparam int SUMW = W + N + $clog2(M);

    // Stage 0
    logic [SUMW-1:0] stage0 [0:M-1];
    always_comb begin
        for (int j = 0; j < M; j++) stage0[j] = prods[j];
    end



        localparam int STAGE1_SIZE = (M + 1) >> 1;
        logic [SUMW-1:0] stage1 [0:STAGE1_SIZE-1];
        always_ff @(posedge clk) begin
            if (rst) begin
                for (int k=0; k<STAGE1_SIZE; k++) stage1[k] <= '0;
            end else begin
                for (int k=0; k<STAGE1_SIZE; k++) begin
                    if (2*k+1 < M)
                        stage1[k] <= stage0[2*k] + stage0[2*k+1];
                    else
                        stage1[k] <= stage0[2*k];
                end
            end
        end
        localparam int STAGE2_SIZE = (STAGE1_SIZE + 1) >> 1;
        logic [SUMW-1:0] stage2 [0:STAGE2_SIZE-1];
        always_ff @(posedge clk) begin
            if (rst) begin
                for (int k=0; k<STAGE2_SIZE; k++) stage2[k] <= '0;
            end else begin
                for (int k=0; k<STAGE2_SIZE; k++) begin
                    if (2*k+1 < STAGE1_SIZE)
                        stage2[k] <= stage1[2*k] + stage1[2*k+1];
                    else
                        stage2[k] <= stage1[2*k];
                end
            end
        end
        localparam int STAGE3_SIZE = (STAGE2_SIZE + 1) >> 1;
        logic [SUMW-1:0] stage3 [0:STAGE3_SIZE-1];
        always_ff @(posedge clk) begin
            if (rst) begin
                for (int k=0; k<STAGE3_SIZE; k++) stage3[k] <= '0;
            end else begin
                for (int k=0; k<STAGE3_SIZE; k++) begin
                    if (2*k+1 < STAGE2_SIZE)
                        stage3[k] <= stage2[2*k] + stage2[2*k+1];
                    else
                        stage3[k] <= stage2[2*k];
                end
            end
        end
        localparam int STAGE4_SIZE = (STAGE3_SIZE + 1) >> 1;
        logic [SUMW-1:0] stage4 [0:STAGE4_SIZE-1];
        always_ff @(posedge clk) begin
            if (rst) begin
                for (int k=0; k<STAGE4_SIZE; k++) stage4[k] <= '0;
            end else begin
                for (int k=0; k<STAGE4_SIZE; k++) begin
                    if (2*k+1 < STAGE3_SIZE)
                        stage4[k] <= stage3[2*k] + stage3[2*k+1];
                    else
                        stage4[k] <= stage3[2*k];
                end
            end
        end
        localparam int STAGE5_SIZE = (STAGE4_SIZE + 1) >> 1;
        logic [SUMW-1:0] stage5 [0:STAGE5_SIZE-1];
        always_ff @(posedge clk) begin
            if (rst) begin
                for (int k=0; k<STAGE5_SIZE; k++) stage5[k] <= '0;
            end else begin
                for (int k=0; k<STAGE5_SIZE; k++) begin
                    if (2*k+1 < STAGE4_SIZE)
                        stage5[k] <= stage4[2*k] + stage4[2*k+1];
                    else
                        stage5[k] <= stage4[2*k];
                end
            end
        end
        localparam int STAGE6_SIZE = (STAGE5_SIZE + 1) >> 1;
        logic [SUMW-1:0] stage6 [0:STAGE6_SIZE-1];
        always_ff @(posedge clk) begin
            if (rst) begin
                for (int k=0; k<STAGE6_SIZE; k++) stage6[k] <= '0;
            end else begin
                for (int k=0; k<STAGE6_SIZE; k++) begin
                    if (2*k+1 < STAGE5_SIZE)
                        stage6[k] <= stage5[2*k] + stage5[2*k+1];
                    else
                        stage6[k] <= stage5[2*k];
                end
            end
        end
        localparam int STAGE7_SIZE = (STAGE6_SIZE + 1) >> 1;
        logic [SUMW-1:0] stage7 [0:STAGE7_SIZE-1];
        always_ff @(posedge clk) begin
            if (rst) begin
                for (int k=0; k<STAGE7_SIZE; k++) stage7[k] <= '0;
            end else begin
                for (int k=0; k<STAGE7_SIZE; k++) begin
                    if (2*k+1 < STAGE6_SIZE)
                        stage7[k] <= stage6[2*k] + stage6[2*k+1];
                    else
                        stage7[k] <= stage6[2*k];
                end
            end
        end
        localparam int STAGE8_SIZE = (STAGE7_SIZE + 1) >> 1;
        logic [SUMW-1:0] stage8 [0:STAGE8_SIZE-1];
        always_ff @(posedge clk) begin
            if (rst) begin
                for (int k=0; k<STAGE8_SIZE; k++) stage8[k] <= '0;
            end else begin
                for (int k=0; k<STAGE8_SIZE; k++) begin
                    if (2*k+1 < STAGE7_SIZE)
                        stage8[k] <= stage7[2*k] + stage7[2*k+1];
                    else
                        stage8[k] <= stage7[2*k];
                end
            end
        end

    // ------------------------------------------------------------------------
    // Valid pipeline
    // ------------------------------------------------------------------------
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
            vld_layer1 <= 1'b0;
            vld_layer2 <= 1'b0;
            vld_layer3 <= 1'b0;
            vld_layer4 <= 1'b0;
            vld_layer5 <= 1'b0;
            vld_layer6 <= 1'b0;
            vld_layer7 <= 1'b0;
            vld_layer8 <= 1'b0;
        end else begin
            vld_layer1 <= vld_in;
            vld_layer2 <= vld_layer1;
            vld_layer3 <= vld_layer2;
            vld_layer4 <= vld_layer3;
            vld_layer5 <= vld_layer4;
            vld_layer6 <= vld_layer5;
            vld_layer7 <= vld_layer6;
            vld_layer8 <= vld_layer7;
        end
    end

    assign result  = stage8[0];
    assign vld_out = vld_layer8;

endmodule

