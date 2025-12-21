<%!
    import math
%>

module ${top} #(
    parameter M = ${M},
    parameter W = ${W},
    parameter N = ${N}
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

<%
levels = int(math.ceil(math.log(M,2)))
%>

% for lvl in range(1, levels+1):
    % if lvl == 1:
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
    % else:
        localparam int STAGE${lvl}_SIZE = (STAGE${lvl-1}_SIZE + 1) >> 1;
        logic [SUMW-1:0] stage${lvl} [0:STAGE${lvl}_SIZE-1];
        always_ff @(posedge clk) begin
            if (rst) begin
                for (int k=0; k<STAGE${lvl}_SIZE; k++) stage${lvl}[k] <= '0;
            end else begin
                for (int k=0; k<STAGE${lvl}_SIZE; k++) begin
                    if (2*k+1 < STAGE${lvl-1}_SIZE)
                        stage${lvl}[k] <= stage${lvl-1}[2*k] + stage${lvl-1}[2*k+1];
                    else
                        stage${lvl}[k] <= stage${lvl-1}[2*k];
                end
            end
        end
    % endif
% endfor

    // ------------------------------------------------------------------------
    // Valid pipeline
    // ------------------------------------------------------------------------
% for i in range(1, levels+1):
    logic vld_layer${i};
% endfor

    always_ff @(posedge clk) begin
        if (rst) begin
% for i in range(1, levels+1):
            vld_layer${i} <= 1'b0;
% endfor
        end else begin
            vld_layer1 <= vld_in;
% for i in range(2, levels+1):
            vld_layer${i} <= vld_layer${i-1};
% endfor
        end
    end

    assign result  = stage${levels}[0];
    assign vld_out = vld_layer${levels};

endmodule

