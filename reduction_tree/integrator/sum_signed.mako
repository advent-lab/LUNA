<%!
    import math
%>\

module sum_signed #(
    parameter int M = ${M},   // number of inputs
    parameter int W = ${W}   // input width
)(
    input  logic clk,
    input  logic rst,
    input  logic inp_vld,
    output logic outp_vld,

    input  logic signed [W-1:0] inp [M],
    output logic signed [W+${math.ceil(math.log2(M))}-1:0] sum
);

/////////////////////////////////////
// Inputs are layer 0
/////////////////////////////////////

% for g in range(M):
    logic signed [W-1:0] reduce_layer0${g};
    assign reduce_layer0${g} = inp[${g}];
% endfor

<%
    idx = 1
    layer_inputs = M
    out_width = W
%>\
% while layer_inputs > 1:
<%
    layer_outputs = math.ceil(layer_inputs / 2)
    odd_input = (layer_inputs % 2 == 1)
    out_width = out_width + 1
%>\

/////////////////////////////////////
// Layer ${idx}
/////////////////////////////////////

// Adders
% for g in range(layer_outputs - (1 if odd_input else 0)):
    logic signed [${out_width-1}:0] w_reduce_layer${idx}${g};
    assign w_reduce_layer${idx}${g} = reduce_layer${idx-1}${2*g} + reduce_layer${idx-1}${2*g+1};
% endfor

// Registers
% for g in range(layer_outputs - (1 if odd_input else 0)):
    logic signed [${out_width-1}:0] reduce_layer${idx}${g};
% endfor

always_ff @(posedge clk) begin
  if (rst) begin
% for g in range(layer_outputs - (1 if odd_input else 0)):
    reduce_layer${idx}${g} <= '0;
% endfor
  end
  else begin
% for g in range(layer_outputs - (1 if odd_input else 0)):
    reduce_layer${idx}${g} <= w_reduce_layer${idx}${g};
% endfor
  end
end

// Odd input passthrough
% if odd_input:
    logic signed [${out_width-1}:0] w_reduce_layer${idx}${layer_outputs-1};
    logic signed [${out_width-1}:0] reduce_layer${idx}${layer_outputs-1};
    assign w_reduce_layer${idx}${layer_outputs-1} = reduce_layer${idx-1}${layer_inputs-1};
    always_ff @(posedge clk) begin
      if (rst)
        reduce_layer${idx}${layer_outputs-1} <= '0;
      else
        reduce_layer${idx}${layer_outputs-1} <= w_reduce_layer${idx}${layer_outputs-1};
    end
% endif

<%
    layer_inputs = layer_outputs
    idx += 1
%>\
% endwhile

/////////////////////////////////////
// Valid pipeline
/////////////////////////////////////

logic vld_layer0;
assign vld_layer0 = inp_vld;

% for i in range(1, idx):
    logic vld_layer${i};
% endfor

always_ff @(posedge clk) begin
  if (rst) begin
% for i in range(1, idx):
    vld_layer${i} <= 0;
% endfor
  end else begin
% for i in range(1, idx):
    vld_layer${i} <= vld_layer${i-1};
% endfor
  end
end

assign sum      = reduce_layer${idx-1}0;
assign outp_vld = vld_layer${idx-1};

endmodule

