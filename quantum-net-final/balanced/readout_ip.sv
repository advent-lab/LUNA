// This Verilog module implements the top-level wrapper for the Neural Network Classifier IP
// based on the QICK system description, handling AXI-Stream data loading,
// and local BRAM storage of the prediction results.
`include "sum_signed.sv"
//`include "./logicnets/"
module nn_classifier_wrapper #(
    // --- Configuration Parameters ---
    // Total number of I/Q samples (14-bit pairs) to load per trigger.
    parameter WINDOW_SIZE     = 400, // UPDATED: Total signal length is 400
    // The number of internal windows or data paths the data is split into for the accelerator.
    parameter NUM_WINDOWS     = 2,
    // Number of LSBs to ignore (shift right) from the 14-bit I/Q values (Pre-accumulation shift).
    parameter SHIFT_M         = 9, // UPDATED: Pre-accumulation shift is 7
    // Number of LSBs to ignore from the accumulated sum (Post-accumulation shift).
    parameter SHIFT_N         = 1, // UPDATED: Post-accumulation shift is now 1
    
    // --- Interface Parameters ---
    // Width of the input AXI-Stream data (I, Q, Padding).
    parameter DATA_WIDTH      = 32,
    // Width of the raw I/Q values as they come from the ADC (14 bits).
    parameter IQ_WIDTH_IN     = 14,
    // Width of the I/Q values after the SHIFT_M operation.
    localparam IQ_WIDTH_OUT   = IQ_WIDTH_IN - SHIFT_M, // 14 - 7 = 7 bits
    // Width of the NN prediction output (User requested 2 bits).
    parameter PRED_BITS       = 2,
    // BRAM address bits for the 16384 predictions (2^14 = 16384).
    parameter BRAM_ADDR_BITS  = 14
)(
    // --- System Signals ---
    input  wire ap_clk,          // Clock signal
    input  wire ap_rst_n,        // Asynchronous reset (active low)
    
    // --- AXI4-Lite Configuration Interface (Simplified Placeholder) ---
    // Signals required by HLS for the s_axi_config block, only control signals shown.
    input  wire [3:0] config_AWADDR,
    input  wire config_AWVALID,
    output wire config_AWREADY,
    // ... other AXI-Lite signals would typically be here
    
    // --- Trigger Signal ---
    input  wire trigger,         // Pulse to initiate classification
    
    // --- AXIS Input Interface ('in') ---
    // Data stream from QICK Readout Block (No TREADY/TLAST as per description)
    // 32-bit word: [31:18] I, [17:4] Q, [3:0] Padding
    input  wire [DATA_WIDTH-1:0] in_TDATA,  // 32-bit word
    input  wire                  in_TVALID, // Data is valid
    // in_TREADY is omitted as the IP is assumed to always accept data (no backpressure)

    // --- BRAM Output Interface ('out') ---
    // Simple memory write interface to PL BRAM for prediction storage
    output wire [BRAM_ADDR_BITS-1:0] out_ADDR, // BRAM write address
    output reg  [PRED_BITS-1:0]      out_DATA, // BRAM write data (2-bit prediction)
    output reg                   out_WE      // BRAM Write Enable
);

    // =========================================================================
    // I. Internal State Machine and Control
    // =========================================================================
    
    // State Definitions
    localparam S_IDLE       = 3'b000; // Waiting for trigger
    localparam S_LOAD       = 3'b001; // Loading data from AXIS
    localparam S_COMPUTE    = 3'b010; // NN Inference (Accelerator active)
    localparam S_STORE      = 3'b011; // Storing result to BRAM
    
    reg [2:0] current_state, next_state;
    
    // Counters
    reg [$clog2(WINDOW_SIZE):0] load_count; // Tracks number of samples loaded (0 to WINDOW_SIZE)
    reg [3:0] compute_count;               // Tracks NN computation cycles (used for placeholder latency)
    reg [1:0] store_count;                 // Tracks BRAM write cycles (2 cycles)
    reg [BRAM_ADDR_BITS-1:0] pred_addr_reg; // BRAM address for prediction results

    // --- State Logic (Sequential Block) ---
    always @(posedge ap_clk or negedge ap_rst_n) begin
        if (!ap_rst_n)
            current_state <= S_IDLE;
        else
            current_state <= next_state;
    end

    // FSM Transitions (Combinational Block)
    always @(*) begin
        next_state = current_state;
        
        case (current_state)
            S_IDLE: begin
                // Transition on trigger pulse
                if (trigger)
                    next_state = S_LOAD;
            end
            
            S_LOAD: begin
                // Load phase lasts for WINDOW_SIZE clock cycles
                if (load_count == WINDOW_SIZE)
                    next_state = S_COMPUTE;
            end
            
            S_COMPUTE: begin
                // Transition after NN latency (placeholder 5 cycles minimum)
                if (compute_count >= 5) 
                    next_state = S_STORE;
            end
            
            S_STORE: begin
                // Fixed 2-cycle store latency
                if (store_count == 2) 
                    next_state = S_IDLE;
            end
            
            default: next_state = S_IDLE;
        endcase
    end
    
    // =========================================================================
    // II. Data Loading and Shifting Logic
    // =========================================================================

    // Data Memories for I and Q samples
    // Total size is WINDOW_SIZE samples, each of width IQ_WIDTH_OUT (7 bits)
    reg [IQ_WIDTH_OUT-1:0] i_memory [0:WINDOW_SIZE-1];
    reg [IQ_WIDTH_OUT-1:0] q_memory [0:WINDOW_SIZE-1];

    // Data Load/Counter Logic (Sequential Block)
    always @(posedge ap_clk or negedge ap_rst_n) begin
        if (!ap_rst_n) begin
            load_count <= 0;
            pred_addr_reg <= 0;
        end else begin
            case (current_state)
                S_IDLE: begin
                    load_count <= 0;
                    // Pre-increment the BRAM address on transition to LOAD
                    if (next_state == S_LOAD)
                        pred_addr_reg <= pred_addr_reg + 1;
                end
                
                S_LOAD: begin
                    if (in_TVALID && load_count < WINDOW_SIZE) begin
                        
                        // Shift Operation (Pre-accumulation): [13:SHIFT_M] bits are captured
                        
                        // I value storage (shifted)
                        // I is [31:18]. Shifted slice is [31 : 18 + SHIFT_M]
                        // 31-7=24, so in_TDATA[24:18]
                        i_memory[load_count] <= in_TDATA[31 : 18 + SHIFT_M];
                        
                        // Q value storage (shifted)
                        // Q is [13:0]. Shifted slice is [13 : SHIFT_M]
                        // 17-7=10, so in_TDATA[10:4]
                        q_memory[load_count] <= in_TDATA[13 : SHIFT_M];
                        
                        load_count <= load_count + 1;
                    end
                end
                
                // Manage Computation and Store Counters
                S_COMPUTE: compute_count <= (next_state == S_STORE) ? 0 : compute_count + 1;
                S_STORE: compute_count <= 0; // Reset compute count
                
                default: ; // Latch other counters
            endcase
            
            // Store counter logic separate for clean store WE control
            if (current_state == S_STORE && store_count < 2) begin
                store_count <= store_count + 1;
            end else if (current_state != S_STORE) begin
                store_count <= 0;
            end
        end
    end
    
    // =========================================================================
    // III. NN Accelerator (Instantiation)
    // =========================================================================
    
    // Placeholder output for the 2-bit prediction
    wire [PRED_BITS-1:0] nn_prediction_out;
    
    // Instantiate the accelerator. It uses the loaded I/Q memories as input.
    nn_accelerator #(
        .WINDOW_SIZE(WINDOW_SIZE),
        .NUM_WINDOWS(NUM_WINDOWS),
        .IQ_WIDTH_OUT(IQ_WIDTH_OUT),
        .PRED_BITS(PRED_BITS),
        .SHIFT_N(SHIFT_N) // Pass the new post-accumulation shift parameter
    )
    u_nn_accelerator (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .i_memory(i_memory),
        .q_memory(q_memory),
        .start(current_state == S_COMPUTE),
        .nn_prediction_out(nn_prediction_out) // 2-bit prediction output
    );
    
    // =========================================================================
    // IV. BRAM Storage Logic
    // =========================================================================
    
    // BRAM Output Assignment
    assign out_ADDR = pred_addr_reg; // The address is set at the start of the transaction
    
    always @(posedge ap_clk or negedge ap_rst_n) begin
        if (!ap_rst_n) begin
            out_WE <= 0;
            out_DATA <= 0;
        end else begin
            out_WE <= 0; // Default low
            
            case (current_state)
                S_COMPUTE: begin
                    // Latch the result from the accelerator once it's computed
                    if (next_state == S_STORE) begin
                        out_DATA <= nn_prediction_out;
                    end
                end
                
                S_STORE: begin
                    // BRAM write sequence (2 clock cycles)
                    if (store_count == 0) begin // Cycle 1
                        out_WE <= 1;
                    end
                    // store_count=1 (Cycle 2)
                    // next_state transitions to S_IDLE after store_count=2 completes
                end
                
                default: ;
            endcase
        end
    end

    // Dummy assignment for unused config signals (for HLS block compatibility)
    assign config_AWREADY = 1'b0; 

endmodule

// =========================================================================
// NN ACCELERATOR MODULE (Fleshed Out)
// This module implements the data partitioning, adder trees, interleaving,
// and the LogicNet IP interface.
// =========================================================================

module nn_accelerator #(
    parameter WINDOW_SIZE  = 400, // UPDATED: Total signal length
    parameter NUM_WINDOWS  = 2,
    parameter IQ_WIDTH_OUT = 7,   // UPDATED: 14 - SHIFT_M=7
    parameter PRED_BITS    = 2,
    parameter SHIFT_N      = 1    // UPDATED: Post-accumulation shift is now 1
)(
    input  wire ap_clk,
    input  wire ap_rst_n,
    input  wire start,
    // The memory arrays are accessed directly here
    input  wire [IQ_WIDTH_OUT-1:0] i_memory [0:WINDOW_SIZE-1],
    input  wire [IQ_WIDTH_OUT-1:0] q_memory [0:WINDOW_SIZE-1],
    
    output wire [PRED_BITS-1:0] nn_prediction_out
);
    

    // User requested to assume 2 windows for this stage's logic
    localparam ACCEL_NUM_WINDOWS = NUM_WINDOWS; 
    localparam SEGMENT_SIZE      = WINDOW_SIZE / ACCEL_NUM_WINDOWS; // 400 / 2 = 200
    
    // Raw output width of sum_signed: W=7. Output width is W+9-1 = 15 bits.
    localparam RAW_SUM_WIDTH     = IQ_WIDTH_OUT + 8; // 7 + 8 = 15 bits
    
    // Final width after applying SHIFT_N truncation
    localparam FINAL_SUM_WIDTH   = RAW_SUM_WIDTH - SHIFT_N; // 15 - 1 = 14 bits (UPDATED)
    
    localparam LOGICNET_IN_WIDTH = 56;                               // From logicnet IP definition
    localparam INTERLEAVED_WIDTH = ACCEL_NUM_WINDOWS * 2 * FINAL_SUM_WIDTH; // 2 * 2 * 14 = 56 bits (UPDATED)
    localparam PADDING_WIDTH     = LOGICNET_IN_WIDTH - INTERLEAVED_WIDTH; // 300 - 56 = 244 bits (UPDATED)

    // Wires for raw sum outputs (4 total sums, 15 bits wide)
    wire signed [RAW_SUM_WIDTH-1:0] sum_raw_i0, sum_raw_q0, sum_raw_i1, sum_raw_q1;
    
    // Define array wires for the adder inputs (unpacked for sum_signed IP)
    genvar i;
    // Input width W=7
    wire signed [IQ_WIDTH_OUT-1:0] inp_i0 [0:SEGMENT_SIZE-1];
    wire signed [IQ_WIDTH_OUT-1:0] inp_q0 [0:SEGMENT_SIZE-1];
    wire signed [IQ_WIDTH_OUT-1:0] inp_i1 [0:SEGMENT_SIZE-1];
    wire signed [IQ_WIDTH_OUT-1:0] inp_q1 [0:SEGMENT_SIZE-1];

    // --- 1. Data Partitioning and Concurrent Assignment ---
    generate
        for (i = 0; i < SEGMENT_SIZE; i = i + 1) begin : segment_connect
            // Window 0 (I and Q)
            assign inp_i0[i] = i_memory[i];
            assign inp_q0[i] = q_memory[i];

            // Window 1 (I and Q)
            assign inp_i1[i] = i_memory[i + SEGMENT_SIZE];
            assign inp_q1[i] = q_memory[i + SEGMENT_SIZE];
        end
    endgenerate

    // --- 2. Instantiate Adder Trees (sum_signed IP) ---

    // I Window 0
    sum_signed #(
        .M(SEGMENT_SIZE), // 200
        .W(IQ_WIDTH_OUT)  // 7
    ) u_sum_signed_i0 (
        .clk(ap_clk),
        .rst(~ap_rst_n),
        .inp_vld(start),
        .outp_vld(),
        .inp(inp_i0),
        .sum(sum_raw_i0)
    );

    // Q Window 0
    sum_signed #(
        .M(SEGMENT_SIZE), // 200
        .W(IQ_WIDTH_OUT)  // 7
    ) u_sum_signed_q0 (
        .clk(ap_clk),
        .rst(~ap_rst_n),
        .inp_vld(start),
        .outp_vld(),
        .inp(inp_q0),
        .sum(sum_raw_q0)
    );
    
    // I Window 1
    sum_signed #(
        .M(SEGMENT_SIZE), // 200
        .W(IQ_WIDTH_OUT)  // 7
    ) u_sum_signed_i1 (
        .clk(ap_clk),
        .rst(~ap_rst_n),
        .inp_vld(start),
        .outp_vld(),
        .inp(inp_i1),
        .sum(sum_raw_i1)
    );

    // Q Window 1
    sum_signed #(
        .M(SEGMENT_SIZE), // 200
        .W(IQ_WIDTH_OUT)  // 7
    ) u_sum_signed_q1 (
        .clk(ap_clk),
        .rst(~ap_rst_n),
        .inp_vld(start),
        .outp_vld(),
        .inp(inp_q1),
        .sum(sum_raw_q1)
    );

    // --- 3. Post-Accumulation Shift (SHIFT_N) and Interleaving ---
    
    // Apply SHIFT_N truncation (dropping SHIFT_N LSBs)
    // New sum width is [RAW_SUM_WIDTH-1 : SHIFT_N] which is FINAL_SUM_WIDTH (14 bits) wide.
    wire signed [FINAL_SUM_WIDTH-1:0] sum_final_i0 = sum_raw_i0[RAW_SUM_WIDTH-1 : SHIFT_N];
    wire signed [FINAL_SUM_WIDTH-1:0] sum_final_q0 = sum_raw_q0[RAW_SUM_WIDTH-1 : SHIFT_N];
    wire signed [FINAL_SUM_WIDTH-1:0] sum_final_i1 = sum_raw_i1[RAW_SUM_WIDTH-1 : SHIFT_N];
    wire signed [FINAL_SUM_WIDTH-1:0] sum_final_q1 = sum_raw_q1[RAW_SUM_WIDTH-1 : SHIFT_N];

    wire [LOGICNET_IN_WIDTH-1:0] logicnet_input;

    // Interleaving: I0, Q0, I1, Q1 (56 bits total) followed by 244 bits of zero padding
    assign logicnet_input = {
        sum_final_q1,            // [55:42]
        sum_final_i1,            // [41:28]
        sum_final_q0,            // [27:14]
        sum_final_i0             // [13:0]
    };
    
    // --- 4. Instantiate LogicNet IP ---
    wire [PRED_BITS-1:0] logicnet_output;

    logicnet u_logicnet (
        .M0(logicnet_input),
        .clk(ap_clk),
        .rst(~ap_rst_n),
        .M4(logicnet_output) // 2-bit prediction output
    );

    // The overall prediction output is the LogicNet output.
    assign nn_prediction_out = logicnet_output;

endmodule

