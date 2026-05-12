`timescale 1ns / 1ps

// ===========================================================================
// tb_accel.sv  —  fid-opt accelerator isolation testbench
//
// Drives the internal memories of `nn_accelerator` directly using data loaded
// from axis_stimulus.hex, simulates the adder tree latency, and compares the
// resulting `logicnet_input` (the bit-reversed concatenation of sum_final_*)
// to the `logicnet_stimulus.hex` python reference.
// ===========================================================================

module tb_accel;

    localparam integer N_SAMPLES  = 10000;
    localparam integer WINDOW_SIZE = 400;
    localparam integer CLK_HALF_NS = 3;   

    // ---- System Signals ----
    reg ap_clk = 0;
    reg ap_rst_n = 0;
    reg start = 0;

    always #CLK_HALF_NS ap_clk = ~ap_clk;

    // ---- DUT ----
    // We instantiate nn_accelerator
    reg [4:0] i_memory [0:WINDOW_SIZE-1];
    reg [4:0] q_memory [0:WINDOW_SIZE-1];
    wire [1:0] nn_prediction_out;

    nn_accelerator #(
        .WINDOW_SIZE(WINDOW_SIZE),
        .NUM_WINDOWS(2),
        .IQ_WIDTH_OUT(5),
        .PRED_BITS(2),
        .SHIFT_N(1)
    ) dut (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .start(start),
        .i_memory(i_memory),
        .q_memory(q_memory),
        .nn_prediction_out(nn_prediction_out)
    );

    // ---- Memories ----
    reg [31:0] axis_mem [0 : (N_SAMPLES * WINDOW_SIZE) - 1];
    reg [47:0] logicnet_mem [0 : N_SAMPLES - 1];
    
    integer match_count, fail_count;

    // ---- Main ----
    initial begin
        $readmemh("axis_stimulus.hex", axis_mem);
        $readmemh("logicnet_stimulus.hex", logicnet_mem);

        $display("=== tb_accel [latency-opt] ===");
        
        match_count = 0; fail_count = 0;

        // Reset
        ap_rst_n = 0;
        start = 0;
        repeat(10) @(posedge ap_clk);
        ap_rst_n = 1;
        repeat(5) @(posedge ap_clk);

        // Run tests
        for (int test = 0; test < N_SAMPLES; test = test + 1) begin
            // 1. Load memories simulating the SHIFT_M logic from readout_ip
            for (int t = 0; t < WINDOW_SIZE; t = t + 1) begin
                logic [31:0] axis_word = axis_mem[test * WINDOW_SIZE + t];
                // I is [31:18], shift right by SHIFT_M=9 means slice is [31 : 18+9] = [31:27]
                i_memory[t] = axis_word[31:27];
                // Q is [17:4], shift right by SHIFT_M=9 means slice is [17 : 4+9] = [17:13]
                q_memory[t] = axis_word[17:13];
            end

            // 2. Pulse start
            start = 1;
            @(posedge ap_clk);
            start = 0;

            // 3. Wait for adder tree latency (sum_signed has 8 stages, so 8 clock cycles)
            repeat(8) @(posedge ap_clk);

            // 4. Sample logicnet_input using hierarchical reference
            if (dut.logicnet_input === logicnet_mem[test]) begin
                match_count++;
            end else begin
                fail_count++;
                if (fail_count <= 5) begin
                    $display("MISMATCH at test %0d:", test);
                    $display("  Expected : %014x", logicnet_mem[test]);
                    $display("  Got      : %014x", dut.logicnet_input);
                end
            end

            // Allow the rest of the pipeline to flush
            repeat(10) @(posedge ap_clk);
        end

        $display("----------------------------------------");
        $display("Tests run : %0d", N_SAMPLES);
        $display("Matches   : %0d", match_count);
        $display("Failures  : %0d", fail_count);
        $display("----------------------------------------");
        $finish;
    end

endmodule
