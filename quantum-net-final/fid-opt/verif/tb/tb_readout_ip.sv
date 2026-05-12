`timescale 1ns / 1ps

// ===========================================================================
// tb_readout_ip.sv  —  fid-opt top-level wrapper testbench
//
// Drives the 32-bit AXI-Stream interface of `nn_classifier_wrapper` with
// 400 samples per test, monitors the BRAM write interface for the prediction,
// and computes fidelity.
//
// Generate stimulus first:
//   python3 generate_tb_data.py
// Then run:
//   bash run_tb_readout_ip.sh
// ===========================================================================

module tb_readout_ip;

    // ---- Parameters ----
    localparam integer N_SAMPLES   = 10000;
    localparam integer WINDOW_SIZE = 400;
    localparam integer CLK_HALF_NS = 3;   // 167 MHz

    // ---- System Signals ----
    reg ap_clk = 0;
    reg ap_rst_n = 0;
    reg trigger = 0;

    // ---- AXIS Input Interface ----
    reg [31:0] in_TDATA = 0;
    reg        in_TVALID = 0;

    // ---- BRAM Output Interface ----
    wire [13:0] out_ADDR;
    wire [1:0]  out_DATA;
    wire        out_WE;

    always #CLK_HALF_NS ap_clk = ~ap_clk;

    // ---- DUT ----
    nn_classifier_wrapper #(
        .WINDOW_SIZE(WINDOW_SIZE),
        .NUM_WINDOWS(2),
        .SHIFT_M(7),
        .SHIFT_N(1),
        .DATA_WIDTH(32),
        .IQ_WIDTH_IN(14),
        .PRED_BITS(2),
        .BRAM_ADDR_BITS(14)
    ) dut (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .config_AWADDR(4'b0),
        .config_AWVALID(1'b0),
        .config_AWREADY(),
        .trigger(trigger),
        .in_TDATA(in_TDATA),
        .in_TVALID(in_TVALID),
        .out_ADDR(out_ADDR),
        .out_DATA(out_DATA),
        .out_WE(out_WE)
    );

    // ---- Memories ----
    reg [31:0] axis_mem [0 : (N_SAMPLES * WINDOW_SIZE) - 1];
    reg [0:0]  label_mem [0 : N_SAMPLES - 1];

    // ---- Fidelity tracking ----
    integer n0, n1, err01, err10;
    integer label_i, pred_i;
    real    p01, p10, fid;
    int     timeout_ctr;

    // ---- Main ----
    initial begin
        $readmemh("axis_stimulus.hex", axis_mem);
        $readmemh("labels.hex",        label_mem);

        $display("=== tb_readout_ip [fid-opt] ===");
        $display("N=%0d tests, WINDOW_SIZE=%0d", N_SAMPLES, WINDOW_SIZE);

        n0 = 0; n1 = 0; err01 = 0; err10 = 0;

        // Reset
        ap_rst_n = 0;
        trigger = 0;
        in_TVALID = 0;
        repeat(10) @(posedge ap_clk);
        ap_rst_n = 1;
        repeat(5) @(posedge ap_clk);

        // Run tests
        for (int test = 0; test < N_SAMPLES; test = test + 1) begin
            label_i = label_mem[test] & 1;

            // 1. Trigger
            trigger = 1;
            @(posedge ap_clk);
            trigger = 0;

            // 2. Feed AXIS Data
            in_TVALID = 1;
            for (int t = 0; t < WINDOW_SIZE; t = t + 1) begin
                in_TDATA = axis_mem[test * WINDOW_SIZE + t];
                @(posedge ap_clk);
                // If FSM isn't in S_LOAD yet, this data wasn't accepted. 
                // Wait until it is in S_LOAD so it gets counted.
                timeout_ctr = 0;
                while (dut.current_state !== 3'b001 && timeout_ctr < 50) begin
                    @(posedge ap_clk);
                    timeout_ctr++;
                end
                if (timeout_ctr == 50) begin
                    $display("ERROR: Hang waiting for S_LOAD at test %0d! FSM current_state = %b", test, dut.current_state);
                    $finish;
                end
            end
            
            // Hold the last data valid until load_count reaches WINDOW_SIZE
            // (in case the last data wasn't clocked in yet due to cycle offsets)
            timeout_ctr = 0;
            while (dut.load_count < WINDOW_SIZE && timeout_ctr < 500) begin
                @(posedge ap_clk);
                timeout_ctr++;
            end
            if (timeout_ctr == 500) begin
                $display("ERROR: Hang waiting for load_count to reach %0d at test %0d! load_count = %0d", WINDOW_SIZE, test, dut.load_count);
                $finish;
            end
            in_TVALID = 0;
            
            // $display("Debug: test %0d data fed. load_count = %0d, state = %b", test, dut.load_count, dut.current_state);

            // 3. Wait for BRAM Write
            // The FSM takes 5 compute cycles + 1 or 2 store cycles
            // We wait until out_WE is asserted.
            timeout_ctr = 0;
            while (out_WE !== 1'b1 && timeout_ctr < 450) begin
                @(posedge ap_clk);
                timeout_ctr++;
            end
            
            if (timeout_ctr == 450) begin
                $display("ERROR: Timeout waiting for out_WE at test %0d! FSM current_state = %b, load_count = %0d", test, dut.current_state, dut.load_count);
                $finish;
            end
            
            // out_WE is now 1. Capture prediction!
            pred_i = (out_DATA >> 1) & 1; // Class is MSB of 2-bit prediction

            if (label_i == 0) begin
                n0 = n0 + 1;
                if (pred_i == 1) err01 = err01 + 1;
            end else begin
                n1 = n1 + 1;
                if (pred_i == 0) err10 = err10 + 1;
            end

            if (test < 5) begin
                $display("  [%0d] expected=%0d  got=%0d  (out_DATA=%0b)",
                         test, label_i, pred_i, out_DATA);
            end else if (test % 100 == 0) begin
                $display("  [%0d] completed...", test);
            end

            // Wait until FSM is back to IDLE
            repeat(3) @(posedge ap_clk);
        end

        // --- Fidelity report ---
        p01 = (n0 > 0) ? $itor(err01) / $itor(n0) : 0.0;
        p10 = (n1 > 0) ? $itor(err10) / $itor(n1) : 0.0;
        fid = 1.0 - (p01 + p10) / 2.0;

        $display("----------------------------------------");
        $display("N0=%0d  N1=%0d", n0, n1);
        $display("P(1|0) [FP] = %.4f %%", p01 * 100.0);
        $display("P(0|1) [FN] = %.4f %%", p10 * 100.0);
        $display("Fidelity    = %.3f %%", fid * 100.0);
        $display("SW target   = 95.995 %%");
        $display("----------------------------------------");
        $finish;
    end

endmodule
