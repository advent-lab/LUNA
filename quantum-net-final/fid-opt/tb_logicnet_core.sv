`timescale 1ns / 1ps

// ===========================================================================
// tb_logicnet_core.sv  (fid-opt)
// ---------------------------------------------------------------------------
// Tests both packing orderings against the fid-opt logicnet, mirrors
// test_logicnets_core.py + ref_model.py.
//
// PASS 0 — 'training': {I0,I1,Q0,Q1} MSB-first
// PASS 1 — 'rtl'     : {Q1,I1,Q0,I0} MSB-first
//
// Pipeline latency = 4 clocks (4 myreg stages in logicnet.v).
// Class bit = M4[1] (as confirmed by layer3_N0 ROM: 10=class1, 00=class0).
// ===========================================================================

module tb_logicnet_core;

    localparam integer N_SAMPLES  = 10000;
    localparam integer LATENCY    = 4;
    localparam integer M0W        = 56;
    localparam integer CLK_HALF   = 3;   // 3 ns half-period = 167 MHz

    // ---- DUT signals ----
    reg              clk  = 0;
    reg              rst  = 0;
    reg  [M0W-1:0]   M0   = 0;
    wire [1:0]       M4;

    always #CLK_HALF clk = ~clk;

    logicnet dut (.clk(clk), .rst(rst), .M0(M0), .M4(M4));

    // ---- memories ----
    reg [M0W-1:0]  training_mem [0:N_SAMPLES-1];
    reg [M0W-1:0]  rtl_mem      [0:N_SAMPLES-1];
    reg [0:0]      labels_mem   [0:N_SAMPLES-1];
    integer        preds         [0:N_SAMPLES-1];

    // ---- counters for fidelity ----
    integer n0, n1, err01, err10;
    integer label_i, pred_i;
    real    p01, p10, fid;

    // ---- loop vars ----
    integer k, di;

    // ---- reset: 8 clocks asserted, 4 deasserted ----
    task do_reset;
        integer j;
        begin
            rst = 1; M0 = 0;
            for (j = 0; j < 8; j = j+1) @(posedge clk);
            rst = 0;
            for (j = 0; j < 4; j = j+1) @(posedge clk);
        end
    endtask

    // ---- stream one memory, collect predictions ----
    // Mirrors run_one_ordering() in test_logicnets_core.py:
    //   pre-fill LATENCY cycles, then for each k drive k+LATENCY and sample M4.
    task run_and_capture;
        input [M0W-1:0] mem [0:N_SAMPLES-1];
        integer i;
        begin
            // Pre-fill pipeline
            for (i = 0; i < LATENCY; i = i+1) begin
                if (i < N_SAMPLES) M0 = mem[i];
                @(posedge clk);
            end
            // Stream + sample
            for (k = 0; k < N_SAMPLES; k = k+1) begin
                di = k + LATENCY;
                if (di < N_SAMPLES) M0 = mem[di];
                else                M0 = 0;
                @(posedge clk);
                preds[k] = {30'b0, M4};
            end
            // Drain
            for (i = 0; i < LATENCY + 2; i = i+1) @(posedge clk);
        end
    endtask

    // ---- fidelity calculation (matches ref_model.fidelity) ----
    // class = preds[k][1]  (M4[1] is the class indicator)
    task calc_and_print_fidelity;
        input [127:0] pass_name;  // string passed as packed bits, display with %0s
        integer n;
        begin
            n0 = 0; n1 = 0; err01 = 0; err10 = 0;
            for (n = 0; n < N_SAMPLES; n = n+1) begin
                label_i = labels_mem[n] & 1;
                pred_i  = (preds[n] >> 1) & 1;
                if (label_i == 0) begin
                    n0 = n0 + 1;
                    if (pred_i == 1) err01 = err01 + 1;
                end else begin
                    n1 = n1 + 1;
                    if (pred_i == 0) err10 = err10 + 1;
                end
            end
            p01 = (n0 > 0) ? $itor(err01) / $itor(n0) : 0.0;
            p10 = (n1 > 0) ? $itor(err10) / $itor(n1) : 0.0;
            fid = 1.0 - (p01 + p10) / 2.0;
            $display("  [%0s]  N0=%0d N1=%0d  P(1|0)=%.4f  P(0|1)=%.4f  Fidelity=%.3f%%",
                     pass_name, n0, n1, p01, p10, fid*100.0);
        end
    endtask

    // ---- test sequence ----
    initial begin
        $readmemh("training_stimulus.hex", training_mem);
        $readmemh("rtl_stimulus.hex",      rtl_mem);
        $readmemh("labels.hex",            labels_mem);

        $display("=== tb_logicnet_core (fid-opt, fidelity config) ===");
        $display("N=%0d  LATENCY=%0d  M0=%0d-bit", N_SAMPLES, LATENCY, M0W);

        // PASS 0: training ordering
        $display("--- PASS 0: training  {I0,I1,Q0,Q1} MSB-first ---");
        do_reset;
        run_and_capture(training_mem);
        calc_and_print_fidelity("training");

        // PASS 1: RTL ordering
        $display("--- PASS 1: rtl       {Q1,I1,Q0,I0} MSB-first ---");
        do_reset;
        run_and_capture(rtl_mem);
        calc_and_print_fidelity("rtl");

        $display("=== Done === SW target: 95.995%%");
        $finish;
    end

endmodule
