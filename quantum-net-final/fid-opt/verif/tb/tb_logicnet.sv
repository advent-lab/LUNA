`timescale 1ns / 1ps

// ===========================================================================
// tb_logicnet.sv  —  fid-opt LogicNet core testbench
//
// Drives pre-computed 56-bit M0 words from logicnet_stimulus.hex into the
// 4-stage pipelined `logicnet` module and computes fidelity.
//
// Pipeline: M0 -> layer0_reg -> layer0 -> layer1_reg -> layer1 ->
//           layer2_reg -> layer2 -> layer3_reg -> layer3 -> M4
// LATENCY = 4 register stages (confirmed from logicnet.v)
//
// Class decode: M4[1] = 1  =>  class 1
//               M4[1] = 0  =>  class 0
//
// Fidelity = 1 - (P(1|0) + P(0|1)) / 2
//
// Generate stimulus first:
//   python3 generate_tb_data.py
// Then run:
//   bash run_tb.sh
// ===========================================================================

module tb_logicnet;

    // ---- Parameters ----
    localparam integer N_SAMPLES  = 10000;
    localparam integer LATENCY    = 4;
    localparam integer M0W        = 56;
    localparam integer CLK_HALF_NS = 3;   // 167 MHz

    // ---- DUT ----
    reg              clk = 0;
    reg              rst = 0;
    reg [M0W-1:0]    M0  = 0;
    wire [1:0]       M4;

    always #CLK_HALF_NS clk = ~clk;

    logicnet dut (
        .clk (clk),
        .rst (rst),
        .M0  (M0),
        .M4  (M4)
    );

    // ---- Memories ----
    reg [M0W-1:0]  stim_mem  [0:N_SAMPLES-1];
    reg [0:0]      label_mem [0:N_SAMPLES-1];

    // ---- Fidelity counters ----
    integer n0, n1, err01, err10;
    integer label_i, pred_i;
    real    p01, p10, fid;
    integer k, di, j;

    // ---- Reset: 8 clocks asserted, 4 deasserted ----
    task do_reset;
        begin
            rst = 1; M0 = 0;
            for (j = 0; j < 8; j = j+1) @(posedge clk);
            rst = 0;
            for (j = 0; j < 4; j = j+1) @(posedge clk);
        end
    endtask

    // ---- Main ----
    initial begin
        $readmemh("logicnet_stimulus.hex", stim_mem);
        $readmemh("labels.hex",            label_mem);

        $display("=== tb_logicnet  [fid-opt, fidelity config] ===");
        $display("N=%0d  LATENCY=%0d  M0=%0d-bit  (FEAT_BW=14, [I0,I1,Q0,Q1])",
                 N_SAMPLES, LATENCY, M0W);

        do_reset;

        // --- Pre-fill: drive the first LATENCY inputs ---
        for (j = 0; j < LATENCY; j = j+1) begin
            M0 = stim_mem[j];
            @(posedge clk);
        end

        // --- Stream + capture ---
        // Each posedge: drive stim[k+LATENCY], then sample M4 which reflects stim[k]
        n0 = 0; n1 = 0; err01 = 0; err10 = 0;
        for (k = 0; k < N_SAMPLES; k = k+1) begin
            di = k + LATENCY;
            M0 = (di < N_SAMPLES) ? stim_mem[di] : 0;
            @(posedge clk);

            // Classify: M4[1] is the class bit
            pred_i  = (M4 >> 1) & 1;
            label_i = label_mem[k] & 1;

            if (label_i == 0) begin
                n0 = n0 + 1;
                if (pred_i == 1) err01 = err01 + 1;
            end else begin
                n1 = n1 + 1;
                if (pred_i == 0) err10 = err10 + 1;
            end

            // Print first 5 for visual check
            if (k < 5)
                $display("  [%0d] M0=0x%014h  M4=%0b  pred=%0d  label=%0d",
                         k, stim_mem[k], M4, pred_i, label_i);
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
