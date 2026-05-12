`timescale 1ns / 1ps

// ===========================================================================
// tb_logicnet.sv  —  LogicNet Standalone Pipelined Testbench
// ===========================================================================

module tb_logicnet;

    // ---- Parameters ----
    localparam integer N_SAMPLES  = 10000;
    localparam integer LATENCY    = 4;
    localparam integer M0W        = 28; // area-opt LogicNets width
    localparam integer CLK_HALF_NS = 3;   // 167 MHz

    // ---- DUT ----
    reg              clk = 0;
    reg              rst_n = 0;
    reg [M0W-1:0]    M0  = 0;
    wire [0:0]       M4;

    always #CLK_HALF_NS clk = ~clk;

    logicnet dut (
        .clk(clk),
        .rst(~rst_n),
        .M0(M0),
        .M4(M4)
    );

    // ---- Memories ----
    reg [M0W-1:0] stimulus_mem [0 : N_SAMPLES - 1];
    reg [1:0]     expected_mem [0 : N_SAMPLES - 1];
    
    // ---- Pipelined Verification Queues ----
    // We will push expected labels into a queue and pop them L cycles later
    reg [1:0] expected_queue [0 : LATENCY - 1];
    integer q_head, q_tail;

    // ---- Metrics ----
    integer n0 = 0, n1 = 0;
    integer err01 = 0, err10 = 0;
    real p01, p10, fid;

    // ---- Main ----
    initial begin
        $readmemh("logicnet_stimulus.hex", stimulus_mem);
        $readmemh("labels.hex", expected_mem);

        // Reset
        rst_n = 0;
        M0 = 0;
        q_head = 0;
        q_tail = 0;
        
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        // Feed Pipeline
        for (int i = 0; i < N_SAMPLES + LATENCY; i = i + 1) begin
            
            // Pop from pipeline
            if (i >= LATENCY) begin
                integer pop_idx = i - LATENCY;
                integer label = expected_queue[q_tail];
                q_tail = (q_tail + 1) % LATENCY;
                
                // M4[0] is the class bit
                integer pred = M4[0];
                
                if (label == 0) begin
                    n0++;
                    if (pred == 1) err01++;
                end else begin
                    n1++;
                    if (pred == 0) err10++;
                end
            end
            
            // Push to pipeline
            if (i < N_SAMPLES) begin
                M0 = stimulus_mem[i];
                expected_queue[q_head] = expected_mem[i];
                q_head = (q_head + 1) % LATENCY;
            end else begin
                M0 = 0; // Flush
            end
            
            @(posedge clk);
        end
        
        // Final Output
        $display("--------------------------------");
        $display("N0=%0d  N1=%0d", n0, n1);
        
        p01 = (n0 > 0) ? $itor(err01) / $itor(n0) : 0.0;
        p10 = (n1 > 0) ? $itor(err10) / $itor(n1) : 0.0;
        fid = 1.0 - (p01 + p10) / 2.0;
        
        $display("P(1|0) [FP] = %f %%", p01 * 100.0);
        $display("P(0|1) [FN] = %f %%", p10 * 100.0);
        $display("Fidelity    = %f %%", fid * 100.0);
        $display("--------------------------------");
        
        $finish;
    end

endmodule
