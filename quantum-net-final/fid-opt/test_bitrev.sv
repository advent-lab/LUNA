module tb;
    reg [3:0] orig = 4'b1100;
    wire [0:3] rev1 = orig;
    wire [3:0] rev2;
    assign rev2 = {<<{orig}};
    initial begin
        $display("orig = %b", orig);
        $display("rev1 = %b (using [0:3])", rev1);
        $display("rev2 = %b (using {<<{orig}})", rev2);
        $finish;
    end
endmodule
