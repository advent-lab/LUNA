module layer3_N0 ( input [4:0] M0, output [0:0] M1 );

	(*rom_style = "distributed" *) reg [0:0] M1r;
	assign M1 = M1r;
	always @ (M0) begin
		case (M0)
			5'b00000: M1r = 1'b1;
			5'b10000: M1r = 1'b1;
			5'b01000: M1r = 1'b0;
			5'b11000: M1r = 1'b0;
			5'b00100: M1r = 1'b1;
			5'b10100: M1r = 1'b1;
			5'b01100: M1r = 1'b1;
			5'b11100: M1r = 1'b1;
			5'b00010: M1r = 1'b0;
			5'b10010: M1r = 1'b0;
			5'b01010: M1r = 1'b0;
			5'b11010: M1r = 1'b0;
			5'b00110: M1r = 1'b1;
			5'b10110: M1r = 1'b1;
			5'b01110: M1r = 1'b0;
			5'b11110: M1r = 1'b0;
			5'b00001: M1r = 1'b0;
			5'b10001: M1r = 1'b0;
			5'b01001: M1r = 1'b0;
			5'b11001: M1r = 1'b0;
			5'b00101: M1r = 1'b1;
			5'b10101: M1r = 1'b1;
			5'b01101: M1r = 1'b1;
			5'b11101: M1r = 1'b0;
			5'b00011: M1r = 1'b0;
			5'b10011: M1r = 1'b0;
			5'b01011: M1r = 1'b0;
			5'b11011: M1r = 1'b0;
			5'b00111: M1r = 1'b0;
			5'b10111: M1r = 1'b0;
			5'b01111: M1r = 1'b0;
			5'b11111: M1r = 1'b0;

		endcase
	end
endmodule
