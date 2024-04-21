#include "max5742.h" 
#include "common.h"

void dac_spi_1(int16_t data){
	int i, j;

	CS_1 = 0;
	for (i=0; i<16; i++){
		j = 15 - i;
		if (data & bit(j))	DIN_1 = 1;
		else				DIN_1 = 0;
		CLK_1 = 1;
		delay_us(DELAY_dac);
		CLK_1 = 0;
		delay_us(DELAY_dac);}
	CS_1 = 1;
}

void dac_spi_2(int16_t data){
	int i, j;

	CS_2 = 0;
	for (i=0; i<16; i++){
		j = 15 - i;
		if (data & bit(j))	DIN_2 = 1;
		else				DIN_2 = 0;
		CLK_2 = 1;
		delay_us(DELAY_dac);
		CLK_2 = 0;
		delay_us(DELAY_dac);}
	CS_2 = 1;
}

void dac_spi_3(int16_t data){
	int i, j;

	CS_3 = 0;
	for (i=0; i<16; i++){
		j = 15 - i;
		if (data & bit(j))	DIN_3 = 1;
		else				DIN_3 = 0;
		CLK_3 = 1;
		delay_us(DELAY_dac);
		CLK_3 = 0;
		delay_us(DELAY_dac);}
	CS_3 = 1;
}

void dac_spi_4(int16_t data){
	int i, j;

	CS_4 = 0;
	for (i=0; i<16; i++){
		j = 15 - i;
		if (data & bit(j))	DIN_4 = 1;
		else				DIN_4 = 0;
		CLK_4 = 1;
		delay_us(DELAY_dac);
		CLK_4 = 0;
		delay_us(DELAY_dac);}
	CS_4 = 1;
}

void dac_spi_5(int16_t data){
	int i, j;

	CS_5 = 0;
	for (i=0; i<16; i++){
		j = 15 - i;
		if (data & bit(j))	DIN_5 = 1;
		else				DIN_5 = 0;
		CLK_5 = 1;
		delay_us(DELAY_dac);
		CLK_5 = 0;
		delay_us(DELAY_dac);}
	CS_5 = 1;
}

void dac_spi_6(int16_t data){
	int i, j;

	CS_6 = 0;
	for (i=0; i<16; i++){
		j = 15 - i;
		if (data & bit(j))	DIN_6 = 1;
		else				DIN_6 = 0;
		CLK_6 = 1;
		delay_us(DELAY_dac);
		CLK_6 = 0;
		delay_us(DELAY_dac);}
	CS_6 = 1;
}

void dac_spi_7(int16_t data){
	int i, j;

	CS_7 = 0;
	for (i=0; i<16; i++){
		j = 15 - i;
		if (data & bit(j))	DIN_7 = 1;
		else				DIN_7 = 0;
		CLK_7 = 1;
		delay_us(DELAY_dac);
		CLK_7 = 0;
		delay_us(DELAY_dac);}
	CS_7 = 1;
}

void dac_spi_8(int16_t data){
	int i, j;

	CS_8 = 0;
	for (i=0; i<16; i++){
		j = 15 - i;
		if (data & bit(j))	DIN_8 = 1;
		else				DIN_8 = 0;
		CLK_8 = 1;
		delay_us(DELAY_dac);
		CLK_8 = 0;
		delay_us(DELAY_dac);}
	CS_8 = 1;
}

void dac_spi_9(int16_t data){
	int i, j;

	CS_9 = 0;
	for (i=0; i<16; i++){
		j = 15 - i;
		if (data & bit(j))	DIN_9 = 1;
		else				DIN_9 = 0;
		CLK_9 = 1;
		delay_us(DELAY_dac);
		CLK_9 = 0;
		delay_us(DELAY_dac);}
	CS_9 = 1;
}

void dac_spi_10(int16_t data){
	int i, j;

	CS_10 = 0;
	for (i=0; i<16; i++){
		j = 15 - i;
		if (data & bit(j))	DIN_10 = 1;
		else				DIN_10 = 0;
		CLK_10 = 1;
		delay_us(DELAY_dac);
		CLK_10 = 0;
		delay_us(DELAY_dac);}
	CS_10 = 1;
}


// Initialization  power-up 4¸öDAC
void max5742_Init(void)  {
	dac_spi_1(0xF010);
	dac_spi_2(0xF010);
	dac_spi_3(0xF010);
	dac_spi_4(0xF010);
	dac_spi_5(0xF010);
	dac_spi_6(0xF010);
	dac_spi_7(0xF010);
	dac_spi_8(0xF010);
	dac_spi_9(0xF010);
	dac_spi_10(0xF010);
}

// Output DAC
void max5742_A(uint16_t x, int idx)	{
	switch (idx){
		case 1: dac_spi_1(x | 0x0000); break;
		case 2: dac_spi_2(x | 0x0000); break;
		case 3: dac_spi_3(x | 0x0000); break;
		case 4: dac_spi_4(x | 0x0000); break;
		case 5: dac_spi_5(x | 0x0000); break;
		case 6: dac_spi_6(x | 0x0000); break;
		case 7: dac_spi_7(x | 0x0000); break;
		case 8: dac_spi_8(x | 0x0000); break;
		case 9: dac_spi_9(x | 0x0000); break;
		case 10: dac_spi_10(x | 0x0000); break;
	}
}

void max5742_B(uint16_t x, int idx)	{
	switch (idx){
		case 1: dac_spi_1(x | 0x1000); break;
		case 2: dac_spi_2(x | 0x1000); break;
		case 3: dac_spi_3(x | 0x1000); break;
		case 4: dac_spi_4(x | 0x1000); break;
		case 5: dac_spi_5(x | 0x1000); break;
		case 6: dac_spi_6(x | 0x1000); break;
		case 7: dac_spi_7(x | 0x1000); break;
		case 8: dac_spi_8(x | 0x1000); break;
		case 9: dac_spi_9(x | 0x1000); break;
		case 10: dac_spi_10(x | 0x1000); break;
	}
}

void max5742_C(uint16_t x, int idx)	{
	switch (idx){
		case 1: dac_spi_1(x | 0x2000); break;
		case 2: dac_spi_2(x | 0x2000); break;
		case 3: dac_spi_3(x | 0x2000); break;
		case 4: dac_spi_4(x | 0x2000); break;
		case 5: dac_spi_5(x | 0x2000); break;
		case 6: dac_spi_6(x | 0x2000); break;
		case 7: dac_spi_7(x | 0x2000); break;
		case 8: dac_spi_8(x | 0x2000); break;
		case 9: dac_spi_9(x | 0x2000); break;
		case 10: dac_spi_10(x | 0x2000); break;
	}
}

void max5742_D(uint16_t x, int idx)	{
	switch (idx){
		case 1: dac_spi_1(x | 0x3000); break;
		case 2: dac_spi_2(x | 0x3000); break;
		case 3: dac_spi_3(x | 0x3000); break;
		case 4: dac_spi_4(x | 0x3000); break;
		case 5: dac_spi_5(x | 0x3000); break;
		case 6: dac_spi_6(x | 0x3000); break;
		case 7: dac_spi_7(x | 0x3000); break;
		case 8: dac_spi_8(x | 0x3000); break;
		case 9: dac_spi_9(x | 0x3000); break;
		case 10: dac_spi_10(x | 0x3000); break;
	}
}




