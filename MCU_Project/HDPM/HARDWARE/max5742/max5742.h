#ifndef __MAX5742_H
#define __MAX5742_H
#include "common.h"

#define bit(X) (1<<X)

#define DELAY_dac 2

#define DIN_1   PBout(3)
#define CS_1   	PBout(4) 
#define CLK_1   PBout(5)

#define DIN_2   PBout(6)
#define CS_2   	PBout(7) 
#define CLK_2   PBout(8)

#define DIN_3   PBout(9)
#define CS_3   	PEout(0) 
#define CLK_3   PEout(1)

#define DIN_4   PBout(13)
#define CS_4   	PBout(12) 
#define CLK_4   PBout(11)

#define DIN_5   PBout(10)
#define CS_5   	PEout(15) 
#define CLK_5   PEout(14)

#define DIN_6   PEout(13)
#define CS_6   	PEout(12) 
#define CLK_6   PEout(11)

#define DIN_7   PEout(10)
#define CS_7   	PEout(9) 
#define CLK_7   PEout(8)

#define DIN_8   PFout(15)
#define CS_8   	PFout(14) 
#define CLK_8   PFout(13)

#define DIN_9   PFout(5)
#define CS_9   	PFout(4) 
#define CLK_9   PFout(3)

#define DIN_10	PFout(2)
#define CS_10   PFout(1) 
#define CLK_10  PFout(0)

void dac_spi_1(int16_t);
void dac_spi_2(int16_t);
void dac_spi_3(int16_t);
void dac_spi_4(int16_t);
void dac_spi_5(int16_t);
void dac_spi_6(int16_t);
void dac_spi_7(int16_t);
void dac_spi_8(int16_t);
void dac_spi_9(int16_t);
void dac_spi_10(int16_t);


void max5742_Init(void);
void max5742_A(uint16_t, int);
void max5742_B(uint16_t, int);
void max5742_C(uint16_t, int);
void max5742_D(uint16_t, int);

#endif
