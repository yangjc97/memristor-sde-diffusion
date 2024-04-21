#include "rng.h"
#include "common.h"

u8 RNG_Init(void)
{
	u16 retry=0; 
	
  RCC_AHB2PeriphClockCmd(RCC_AHB2Periph_RNG, ENABLE);
	
	RNG_Cmd(ENABLE);
	
	while(RNG_GetFlagStatus(RNG_FLAG_DRDY)==RESET&&retry<10000)	
	{
		retry++;
		delay_us(100);
	}
	if(retry>=10000)return 1;
	return 0;
}

u32 RNG_Get_RandomNum(void)
{	 
	while(RNG_GetFlagStatus(RNG_FLAG_DRDY)==RESET);
	return RNG_GetRandomNumber();	
}

int RNG_Get_RandomRange(int min,int max)
{ 
   return RNG_Get_RandomNum()%(max-min+1) +min;
}

double uniform(double a,double b,long int* seed)
{
	double t;
	*seed = 2045 * (*seed) + 1;
	*seed = *seed - (*seed/1048576)*1048576;
	t = (*seed)/1048576.0;
	t = a + (b - a) * t;
	return t;
}


double gauss(double mean,double sigma,long int* seed)
{
	int i;
	double x,y;
 
	for(x=0,i=0;i<12;i++)
		x = x + uniform(0.0,1.0,seed);
		x = x - 6.0;
		y = mean + x * sigma;
		return y;
}

double gauss_my(float mean, float sigma)
{
	int i;
	float x,y;
 
	for(x=0,i=0;i<12;i++)
		x = x + RNG_Get_RandomRange(1, 8192) / 8192.0f;
		x = x - 6.0f;
		y = mean + x * sigma;
		return y;
}



