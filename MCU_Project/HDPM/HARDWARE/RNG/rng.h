#ifndef __RNG_H
#define __RNG_H	 
#include "common.h" 

	
u8  RNG_Init(void);	
u32 RNG_Get_RandomNum(void);
int RNG_Get_RandomRange(int min,int max);


double uniform(double a,double b,long int* seed);
double gauss(double mean,double sigma,long int* seed);

double gauss_my(float mean,float sigma);

#endif

















