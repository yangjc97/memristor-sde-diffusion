#include "timer.h"
#include "max5742.h"
#include "arm_math.h"
#include "charge.h"
#include "common.h"
#include "rng.h"


void TIM3_Int_Init(u16 arr,u16 psc)
{
	TIM_TimeBaseInitTypeDef TIM_TimeBaseInitStructure;
	NVIC_InitTypeDef NVIC_InitStructure;
	
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3,ENABLE);  		
	
	TIM_TimeBaseInitStructure.TIM_Period = arr; 		
	TIM_TimeBaseInitStructure.TIM_Prescaler=psc; 	 				
	TIM_TimeBaseInitStructure.TIM_CounterMode=TIM_CounterMode_Up; 
	TIM_TimeBaseInitStructure.TIM_ClockDivision=TIM_CKD_DIV1; 
	
	TIM_TimeBaseInit(TIM3,&TIM_TimeBaseInitStructure);				
	
	TIM_ITConfig(TIM3,TIM_IT_Update,ENABLE);				 	
	TIM_Cmd(TIM3,ENABLE); 											
	
	NVIC_InitStructure.NVIC_IRQChannel=TIM3_IRQn; 				
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=0x01; 		
	NVIC_InitStructure.NVIC_IRQChannelSubPriority=0x03; 		
	NVIC_InitStructure.NVIC_IRQChannelCmd=ENABLE;
	NVIC_Init(&NVIC_InitStructure);
	
}


float t = 0.0;

float g2, fx;
float t1, t2, t3, t4, t5, t6, t7;
float t8, t9, t10, t11, t12, t13, t14;

float v1, v2;

int	p = 0;
int flag_p = 0;

Eint go;

Eint open;

Eint FLAG;
Eint begin;

Efloat beta1, beta0, delta_beta;
Efloat scale;
Efloat t_coef1, t_coef2, t_coef3, t_coef4, t_coef5, t_coef6, t_coef7;

Eint type;
Efloat c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14; 

Efloat adc1_v;

Eint NoiseFlag;
Efloat noise1, noise2;

Efloat offset1, offset2;
Efloat coef;
Efloat noise_scale;


void TIM3_IRQHandler(void)
{
	if(TIM_GetITStatus(TIM3,TIM_IT_Update)==SET)
	{
			p++;          				// 5ms
		// 200 * 600, 10min refresh
		if ( p >= 120000 ){
			
			p = 0;
		}
			
		
		if (go == 1){
			go = 0;
			if ( p > 1000 ){
				flag_p = 1;
				p = 0;}}
		
		if (flag_p == 1){
			
			flag_p = 0;
			
			S3 = 1;
			S1 = 1;
			delay_us(100);
		
			begin = 1;
			FLAG = 1;
			
			S4 = 1;
			S2 = 1;
		}
		
		if (begin == 1) t = 0.0;
		
		// start
		if (FLAG == 1){
			begin = 0;

			if (t > 1.2f){
				t = 0.0f;
				fx = 0.0f;
				g2 = 0.0f;
				t1 = 0.0f;
				t2 = 0.0f;
				t3 = 0.0f;
				t4 = 0.0f;
				t5 = 0.0f;
				t6 = 0.0f;
				t7 = 0.0f;
				t8 = 0.0f;
				t9 = 0.0f;
				t10= 0.0f;
				t11= 0.0f;
				t12= 0.0f;
				t13= 0.0f;
				t14= 0.0f;
				
				// end
				FLAG = 0;
				S4 = 0;
				S2 = 0;
				delay_us(100);	
				S3 = 0;
				S1 = 0;
			}
			else{
				fx = - 0.25f * ( (1-t)*delta_beta + beta0 );
				g2 =   0.5f  * ( (1-t)*delta_beta + beta0 ) * coef;
				t1 = scale * (arm_sin_f32( 2*PI * t_coef1 * (1-t) ) + c1);
				t2 = scale * (arm_sin_f32( 2*PI * t_coef2 * (1-t) ) + c2);
				t3 = scale * (arm_sin_f32( 2*PI * t_coef3 * (1-t) ) + c3);
				t4 = scale * (arm_sin_f32( 2*PI * t_coef4 * (1-t) ) + c4);
				t5 = scale * (arm_sin_f32( 2*PI * t_coef5 * (1-t) ) + c5);
				t6 = scale * (arm_sin_f32( 2*PI * t_coef6 * (1-t) ) + c6);
				t7 = scale * (arm_sin_f32( 2*PI * t_coef7 * (1-t) ) + c7);
				
				t8 = scale * (arm_cos_f32( 2*PI * t_coef1 * (1-t) ) + c8);
				t9 = scale * (arm_cos_f32( 2*PI * t_coef2 * (1-t) ) + c9);
				t10= scale * (arm_cos_f32( 2*PI * t_coef3 * (1-t) ) + c10);
				t11= scale * (arm_cos_f32( 2*PI * t_coef4 * (1-t) ) + c11);
				t12= scale * (arm_cos_f32( 2*PI * t_coef5 * (1-t) ) + c12);
				t13= scale * (arm_cos_f32( 2*PI * t_coef6 * (1-t) ) + c13);
				t14= scale * (arm_cos_f32( 2*PI * t_coef7 * (1-t) ) + c14);
			}
			
			t += 0.005f;  		// 5ms
	
			max5742_A( (uint16_t) 2048*(1 + g2 ), 9);
			max5742_B( (uint16_t) 2048*(1 + fx ), 9);
			
			max5742_A( (uint16_t) 2048*(1 + t1 ), 1);
			max5742_B( (uint16_t) 2048*(1 + t2 ), 1);
			max5742_C( (uint16_t) 2048*(1 + t3 ), 1);
			max5742_D( (uint16_t) 2048*(1 + t4 ), 1);
			
			max5742_A( (uint16_t) 2048*(1 + t5 ), 2);
			max5742_B( (uint16_t) 2048*(1 + t6 ), 2);
			max5742_C( (uint16_t) 2048*(1 + t7 ), 2);
			max5742_D( (uint16_t) 2048*(1 + t8 ), 2);
			
			max5742_A( (uint16_t) 2048*(1 + t9 ), 3);
			max5742_B( (uint16_t) 2048*(1 + t10), 3);
			max5742_C( (uint16_t) 2048*(1 + t11), 3);
			max5742_D( (uint16_t) 2048*(1 + t12), 3);
			
			max5742_A( (uint16_t) 2048*(1 + t13), 4);
			max5742_B( (uint16_t) 2048*(1 + t14), 4);
			
			if (NoiseFlag==1){
				noise1 = gauss_my(0, noise_scale)*( (1-t)*delta_beta + beta0 ) + offset1;   // ADC OFFSET
				noise2 = gauss_my(0, noise_scale)*( (1-t)*delta_beta + beta0 ) + offset2;
				max5742_C( (uint16_t) 2048*(1 + noise1), 9);
				max5742_D( (uint16_t) 2048*(1 + noise2), 9);
			}
			else{
				noise1 = offset1;
				noise2 = offset2;
				max5742_C( (uint16_t) 2048*(1 + noise1), 9);
				max5742_D( (uint16_t) 2048*(1 + noise2), 9);
			}
			
		}	
	}
	
	TIM_ClearITPendingBit(TIM3,TIM_IT_Update);
}
