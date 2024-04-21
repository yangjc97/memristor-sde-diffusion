#include "GPIO.h" 

void io_Init(void)
{    	 
	GPIO_InitTypeDef  GPIO_InitStructure_A, GPIO_InitStructure_B, GPIO_InitStructure_C, GPIO_InitStructure_E, GPIO_InitStructure_F, GPIO_InitStructure_G;

	// DAC_1 GPPIOA 1,2,3,4,5
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);

	GPIO_InitStructure_A.GPIO_Pin = GPIO_Pin_0|GPIO_Pin_1|GPIO_Pin_2|GPIO_Pin_3|GPIO_Pin_4|GPIO_Pin_5|GPIO_Pin_6|GPIO_Pin_7|GPIO_Pin_8|GPIO_Pin_11|GPIO_Pin_12;
	GPIO_InitStructure_A.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure_A.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure_A.GPIO_Speed = GPIO_Speed_100MHz;
	GPIO_InitStructure_A.GPIO_PuPd = GPIO_PuPd_UP;
	GPIO_Init(GPIOA, &GPIO_InitStructure_A);
	
	
	// DAC_2 GPPIOC 0,1,2,3,4
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOC, ENABLE);
	
	GPIO_InitStructure_C.GPIO_Pin = GPIO_Pin_0|GPIO_Pin_1|GPIO_Pin_2|GPIO_Pin_3|GPIO_Pin_4|GPIO_Pin_5|GPIO_Pin_6|GPIO_Pin_7|GPIO_Pin_8|GPIO_Pin_9|GPIO_Pin_10|GPIO_Pin_11|GPIO_Pin_12|GPIO_Pin_13;
	GPIO_InitStructure_C.GPIO_Mode = GPIO_Mode_OUT;	
	GPIO_InitStructure_C.GPIO_OType = GPIO_OType_PP;		
	GPIO_InitStructure_C.GPIO_Speed = GPIO_Speed_100MHz;	
	GPIO_InitStructure_C.GPIO_PuPd = GPIO_PuPd_UP;					
	GPIO_Init(GPIOC, &GPIO_InitStructure_C);					
	

	// DAC_3 GPPIOE 7,8,9,10,11
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOG, ENABLE);		
	
	GPIO_InitStructure_G.GPIO_Pin = GPIO_Pin_2|GPIO_Pin_3|GPIO_Pin_4|GPIO_Pin_5|GPIO_Pin_6|GPIO_Pin_9|GPIO_Pin_10|GPIO_Pin_11|GPIO_Pin_12;
	GPIO_InitStructure_G.GPIO_Mode = GPIO_Mode_OUT;			
	GPIO_InitStructure_G.GPIO_OType = GPIO_OType_PP;				
	GPIO_InitStructure_G.GPIO_Speed = GPIO_Speed_100MHz;		
	GPIO_InitStructure_G.GPIO_PuPd = GPIO_PuPd_UP;					
	GPIO_Init(GPIOG, &GPIO_InitStructure_G);				
	
	
	// Switch GPPIOD 1,2,3,4,5
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE);
	
	GPIO_InitStructure_B.GPIO_Pin = GPIO_Pin_0|GPIO_Pin_1|GPIO_Pin_2|GPIO_Pin_3|GPIO_Pin_4|GPIO_Pin_5|GPIO_Pin_6|GPIO_Pin_7|GPIO_Pin_8|GPIO_Pin_9|GPIO_Pin_10|GPIO_Pin_11|GPIO_Pin_12|GPIO_Pin_13;
	GPIO_InitStructure_B.GPIO_Mode = GPIO_Mode_OUT;				
	GPIO_InitStructure_B.GPIO_OType = GPIO_OType_PP;				
	GPIO_InitStructure_B.GPIO_Speed = GPIO_Speed_100MHz;			
	GPIO_InitStructure_B.GPIO_PuPd = GPIO_PuPd_UP;				
	GPIO_Init(GPIOB, &GPIO_InitStructure_B);						

	
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);
	
	GPIO_InitStructure_E.GPIO_Pin = GPIO_Pin_0|GPIO_Pin_1|GPIO_Pin_2|GPIO_Pin_3|GPIO_Pin_4|GPIO_Pin_5|GPIO_Pin_6|GPIO_Pin_7|GPIO_Pin_8|GPIO_Pin_9|GPIO_Pin_10|GPIO_Pin_11|GPIO_Pin_12|GPIO_Pin_13|GPIO_Pin_14|GPIO_Pin_15;
	GPIO_InitStructure_E.GPIO_Mode = GPIO_Mode_OUT;			
	GPIO_InitStructure_E.GPIO_OType = GPIO_OType_PP;		
	GPIO_InitStructure_E.GPIO_Speed = GPIO_Speed_100MHz;
	GPIO_InitStructure_E.GPIO_PuPd = GPIO_PuPd_UP;
	GPIO_Init(GPIOE, &GPIO_InitStructure_E);
	
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOF, ENABLE);
	
	GPIO_InitStructure_F.GPIO_Pin = GPIO_Pin_0|GPIO_Pin_1|GPIO_Pin_2|GPIO_Pin_3|GPIO_Pin_4|GPIO_Pin_5|GPIO_Pin_11|GPIO_Pin_13|GPIO_Pin_14|GPIO_Pin_15;
	GPIO_InitStructure_F.GPIO_Mode = GPIO_Mode_OUT;	
	GPIO_InitStructure_F.GPIO_OType = GPIO_OType_PP;					
	GPIO_InitStructure_F.GPIO_Speed = GPIO_Speed_100MHz;	
	GPIO_InitStructure_F.GPIO_PuPd = GPIO_PuPd_UP;					
	GPIO_Init(GPIOF, &GPIO_InitStructure_F);
	
}


