#include "common.h"

void GPIO_group_OUT(_gpio_group *group,u16 outdata)
{
  u8 t;
	for(t=0;t<16;t++)
    {               
        if((outdata&0x8000)>>15)  
				{
						switch(t)
						{
								case 0:	   group->data15=1; break;
								case 1:	   group->data14=1; break;
								case 2:	   group->data13=1; break;
								case 3:	   group->data12=1; break;
								case 4:	   group->data11=1; break;
								case 5:	   group->data10=1; break;
								case 6:	   group->data9=1;  break;
								case 7:	   group->data8=1;  break;
								case 8:	   group->data7=1;  break;
								case 9:	   group->data6=1;  break;
								case 10:	 group->data5=1;  break;
								case 11:	 group->data4=1;  break;
								case 12:	 group->data3=1;  break;
								case 13:	 group->data2=1;  break;
								case 14:	 group->data1=1;  break;
								case 15:	 group->data0=1;  break;
						}
				}
				else
				{
				  switch(t)
						{
								case 0:	   group->data15=0; break;
								case 1:	   group->data14=0; break;
								case 2:	   group->data13=0; break;
								case 3:	   group->data12=0; break;
								case 4:	   group->data11=0; break;
								case 5:	   group->data10=0; break;
								case 6:	   group->data9=0;  break;
								case 7:	   group->data8=0;  break;
								case 8:	   group->data7=0;  break;
								case 9:	   group->data6=0;  break;
								case 10:	 group->data5=0;  break;
								case 11:	 group->data4=0;  break;
								case 12:	 group->data3=0;  break;
								case 13:	 group->data2=0;  break;
								case 14:	 group->data1=0;  break;
								case 15:	 group->data0=0;  break;
						}
				}
     outdata<<=1; 	
	  }
}

void GPIO_bits_OUT(GPIO_TypeDef* GPIOx, u8 start_bit, u8 bit_size,u16 outdata)
{
  u8 i=0;
	u16 bu1=0;u16 middata=1;

	if( bit_size>(16-start_bit) ) 
     bit_size=16-start_bit;
	
	i=start_bit;
	if(i>0)
		 {
			 while(i--)
         { bu1+=middata; middata*=2;}
		 }
	
   assert_param(IS_GPIO_ALL_PERIPH(GPIOx));
   
	 GPIOx->ODR&=(  ( (0xffff<<bit_size) <<start_bit ) |bu1   ); 
	 GPIOx->ODR|=(outdata<<start_bit);		 
}

 
__asm void WFI_SET(void)
{
	WFI;		  
}

__asm void INTX_DISABLE(void)
{
	CPSID   I
	BX      LR	  
}

__asm void INTX_ENABLE(void)
{
	CPSIE   I
	BX      LR  
}

__asm void MSR_MSP(u32 addr) 
{
	MSR MSP, r0 			//set Main Stack value
	BX r14
}


static u8  fac_us=0;    
static u16 fac_ms=0; 

void delay_init()
{
 	SysTick_CLKSourceConfig(SysTick_CLKSource_HCLK_Div8);
	fac_us=SYSCLK/8;	 
	fac_ms=(u16)fac_us*1000; 
}								    

void delay_us(u32 nus)
{		
	u32 midtime;	    	 
	SysTick->LOAD=nus*fac_us;	  		 
	SysTick->VAL=0x00;     
	SysTick->CTRL|=SysTick_CTRL_ENABLE_Msk ; 
	do
	{
		midtime=SysTick->CTRL;
	}
	while((midtime&0x01)&&!(midtime&(1<<16)));
	SysTick->CTRL&=~SysTick_CTRL_ENABLE_Msk;
	SysTick->VAL =0X00; 
}

void delay_xms(u16 nms)
{	 		  	  
	u32 midtime;		   
	SysTick->LOAD=(u32)nms*fac_ms;
	SysTick->VAL =0x00;  
	SysTick->CTRL|=SysTick_CTRL_ENABLE_Msk ;     
	do
	{
		midtime=SysTick->CTRL;
	}
	while((midtime&0x01)&&!(midtime&(1<<16)));
	SysTick->CTRL&=~SysTick_CTRL_ENABLE_Msk;    
	SysTick->VAL =0X00; 	    
} 


void delay_ms(u16 nms)
{	 	 
	u8 repeat=nms/540;	
	u16 remain=nms%540;
	while(repeat)
	{
		delay_xms(540);
		repeat--;
	}
	if(remain)delay_xms(remain);
} 

			 
