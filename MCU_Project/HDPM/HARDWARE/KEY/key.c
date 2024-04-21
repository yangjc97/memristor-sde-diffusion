#include "key.h"


u8  keydown_data=0x00;  
u8  keyup_data=0x00; 
u16  key_time=0x00;     

u8  key_tem=0x00;  
u8  key_bak=0x00; 

void KEY_Init(void)
{
	GPIO_InitTypeDef  GPIO_InitStructure;

  RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOF, ENABLE);
 
  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6|GPIO_Pin_7|GPIO_Pin_8|GPIO_Pin_9; //KEY0 KEY1 KEY2 KEY3
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN; 
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;       
  GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_UP;         
  GPIO_Init(GPIOF, &GPIO_InitStructure);                  

void key_scan(u8 mode)
{	   
	 keyup_data=0;  
	
	if(KEY0==0||KEY1==0||KEY2==0||KEY3==0)   
	{
		if(KEY0==0)      key_tem=1;
		else if(KEY1==0) key_tem=2;
		else if(KEY2==0) key_tem=3;
		
		  if (key_tem == key_bak) 
			  {
				   key_time++;            
					 keydown_data=key_tem; 
					
					 if( (mode==0)&&(key_time>1) )
					    keydown_data=0;
       	}
	    else                  
	      {
		       key_time=0;
		       key_bak=key_tem;
	      }
		
	}
	else      
		{
	     if(key_time>2)
	        {
		        keyup_data=key_tem;         						
	       	}
				key_bak=0;            
	      key_time=0;
				keydown_data=0;		
	  }    
}
