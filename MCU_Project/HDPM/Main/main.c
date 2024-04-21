#include "common.h"
#include "usart.h"
#include "max5742.h" 
#include "timer.h"
#include "gpio.h"
#include "key.h"
#include "charge.h"
#include "adc.h"
#include "arm_math.h"
#include "rng.h"
#include "lcd.h"

float v_test;
uint16_t test;
u8 key;
u16 adc1, adc2, adc3, adc4;
float adc1_v, adc2_v, adc3_v, adc4_v;

int go = 0;

int open=0;

int FLAG;
int begin;

float beta1, beta0, delta_beta;
float scale;
float t_coef1, t_coef2, t_coef3, t_coef4, t_coef5, t_coef6, t_coef7;

int type=0;
float c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14;

int NoiseFlag=0;
float noise1, noise2;
long int seed1, seed2;
char str1[20];
char str2[20];

float offset1=0, offset2=0;
float coef=1;
float noise_scale=1;

float v_bias=0.1f;

int main(void)
{ 
//	u8 t;
//	u8 len;
	
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);	
	delay_init();		  				
	uart_init(9600);							
	io_Init();		          			
	max5742_Init();
	
	TIM3_Int_Init(50-1,8400-1);						  
	KEY_Init();      				
//	Adc_Init();   
	RNG_Init();
	
	v_test = 0;               						// -1 ~ +1
	test = (uint16_t) 2048*(1 + v_test);
	

	max5742_A(test, 1);
	max5742_B(test, 1);
	max5742_C(test, 1);
	max5742_D(test, 1);
	max5742_A(test, 2);
	max5742_B(test, 2);
	max5742_C(test, 2);
	max5742_D(test, 2);
	
	max5742_A(test, 3);
	max5742_B(test, 3);
	max5742_C(test, 3);
	max5742_D(test, 3);
	max5742_A(test, 4);
	max5742_B(test, 4);
	max5742_C(test, 4);
	max5742_D(test, 4);
	
	max5742_A(test, 9);
	max5742_B(test, 9);
	max5742_C(test, 9);
	max5742_D(test, 9);
	

	
	beta1 = 0.5;
	beta0 = 0.001;
	delta_beta = 0.499;

	scale = 0.5f;

//	10k ring
	t_coef1  =  0.26200;
	t_coef2  = -0.81893;
	t_coef3  =  0.24826;
	t_coef4  =  0.41499;
	t_coef5  =  0.03246;
	t_coef6  = -0.27096;
	t_coef7  =  0.07671;
	c1  = 0;
	c2  = 0;
	c3  = 0;
	c4  = 0;
	c5  = 0;
	c6  = 0;
	c7  = 0;
	c8  = 0;
	c9  = 0;
	c10 = 0;
	c11 = 0;
	c12 = 0;
	c13 = 0;
	c14 = 0;
	
	
	while (1)
	{
		if(USART_RX_STA&0x8000)
		{			
			// Initialization
			if (USART_RX_BUF[0]==73 && USART_RX_BUF[1]==110){
				max5742_Init();
				
				printf("Type:%c\tStart...\r\n", USART_RX_BUF[2]);
				
				switch (USART_RX_BUF[2]){
					case '0':
						type = 0;
						scale = v_bias * 5;
						t_coef1  =  0.26200;
						t_coef2  = -0.81893;
						t_coef3  =  0.24826;
						t_coef4  =  0.41499;
						t_coef5  =  0.03246;
						t_coef6  = -0.27096;
						t_coef7  =  0.07671;
						c1  = 0;
						c2  = 0;
						c3  = 0;
						c4  = 0;
						c5  = 0;
						c6  = 0;
						c7  = 0;
						c8  = 0;
						c9  = 0;
						c10 = 0;
						c11 = 0;
						c12 = 0;
						c13 = 0;
						c14 = 0;
						
					break;
					case '1':
						type = 1;
						scale = v_bias * 5;
						t_coef1  = -0.5548;
						t_coef2  = -0.0479;
						t_coef3  = -1.0204;
						t_coef4  = -1.3010;
						t_coef5  = -0.4350;
						t_coef6  = -0.6310;
						t_coef7  = -0.0156;
						c1  =  1.2114;
						c2  =  0.3743;
						c3  =  0.3941;
						c4  =  0.3173;
						c5  =  0.7841;
						c6  =  1.1966;
						c7  =  0.6748;
						c8  = -0.3677;
						c9  = -1.1709;
						c10 =  0.7246;
						c11 =  0.5770;
						c12 =  0.4475;
						c13 =  0.0186;
						c14 = -0.2999;
						break;
					case '2':
						type = 2;
						scale = v_bias * 5;
						t_coef1  = -0.5548;
						t_coef2  = -0.0479;
						t_coef3  = -1.0204;
						t_coef4  = -1.3010;
						t_coef5  = -0.4350;
						t_coef6  = -0.6310;
						t_coef7  = -0.0156;
						c1  =  1.1445;
						c2  =  0.8389;
						c3  =  0.3522;
						c4  = -0.0470;
						c5  =  1.0946;
						c6  =  1.7254;
						c7  = -0.2340;
						c8  = -0.0552;
						c9  = -0.4400;
						c10 = -0.0754;
						c11 =  0.2577;
						c12 =  0.1274;
						c13 =  0.4787;
						c14 =  0.1360;
						break;
					case '3':
						type = 3;
						scale = v_bias * 5;
						t_coef1  = -0.5548;
						t_coef2  = -0.0479;
						t_coef3  = -1.0204;
						t_coef4  = -1.3010;
						t_coef5  = -0.4350;
						t_coef6  = -0.6310;
						t_coef7  = -0.0156;
						c1  =  0.9240;
						c2  =  0.1678;
						c3  =  0.3794;
						c4  =  0.0838;
						c5  =  0.8525;
						c6  =  0.5500;
						c7  =  0.2291;
						c8  =  0.3379;
						c9  = -0.1571;
						c10 =  0.4136;
						c11 =  0.1588;
						c12 = -0.1089;
						c13 =  0.0112;
						c14 = -0.7344;
						break;
					case '4':
						type = 0;
						scale = 0.05f;
						t_coef1  = -0.74262;
						t_coef2  =  1.68016;
						t_coef3  = -1.06879;
						t_coef4  =  0.20218;
						t_coef5  =  0.71796;
						t_coef6  =  1.27302;
						t_coef7  = -0.88574;
						c1  = 0;
						c2  = 0;
						c3  = 0;
						c4  = 0;
						c5  = 0;
						c6  = 0;
						c7  = 0;
						c8  = 0;
						c9  = 0;
						c10 = 0;
						c11 = 0;
						c12 = 0;
						c13 = 0;
						c14 = 0;
						break;
						
					// 7 classes
					case 'a':
						scale = v_bias * 5; t_coef1  = 1.6068;	t_coef2 = 1.9667;	t_coef3  = 0.8657;	t_coef4  = 0.6451; t_coef5  = -0.4855;	t_coef6  = -0.0697;	t_coef7  = -0.1329;
						c1=-1.0133; c2=0.5041; c3=-0.3721; c4=0.2269; c5=0.6161; c6=0.3471; c7=0.5200; c8=0.6466; c9=-0.1238; c10=0.9094; c11=0.6433; c12=0.4844; c13=-0.4175; c14=-0.6643; 
						break;
					case 'b':
						scale = v_bias * 5; t_coef1  = 1.6068;	t_coef2 = 1.9667;	t_coef3  = 0.8657;	t_coef4  = 0.6451; t_coef5  = -0.4855;	t_coef6  = -0.0697;	t_coef7  = -0.1329;
						c1=-0.6985; c2=0.8663; c3=0.2073; c4=0.4679; c5=1.4211; c6=-0.3322; c7=0.8090; c8=0.4955; c9=0.0694; c10=0.9316; c11=0.4901; c12=0.4246; c13=-0.5256; c14=-0.8494; 
						break;
					case 'c':
						scale = v_bias * 5;	t_coef1  = 1.6068;	t_coef2 = 1.9667;	t_coef3  = 0.8657;	t_coef4  = 0.6451; t_coef5  = -0.4855;	t_coef6  = -0.0697;	t_coef7  = -0.1329;
						c1=-1.4975; c2=0.8546; c3=0.0409; c4=-0.1207; c5=1.2654; c6=0.2509; c7=1.0190; c8=0.2180; c9=0.0175; c10=0.6539; c11=0.6163; c12=0.1201; c13=-0.5763; c14=-1.2089;
						break;
					case 'd':
						scale = v_bias * 5;	t_coef1  = 1.6068;	t_coef2 = 1.9667;	t_coef3  = 0.8657;	t_coef4  = 0.6451; t_coef5  = -0.4855;	t_coef6  = -0.0697;	t_coef7  = -0.1329;
						c1=-1.8357; c2=0.4813; c3=-0.5480; c4=-0.3843; c5=0.4446; c6=0.9311; c7=0.7338; c8=0.3699; c9=-0.1787; c10=0.6702; c11=0.7823; c12=0.1669; c13=-0.4941; c14=-1.0194; 
						break;
					case 'e':
						scale = v_bias * 5;	t_coef1  = 1.6068;	t_coef2 = 1.9667;	t_coef3  = 0.8657;	t_coef4  = 0.6451; t_coef5  = -0.4855;	t_coef6  = -0.0697;	t_coef7  = -0.1329;
						c1=-1.3518; c2=0.0478; c3=-0.9491; c4=0.0001; c5=-0.2394; c6=1.0370; c7=0.2489; c8=0.8382; c9=-0.3145; c10=1.0444; c11=0.8570; c12=0.5285; c13=-0.2744; c14=-0.4814; 
						break;
					case 'f':
						scale = v_bias * 5;	t_coef1  = 1.6068;	t_coef2 = 1.9667;	t_coef3  = 0.8657;	t_coef4  = 0.6451; t_coef5  = -0.4855;	t_coef6  = -0.0697;	t_coef7  = -0.1329;
						c1=-0.5486; c2=0.1302; c3=-0.7879; c4=0.6391; c5=-0.0503; c6=0.4617; c7=0.0307; c8=1.1467; c9=-0.2707; c10=1.3338; c11=0.7089; c12=0.8591; c13=-0.1593; c14=-0.1418; 
						break;
					case 'g':
						scale = v_bias * 5;	t_coef1  = 1.6068;	t_coef2 = 1.9667;	t_coef3  = 0.8657;	t_coef4  = 0.6451; t_coef5  = -0.4855;	t_coef6  = -0.0697;	t_coef7  = -0.1329;
						c1=-0.2332; c2=0.5970; c3=-0.2229; c4=0.8545; c5=0.7956; c6=-0.2278; c7=0.3070; c8=0.9650; c9=-0.0870; c10=1.2453; c11=0.4911; c12=0.8032; c13=-0.3075; c14=-0.3238; 
						break;
					
					
					// 9 classes
					case 'A':
						scale = v_bias * 5;	t_coef1  = -0.5856;	t_coef2 = -0.6444;	t_coef3  = 0.0775;	t_coef4  = 1.0223; t_coef5  = 0.5219;	t_coef6  = 0.8863;	t_coef7  = 0.0059;
						c1=1.0038; c2=0.6552; c3=0.6931; c4=-0.3331; c5=-0.1913; c6=-1.2144; c7=0.7771; c8=0.3989; c9=0.8684; c10=-0.1559; c11=-1.1434; c12=1.0901; c13=-0.2858; c14=-0.8726; 
						break;
					case 'B':
						scale = v_bias * 5;	t_coef1  = -0.5856;	t_coef2 = -0.6444;	t_coef3  = 0.0775;	t_coef4  = 1.0223; t_coef5  = 0.5219;	t_coef6  = 0.8863;	t_coef7  = 0.0059;
						c1=1.2160; c2=0.6231; c3=0.6931; c4=0.0473; c5=-0.2854; c6=-1.1713; c7=0.5441; c8=0.7082; c9=0.4113; c10=-0.6303; c11=-1.0125; c12=1.0155; c13=0.0069; c14=-0.4191;
						break;
					case 'C':
						scale = v_bias * 5;	t_coef1  = -0.5856;	t_coef2 = -0.6444;	t_coef3  = 0.0775;	t_coef4  = 1.0223; t_coef5  = 0.5219;	t_coef6  = 0.8863;	t_coef7  = 0.0059;
						c1=1.4242; c2=0.6235; c3=0.7219; c4=0.4146; c5=-0.4057; c6=-1.1322; c7=0.2885; c8=1.0319; c9=-0.0682; c10=-1.0755; c11=-0.8489; c12=0.9452; c13=0.3235; c14=0.0311;  
						break;
					case 'D':
						scale = v_bias * 5;	t_coef1  = -0.5856;	t_coef2 = -0.6444;	t_coef3  = 0.0775;	t_coef4  = 1.0223; t_coef5  = 0.5219;	t_coef6  = 0.8863;	t_coef7  = 0.0059;
						c1=0.8762; c2=0.6638; c3=0.5840; c4=-0.7522; c5=0.0861; c6=-0.7128; c7=0.2289; c8=0.4656; c9=0.9453; c10=-0.1588; c11=-1.1303; c12=0.6593; c13=0.3079; c14=-0.9824;
						break;
					case 'E':
						scale = v_bias * 5;	t_coef1  = -0.5856;	t_coef2 = -0.6444;	t_coef3  = 0.0775;	t_coef4  = 1.0223; t_coef5  = 0.5219;	t_coef6  = 0.8863;	t_coef7  = 0.0059;
						c1=1.1364; c2=0.6437; c3=0.5863; c4=-0.3870; c5=0.0190; c6=-0.6641; c7=-0.0189; c8=0.7700; c9=0.4978; c10=-0.6293; c11=-0.9953; c12=0.5720; c13=0.6184; c14=-0.5262; 
						break;
					case 'F':
						scale = v_bias * 5;	t_coef1  = -0.5856;	t_coef2 = -0.6444;	t_coef3  = 0.0775;	t_coef4  = 1.0223; t_coef5  = 0.5219;	t_coef6  = 0.8863;	t_coef7  = 0.0059;
						c1=1.3899; c2=0.6537; c3=0.6102; c4=-0.0341; c5=-0.0701; c6=-0.6203; c7=-0.2802; c8=1.0897; c9=0.0249; c10=-1.0784; c11=-0.8242; c12=0.4821; c13=0.9510; c14=-0.0699; 
						break;
					case 'G':
						scale = v_bias * 5;	t_coef1  = -0.5856;	t_coef2 = -0.6444;	t_coef3  = 0.0775;	t_coef4  = 1.0223; t_coef5  = 0.5219;	t_coef6  = 0.8863;	t_coef7  = 0.0059;
						c1=0.8726; c2=0.7469; c3=0.5663; c4=-1.1601; c5=0.3938; c6=-0.2294; c7=-0.4225; c8=0.5293; c9=1.0218; c10=-0.1417; c11=-1.1157; c12=0.2161; c13=0.9428; c14=-1.0675;  
						break;
					case 'H':
						scale = v_bias * 5;	t_coef1  = -0.5856;	t_coef2 = -0.6444;	t_coef3  = 0.0775;	t_coef4  = 1.0223; t_coef5  = 0.5219;	t_coef6  = 0.8863;	t_coef7  = 0.0059;
						c1=1.1382; c2=0.7094; c3=0.5253; c4=-0.8268; c5=0.3591; c6=-0.1603; c7=-0.6414; c8=0.8345; c9=0.5893; c10=-0.6297; c11=-0.9789; c12=0.0990; c13=1.2695; c14=-0.6220; 
						break;
					case 'I':
						scale = v_bias * 5;	t_coef1  = -0.5856;	t_coef2 = -0.6444;	t_coef3  = 0.0775;	t_coef4  = 1.0223; t_coef5  = 0.5219;	t_coef6  = 0.8863;	t_coef7  = 0.0059;
						c1=1.3818; c2=0.6984; c3=0.5107; c4=-0.4867; c5=0.2833; c6=-0.1179; c7=-0.8520; c8=1.1493; c9=0.1270; c10=-1.0913; c11=-0.8065; c12=0.0032; c13=1.5867; c14=-0.1728;
						break;
					
				}
				switch (USART_RX_BUF[3]){
					case 48:
						NoiseFlag = 0;
//						LCD_DisplayString(30,120,16, "Noise: No ");
					break;
					case 49:
						NoiseFlag = 1;
//						LCD_DisplayString(30,120,16, "Noise: Yes");
					break;
				}
		
			}
			
			// Go
			if (USART_RX_BUF[0]==71 && USART_RX_BUF[1]==111){
				go=1;
				printf("\r\n");
			}
			
			// config  
			// O
			if (USART_RX_BUF[0]==79){
				offset1 = ((USART_RX_BUF[1]-48)*10 + (USART_RX_BUF[2]-48)) * 0.01f - 0.5f;
				offset2 = ((USART_RX_BUF[3]-48)*10 + (USART_RX_BUF[4]-48)) * 0.01f - 0.5f;
				printf("Offset set1: %f\nOffset set2: %f\r\n", offset1, offset2);
			}
			// C
			if (USART_RX_BUF[0]==67){
				float tmp;
				tmp = ((USART_RX_BUF[1]-48)*100 + (USART_RX_BUF[2]-48)*10 + (USART_RX_BUF[3]-48)) * 0.01f;
				if ((tmp<10)&&(tmp>0)) coef = tmp;
				printf("Coef: %f\r\n", coef);
			}
			// D
			if (USART_RX_BUF[0]==68){
				if (USART_RX_BUF[1]==48) v_bias = 0.05f;
				else if (USART_RX_BUF[1]==49) v_bias = 0.1f;
				else if (USART_RX_BUF[1]==50) v_bias = 0.2f;
				
				printf("V_bias and DAC coef: %f\r\n", v_bias);
			}
			// N
			if (USART_RX_BUF[0]==78){
				noise_scale = ((USART_RX_BUF[1]-48)*100 + (USART_RX_BUF[2]-48)*10 + (USART_RX_BUF[3]-48)) * 0.01f;
				printf("Noise Scale: %f\r\n", noise_scale);
			}
			
			
			USART_RX_STA=0;
		}
		
		
		key_scan(0);
		
		if(keydown_data==KEY0_DATA)   //key0
		{
			NoiseFlag = 1;
		}
		if(keyup_data==KEY1_DATA)     //key1
		{
			NoiseFlag = 0;
		}
		if(key_tem==KEY2_DATA && key_time>200) 
		{

		}
		
	delay_ms(10); 
	}
	
}





