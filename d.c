#include "stdio.h"
#include <math.h>

int n = 10;
int N = 10;
int array[10];

void init_CPU_array(int array[], int n){
	for(int i = 0; i < n; i++) {
		array[i] = i;
	}
}


void kernel_GPU(int acceso, int offset, int array[],int i)
{
	printf("%d\n", n);	
	for(int id=0; id< 10; id++)
	{
		if(id < N/pow(2, i))
		{
			printf("%s\n", "TRABAJO");

			array[id * acceso] = array[id * acceso] + array[id * acceso + offset];
		}
	}
}

int main(int argc, char** argv){

	init_CPU_array(array, n);
	
	//printf("%d\n", (int)ceil(dd));
	for(int i=1; i < n; i++)
	{
		printf("%s\n", "asd");
		kernel_GPU(pow(2,i), pow(2, i-1), array, i);
		if(i==4) break;
	}
	printf("%d\n", array[0]);
	

}
