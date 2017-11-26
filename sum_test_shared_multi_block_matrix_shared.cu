#include "cuda.h"
#include "stdio.h"
#define threads_per_block 10




void printi(int i){
	printf("%d\n", i);
}


void init_CPU_array(int* arreglo_b, int n){
	for(int i=0; i< n; i++)
	{
		arreglo_b[(i*16) + 0] = 1;
		arreglo_b[(i*16) + 1] = 1;
		arreglo_b[(i*16) + 2] = 1;
		arreglo_b[(i*16) + 3] = 1;
		arreglo_b[(i*16) + 4] = 1;
		arreglo_b[(i*16) + 5] = 1;
		arreglo_b[(i*16) + 6] = 1;
		arreglo_b[(i*16) + 7] = 1;
		arreglo_b[(i*16) + 8] = 1;
		arreglo_b[(i*16) + 9] = 1;
		arreglo_b[(i*16) + 10] = 1;
		arreglo_b[(i*16) + 11] = 1;
		arreglo_b[(i*16) + 12] = 1;
		arreglo_b[(i*16) + 13] = 1;
		arreglo_b[(i*16) + 14] = 1;
		arreglo_b[(i*16) + 15] = 1;

	}


}
void print_CPU_array(int array[], int n){
	for(int i = 0; i < n; i++) {
		printi(array[i]);
	}
}


// realiza la suma de determinantes
__global__ void sumador(int* arreglo, int* result, float N)
{
	__shared__ int compartida[threads_per_block * 16];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	compartida[threadIdx.x] = arreglo[tid];
	printf("RESULT: %d, THREAD ID: %d\n", compartida[threadIdx.x], threadIdx.x);

	__syncthreads();
	for(int i=1; pow((float)2,(float)i-1) < 10 * 16; i++)
	{
		int acceso = pow((float)2,(float)i);
		int offset = pow((float)2, (float)i-1) * 16;
		if(threadIdx.x < ((10.0)/acceso) && (threadIdx.x * acceso + offset) < (N * 16- blockIdx.x * blockDim.x ))
		{
				compartida[threadIdx.x * acceso] = compartida[threadIdx.x * acceso] + compartida[threadIdx.x * acceso + offset];
				// compartida[threadIdx.x * acceso + offset * 16] = 0;

		}


	}

	//el primer thread de cada grupo guarda el resultado
	if(threadIdx.x < 16)
		result[blockIdx.x] = compartida[blockIdx.x];

}
//realiza la suma de n matrices
__global__ void sumador_2(int* arreglo, int* result, float N){

	__shared__ int compartido[threads_per_block * 16];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	compartido[threadIdx.x] = arreglo[tid];
	__syncthreads();


	for(int i=1; pow((float)2,(float)i-1) < N; i++)
	{
		int acceso = pow((float)2,(float)i);
		int offset = pow((float)2, (float)i-1) ;
		if( (( (threadIdx.x / 16) & (offset*2)-1 ) == 0) && ( (threadIdx.x+offset*16) < N*16))
		{
				compartido[threadIdx.x] = compartido[threadIdx.x] + compartido[threadIdx.x + offset * 16];
				printf("TRABAJAMOS TID:%d - SUMA:%d + %d \n", threadIdx.x, threadIdx.x, threadIdx.x + offset * 16);
		}
	}
	__syncthreads();

	if(threadIdx.x < 16)
	{
		result[threadIdx.x] = compartido[threadIdx.x];
	}
}




int* arreglo_suma1;
int* d_arreglo_suma1;

int* arreglo_result;
int* d_arreglo_suma2;

int main(int argc, char** argv){

	int N = 4;
	//##################################################################################
	//############################## INICIALIZACION ####################################
	int byte_size = N * sizeof(int) * 16;
	arreglo_suma1 = (int*) malloc(byte_size);
	cudaMalloc(&d_arreglo_suma1, byte_size);

	arreglo_result = (int*) malloc(byte_size);
	cudaMalloc(&d_arreglo_suma2, byte_size);


	init_CPU_array(arreglo_suma1, N);
	cudaMemcpy(d_arreglo_suma1, arreglo_suma1, byte_size, cudaMemcpyHostToDevice);


	//##################################################################################
	//################################ EJECUCIONES #####################################

	dim3 miBloque1D_1(threads_per_block * 16,1);
	dim3 miGrid1D_1(1,1);
	sumador_2<<<miGrid1D_1, miBloque1D_1>>>(d_arreglo_suma1, d_arreglo_suma2, N);

	//##################################################################################
	//################################### READ BACK #####################################

	cudaMemcpy(arreglo_result, d_arreglo_suma2, N * sizeof(int) * 16, cudaMemcpyDeviceToHost);

	printf("%s\n", "RESULTADO DE LA SUMA:");
	print_CPU_array(arreglo_result, 16);

	free(arreglo_suma1);
	cudaFree (d_arreglo_suma1);

	free(arreglo_result);
	cudaFree (d_arreglo_suma2);

}