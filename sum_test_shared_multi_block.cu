#include "cuda.h"
#include "stdio.h"

#define threads_per_block 5



void printi(int i){
	printf("%d\n", i);
}


void init_CPU_array(int* array, int n){
	for(int i = 0; i < n; i++) {
		array[i] = 1;
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
	__shared__ int compartida[threads_per_block];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	compartida[threadIdx.x] = arreglo[tid];
	__syncthreads();
	for(int i=1; pow((float)2,(float)i-1) < N; i++)
	{
		int acceso = pow((float)2,(float)i);
		int offset = pow((float)2, (float)i-1);
		if(threadIdx.x < (N/acceso))
		{
				compartida[threadIdx.x * acceso] = compartida[threadIdx.x * acceso] + compartida[threadIdx.x * acceso + offset];
				compartida[threadIdx.x * acceso + offset] = 0;
				// printf("%s %d\n", "TRABAJA id:", threadIdx.x);
				result[blockIdx.x] = compartida[0];

		}

		// printf("%s\n", "");

		
	}

}





int* arreglo_suma1;
int* d_arreglo_suma1;

int* arreglo_result;
int* d_arreglo_suma2;

int main(int argc, char** argv){

	int N = 26;
	//##################################################################################
	//############################## INICIALIZACION ####################################

	arreglo_suma1 = (int*) malloc(N * sizeof(int));
	cudaMalloc(&d_arreglo_suma1, N * sizeof(int));

	arreglo_result = (int*) malloc(N * sizeof(int));
	cudaMalloc(&d_arreglo_suma2, N * sizeof(int));


	init_CPU_array(arreglo_suma1, N);
	cudaMemcpy(d_arreglo_suma1, arreglo_suma1, N * sizeof(int), cudaMemcpyHostToDevice);

	//float threads_per_block = 10;
	int block_count = ceil((float)N / threads_per_block);

	//##################################################################################
	//################################ EJECUCIONES #####################################

	dim3 miGrid1D_1(block_count,1);
	dim3 miBloque1D_1(threads_per_block,1);
	sumador<<<miGrid1D_1, miBloque1D_1>>>(d_arreglo_suma1, d_arreglo_suma2, threads_per_block);
 	cudaThreadSynchronize();

	printf("%s\n", "fin 1");

	dim3 miGrid1D_2(2,1);
	dim3 miBloque1D_2(threads_per_block,1);
	sumador<<<miGrid1D_2, miBloque1D_2>>>(d_arreglo_suma2, d_arreglo_suma1, threads_per_block);
 	cudaThreadSynchronize();

	printf("%s\n", "fin 2");

	//##################################################################################
	//################################### READ BACK #####################################

	cudaMemcpy(arreglo_result, d_arreglo_suma1, N * sizeof(int), cudaMemcpyDeviceToHost);

	printf("%s\n", "RESULTADO DE LA SUMA:");
	print_CPU_array(arreglo_result, N);

	free(arreglo_suma1);
	cudaFree (d_arreglo_suma1);

	free(arreglo_result);
	cudaFree (d_arreglo_suma2);

}