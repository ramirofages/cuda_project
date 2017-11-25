#include "cuda.h"
#include "stdio.h"

#define threads_per_block 10



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
	for(int i=1; pow((float)2,(float)i-1) < 10; i++)
	{
		int acceso = pow((float)2,(float)i);
		int offset = pow((float)2, (float)i-1);
		if(threadIdx.x < (10.0/acceso))
		{
				if((threadIdx.x * acceso + offset) < (N - blockIdx.x * blockDim.x))
				{
					compartida[threadIdx.x * acceso] = compartida[threadIdx.x * acceso] + compartida[threadIdx.x * acceso + offset];
					compartida[threadIdx.x * acceso + offset] = 0;
				}
				
				// printf("%s %d\n", "TRABAJA id:", threadIdx.x);
				// printf("%s %d\n", "OFFSET:", threadIdx.x * acceso + offset);
				// printf("%s %d\n", "result:", compartida[threadIdx.x * acceso]);
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

	int N = 110;
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
	printf("block count %d\n", block_count);

	//##################################################################################
	//################################ EJECUCIONES #####################################

	dim3 miGrid1D_1(block_count,1);
	dim3 miBloque1D_1(threads_per_block,1);
	sumador<<<miGrid1D_1, miBloque1D_1>>>(d_arreglo_suma1, d_arreglo_suma2, N);
 	cudaThreadSynchronize();

	int remaining_elements = ceil((float)N/threads_per_block);
	printf("fin 1, elementos restantes: %d\n", remaining_elements);

	dim3 miGrid1D_2(2,1);
	dim3 miBloque1D_2(threads_per_block,1);
	sumador<<<miGrid1D_2, miBloque1D_2>>>(d_arreglo_suma2, d_arreglo_suma1, 11);
 	cudaThreadSynchronize();

	// remaining_elements = ceil((float)N/threads_per_block/threads_per_block);
	// printf("fin 2, elementos restantes: %d\n", remaining_elements);

	dim3 miGrid1D_3(1,1);
	dim3 miBloque1D_3(threads_per_block,1);
	sumador<<<miGrid1D_3, miBloque1D_3>>>(d_arreglo_suma1, d_arreglo_suma2, 2);
 	cudaThreadSynchronize();

	// remaining_elements = ceil((float)N/threads_per_block/threads_per_block/threads_per_block);
	// printf("fin 3, elementos restantes: %d\n", remaining_elements);

	//##################################################################################
	//################################### READ BACK #####################################

	cudaMemcpy(arreglo_result, d_arreglo_suma2, N * sizeof(int), cudaMemcpyDeviceToHost);

	printf("%s\n", "RESULTADO DE LA SUMA:");
	print_CPU_array(arreglo_result, 15);

	free(arreglo_suma1);
	cudaFree (d_arreglo_suma1);

	free(arreglo_result);
	cudaFree (d_arreglo_suma2);

}