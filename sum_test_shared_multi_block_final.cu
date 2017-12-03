#include "cuda.h"
#include "stdio.h"
#define threads_per_block 512




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
	if(tid > N)
	{
		return;
	}

	compartida[threadIdx.x] = arreglo[tid];
		__syncthreads();
		for(int i=1; pow((float)2,(float)i-1) < threads_per_block; i++)
		{
			int acceso = pow((float)2,(float)i);
			int offset = pow((float)2, (float)i-1);
			if(threadIdx.x < ((float)threads_per_block/acceso) && (threadIdx.x * acceso + offset) < (N - blockIdx.x * blockDim.x))
			{
					compartida[threadIdx.x * acceso] = compartida[threadIdx.x * acceso] + compartida[threadIdx.x * acceso + offset];
					// compartida[threadIdx.x * acceso + offset] = 0;
			}
			__syncthreads();

		}

		//el primer thread de cada grupo guarda el resultado

		if(threadIdx.x == 0)
			result[blockIdx.x] = compartida[0];
	


}





int* arreglo_suma1;
int* d_arreglo_suma1;

int* arreglo_result;
int* d_arreglo_suma2;

int main(int argc, char** argv){

	int N = 1024000;
	//##################################################################################
	//############################## INICIALIZACION ####################################

	arreglo_suma1 = (int*) malloc(N * sizeof(int));
	cudaMalloc(&d_arreglo_suma1, N * sizeof(int));

	arreglo_result = (int*) malloc(N * sizeof(int));
	cudaMalloc(&d_arreglo_suma2, N * sizeof(int));


	init_CPU_array(arreglo_suma1, N);
	cudaMemcpy(d_arreglo_suma1, arreglo_suma1, N * sizeof(int), cudaMemcpyHostToDevice);

	// int threads_per_block = 10;
	// int block_count = ceil((float)N / threads_per_block);

	//##################################################################################
	//################################ EJECUCIONES #####################################

	dim3 miBloque1D_1(threads_per_block,1);
	for(int i=0; pow(threads_per_block, i) < N ; i++)
	{
		int remaining_elements = ceil((float)N/pow(threads_per_block, i));
		int block_count = ceil((float)N/pow(threads_per_block, i+1));
		dim3 miGrid1D_1(block_count,1);
		sumador<<<miGrid1D_1, miBloque1D_1>>>(d_arreglo_suma1, d_arreglo_suma2, remaining_elements);
		cudaThreadSynchronize();

		int* tmp = d_arreglo_suma1;
		d_arreglo_suma1 = d_arreglo_suma2;
		d_arreglo_suma2 = tmp;

		printf("elementos restantes: %d \n", remaining_elements);
		printf("bloques usados:      %d \n\n", block_count);


	}

	//##################################################################################
	//################################### READ BACK #####################################

	cudaMemcpy(arreglo_result, d_arreglo_suma1, N * sizeof(int), cudaMemcpyDeviceToHost);

	printf("%s\n", "RESULTADO DE LA SUMA:");
	print_CPU_array(arreglo_result, 1);

	free(arreglo_suma1);
	cudaFree (d_arreglo_suma1);

	free(arreglo_result);
	cudaFree (d_arreglo_suma2);

}