#include "cuda.h"
#include "stdio.h"
#define threads_per_block 32




void printi(int i){
	printf("%d\n", i);
}


void init_CPU_array(int* arreglo_b, int n){
	for(int i=0; i< n; i++)
	{
		int valor = 1;
		arreglo_b[(i*16) + 0] = valor;
		arreglo_b[(i*16) + 1] = valor;
		arreglo_b[(i*16) + 2] = valor;
		arreglo_b[(i*16) + 3] = valor;
		arreglo_b[(i*16) + 4] = valor;
		arreglo_b[(i*16) + 5] = valor;
		arreglo_b[(i*16) + 6] = valor;
		arreglo_b[(i*16) + 7] = valor;
		arreglo_b[(i*16) + 8] = valor;
		arreglo_b[(i*16) + 9] = valor;
		arreglo_b[(i*16) + 10] = valor;
		arreglo_b[(i*16) + 11] = valor;
		arreglo_b[(i*16) + 12] = valor;
		arreglo_b[(i*16) + 13] = valor;
		arreglo_b[(i*16) + 14] = valor;
		arreglo_b[(i*16) + 15] = valor;

	}


}
void print_CPU_array(int array[], int n){
	for(int i = 0; i < n; i++) {
		printi(array[i]);
	}
}

void print_CPU_matrix(int array[], int n){
    for(int i = 0; i < n; i++) {
        if(i % 16 == 0)
            printf("%s\n", "");

        printf("%d ", array[i]);
    }
}


__global__ void sumador(int* arreglo, int* result, float N)
{
	__shared__ int compartida[threads_per_block * 16];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid > N * 16)
		return;

	compartida[threadIdx.x] = arreglo[tid];
	__syncthreads();
	for(int i=1; pow((float)2,(float)i-1) < N; i++)
	{
		int acceso = pow((float)2,(float)i);
		int offset = pow((float)2, (float)i-1);

		int t_id = (threadIdx.x/16) * 16;
		int new_access = t_id * acceso + threadIdx.x % 16 ;
		int new_offset = new_access + offset * 16;

		if(t_id < ((float)threads_per_block*16/acceso) && (new_offset  < (threads_per_block*16)))
		{
				

				compartida[new_access] = compartida[new_access] + compartida[new_offset];
				compartida[new_offset] = 0;
				// printf("GRUPO: %d - ITERACION: %d - TID %d - ACCESO: %d - OFFSET %d - REMAINING: %d \n", blockIdx.x,
				// 														i, 			tid, 	new_access , new_offset, threadIdx.x * acceso + offset);
		}
		__syncthreads();


	}

	//el primer thread de cada grupo guarda el resultado
	if(threadIdx.x < 16)
		result[blockIdx.x * 16 + threadIdx.x] = compartida[threadIdx.x];

}





int* arreglo_suma1;
int* d_arreglo_suma1;

int* arreglo_result;
int* d_arreglo_suma2;

int main(int argc, char** argv){

	int N = 1100000;
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


	dim3 miBloque1D_1(threads_per_block *16,1);
	for(int i=1; pow(threads_per_block, i-1) < N; i++)
	{
		int remaining_elements = ceil((float)N/pow(threads_per_block, i-1));
		int block_count = ceil((float)N/pow(threads_per_block * 16, i-1));

		dim3 miGrid1D_1(remaining_elements,1);
		sumador<<<miGrid1D_1, miBloque1D_1>>>(d_arreglo_suma1, d_arreglo_suma2, remaining_elements);
		cudaThreadSynchronize();
		printf("ERROR %s\n", cudaGetErrorString(cudaGetLastError()));
		int* tmp = d_arreglo_suma1;
		d_arreglo_suma1 = d_arreglo_suma2;
		d_arreglo_suma2 = tmp;

		printf("elementos restantes: %d \n ", remaining_elements);
		printf("block_count: %d \n ", block_count);
		printf("\n ", "");


	}


	//##################################################################################
	//################################### READ BACK #####################################

	cudaMemcpy(arreglo_result, d_arreglo_suma1, N * sizeof(int) * 16, cudaMemcpyDeviceToHost);

	printf("%s\n", "RESULTADO DE LA SUMA:");
	print_CPU_matrix(arreglo_result, 1 * 16);

	free(arreglo_suma1);
	cudaFree (d_arreglo_suma1);

	free(arreglo_result);
	cudaFree (d_arreglo_suma2);

}