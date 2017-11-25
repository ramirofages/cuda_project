#include "cuda.h"
#include "stdio.h"


int N = 10;


void printi(int i){
	printf("%d\n", i);
}


void init_CPU_array(int* array, int n){
	for(int i = 0; i < n; i++) {
		array[i] = i;
	}
}
void print_CPU_array(int array[], int n){
	for(int i = 0; i < n; i++) {
		printi(array[i]);
	}
}


// realiza la suma de determinantes
__global__ void sumador_3(int* arreglo, int* result, float N){
	__shared__ int compartida[10];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	compartida[threadIdx.x] = arreglo[tid];
	__syncthreads();
	printf("%d\n", compartida[threadIdx.x]);
	for(int i=1; pow((float)2,(float)i-1) < N; i++)
	{
		int acceso = pow((float)2,(float)i);
		int offset = pow((float)2, (float)i-1);
		if(threadIdx.x < (N/acceso))
		{
				arreglo[threadIdx.x * acceso] = arreglo[threadIdx.x * acceso] + arreglo[threadIdx.x * acceso + offset];
				arreglo[threadIdx.x * acceso + offset] = 0;
				printf("%s\n", "TRABAJO");

		}
		printf("%s\n", "");
		
	}

}




int* arreglo_determinantes;
int* d_arreglo_determinantes;

int* arreglo_suma1;
int* d_arreglo_suma1;

int* arreglo_suma2;
int* d_arreglo_suma2;

int main(int argc, char** argv){

	int* suma_det = (int *) malloc(sizeof(int)); 

	arreglo_determinantes = (int*) malloc(N * sizeof(int));
	cudaMalloc(&d_arreglo_determinantes, N * sizeof(int));

	arreglo_suma1 = (int*) malloc(N * sizeof(int));
	cudaMalloc(&d_arreglo_suma1, N * sizeof(int));

	arreglo_suma2 = (int*) malloc(N * sizeof(int));
	cudaMalloc(&d_arreglo_suma2, N * sizeof(int));


	init_CPU_array(arreglo_determinantes, N);
	cudaMemcpy(d_arreglo_determinantes, arreglo_determinantes, N * sizeof(int), cudaMemcpyHostToDevice);

	dim3 miGrid1D_2(1,1);
	dim3 miBloque1D_2(N,1);

	sumador_3<<<miGrid1D_2, miBloque1D_2>>>(d_arreglo_determinantes, N);
	// for(int i=1; pow(2,i-1) < N; i++)
	// {
	// 	sumador_3<<<miGrid1D_2, miBloque1D_2>>>(d_arreglo_determinantes, (int)pow(2,i), (int)pow(2, i-1), N);
	// 	cudaThreadSynchronize();

	// 	printf("%s\n", " ");

	// }

	cudaMemcpy(arreglo_determinantes, d_arreglo_determinantes, 10 * sizeof(int), cudaMemcpyDeviceToHost);

	printf("%s\n", "TEST SUMA:");
	print_CPU_array(arreglo_determinantes, 10);

	free(arreglo_determinantes);
	cudaFree (d_arreglo_determinantes);

}