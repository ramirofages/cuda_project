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
__global__ void sumador_3(int* arreglo, int acceso, int offset, int i, float N){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// if(tid < N/acceso)
	// {
	// 	printf("%d\n", arreglo[tid * acceso]);
	// 	printf("%d\n", arreglo[tid * acceso + offset]);
	// }
	if(tid < (N/acceso))
	{
			arreglo[tid * acceso] = arreglo[tid * acceso] + arreglo[tid * acceso + offset];
			arreglo[tid * acceso + offset] = 0;
			printf("%s\n", "TRABAJO");

	}
}




int* arreglo_determinantes;
int* d_arreglo_determinantes;

int main(int argc, char** argv){

	int* suma_det = (int *) malloc(sizeof(int)); 

	arreglo_determinantes = (int*) malloc(N * sizeof(int));
	cudaMalloc(&d_arreglo_determinantes, N * sizeof(int));

	init_CPU_array(arreglo_determinantes, N);
	cudaMemcpy(d_arreglo_determinantes, arreglo_determinantes, N * sizeof(int), cudaMemcpyHostToDevice);

	dim3 miGrid1D_2(1,1);
	dim3 miBloque1D_2(N,1);

	for(int i=1; i < N; i++)
	{
		sumador_3<<<miGrid1D_2, miBloque1D_2>>>(d_arreglo_determinantes, (int)pow(2,i), (int)pow(2, i-1), i, N);
		cudaThreadSynchronize();
		
		printf("%s\n", "Acceso:");
		printf("%d\n", (int)pow(2,i));

		printf("%s\n", "Offset:");
		printf("%d\n", (int)pow(2,i-1));

		printf("%s\n", " ");


		if(i==4) break;
	}

	cudaMemcpy(arreglo_determinantes, d_arreglo_determinantes, 10 * sizeof(int), cudaMemcpyDeviceToHost);

	printf("%s\n", "TEST SUMA:");
	//printf("%d\n", *suma_det);
	print_CPU_array(arreglo_determinantes, 10);

	free(arreglo_determinantes);
	cudaFree (d_arreglo_determinantes);

}