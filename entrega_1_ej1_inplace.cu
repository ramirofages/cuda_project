#include "cuda.h"
#include "stdio.h"

#include <sys/time.h>
#include <sys/resource.h>


double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}

int cant = 512;
int cant_elem = cant * cant;



// arreglos usados como matrices
int* arreglo_A;
int* arreglo_B;
int* arreglo_C;

int* d_arreglo_A;
int* d_arreglo_B;
int* d_arreglo_C;


void printi(int i){
	printf("%d\n", i);
}

void init_CPU_array(int array[], int n){
	for(int i = 0; i < n; i++) {
		array[i] = i;
	}
}
void print_CPU_array(int array[], int n){
	for(int i = 0; i < n; i++) {
		printi(array[i]);
	}
}

// calcula la transpuesta in-place
__global__ void transposeador(int* arreglo_b, int N){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = int((1+sqrtf(1+8*tid))/2);
	int j = tid - (i*(i-1)/2);
	int aux;
	if((i<N) && (j<N)){
		aux = arreglo_b[i*N+j];
		arreglo_b[i*N+j] = arreglo_b[j*N+i];
		arreglo_b[j*N+i] = aux;
	}

}

// copia B en C
__global__ void copiador(int* arreglo_b, int* arreglo_c, int N){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < N)
		arreglo_c[tid] = arreglo_b[tid];
}

// C += A
__global__ void sumador(int* arreglo_a, int* arreglo_c, int N){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < N)
		arreglo_c[tid] += arreglo_a[tid];
}

// C += A * B^t
__global__ void multiplicador(int* arreglo_a, int* arreglo_b_trans, int* arreglo_c, int N, int total_elem){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = (int)tid / N; // columna
	int j = (int)tid % N; // fila

	int k;
	int cuenta = 0;
	if(tid < total_elem)
	{
		for (k=0; k< N; k++){
			cuenta += arreglo_a[i*N+k] * arreglo_b_trans[k*N+j];
		}
		arreglo_c[tid] += cuenta;
	}

}


void solucion_CPU(){
	int* arreglo_at;
	int* arreglo_bt;
	int* arreglo_a_por_b;
	int* arreglo_res;


	int numBytes = sizeof(int) * cant_elem; //bytes a alocar

	arreglo_at = (int *) malloc(numBytes);
	arreglo_bt = (int *) malloc(numBytes);
	arreglo_a_por_b = (int *) malloc(numBytes); // resultado de A * B^t
	arreglo_res = (int *) malloc(numBytes);

	double timetick;
	timetick = dwalltime();

	// guardamos en arreglo_bt y arreglo_at los datos que van a ser transpuestos
	for (int i = 0; i < cant_elem; ++i)
	{
		arreglo_bt[i] = arreglo_B[i];
		arreglo_at[i] = arreglo_A[i];
	}

	// calculamos la transpuesta de B
	for (int i = 0; i < (cant * (cant+1))/2; ++i)
	{
		int col = int((1+sqrtf(1+8*i))/2); // columna
		int row = i - (col*(col-1)/2); // fila

		int aux;
		if((col<cant) && (row<cant)){
			aux = arreglo_bt[col*cant+row];
			arreglo_bt[col*cant+row] = arreglo_bt[row*cant+col];
			arreglo_bt[row*cant+col] = aux;
		}
	}

	// calculamos la transpuesta de A
	for (int i = 0; i < (cant * (cant+1))/2; ++i)
	{
		int col = int((1+sqrtf(1+8*i))/2); // columna
		int row = i - (col*(col-1)/2); // fila

		int aux;
		if((col<cant) && (row<cant)){
			aux = arreglo_at[col*cant+row];
			arreglo_at[col*cant+row] = arreglo_at[row*cant+col];
			arreglo_at[row*cant+col] = aux;
		}
	}

	for (int i = 0; i < cant_elem; i++)
	{
		int col = i / cant; // columna
		int row = i % cant; // fila
		int mul = 0;
		for (int k=0; k< cant; k++){
			mul += arreglo_A[col*cant+k] * arreglo_bt[k*cant+row];
		}
		arreglo_a_por_b[i] = mul;
	}

	for (int i = 0; i < cant_elem; i++){
		arreglo_res[i] = 0;
	}
	// C = B + A * B^t + A^t
	for (int i = 0; i < cant_elem; i++){
		arreglo_res[i] += arreglo_B[i] + arreglo_a_por_b[i] + arreglo_at[i];
	}

	
	printf("-> Tiempo transcurrido en la CPU %f\n", dwalltime() - timetick);

	// printf("%s\n", "");
	// printf("%s\n", "Resultados CPU:");
	// for (int i = 0; i < cant_elem; i++){
	// 	printf("%d\n", arreglo_res[i]);
	// }




	free(arreglo_at);
	free(arreglo_bt);
	free(arreglo_a_por_b);
	free(arreglo_res);
}


int main(int argc, char** argv){
	int numBytes = sizeof(int) * cant_elem; //bytes a alocar

	arreglo_A = (int *) malloc(numBytes);
	arreglo_B = (int *) malloc(numBytes);
	arreglo_C = (int *) malloc(numBytes);

	// llenamos los arreglos
	init_CPU_array(arreglo_A, cant_elem);
	init_CPU_array(arreglo_B, cant_elem);
	init_CPU_array(arreglo_C, cant_elem);

	// allocamos memoria en la gpu
	cudaMalloc(&d_arreglo_A, numBytes);
	cudaMalloc(&d_arreglo_B, numBytes);
	cudaMalloc(&d_arreglo_C, numBytes);

	// copiamos los datos de la cpu a la gpu
	cudaMemcpy(d_arreglo_A, arreglo_A, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_arreglo_B, arreglo_B, numBytes, cudaMemcpyHostToDevice);

	double timetick;
	timetick = dwalltime();

	dim3 miGrid1D(512,1);
	dim3 miBloque1D(512,1);
	dim3 miBloque1D_transposeador((cant * (cant + 1))/2,1); // (N*(N+1))/2
	

	// C = B
	copiador<<<miGrid1D, miBloque1D>>>(d_arreglo_B, d_arreglo_C, cant_elem);

	// B^t
	transposeador<<<miGrid1D, miBloque1D_transposeador>>>(d_arreglo_B, cant);

	// C += A * B^t
 	multiplicador <<<miGrid1D, miBloque1D>>>(d_arreglo_A, d_arreglo_B, d_arreglo_C, cant, cant_elem);

	// A^t
	transposeador<<<miGrid1D, miBloque1D_transposeador>>>(d_arreglo_A, cant);

	// C += A^t
	sumador<<<miGrid1D, miBloque1D>>>(d_arreglo_A, d_arreglo_C, cant_elem);

	// esperamos a que termine la ejecucion
 	cudaThreadSynchronize();
	printf("-> Tiempo transcurrido en la GPU %f\n", dwalltime() - timetick);

	// nos traemos los resultados de la gpu a la cpu
	cudaMemcpy(arreglo_C, d_arreglo_C, numBytes, cudaMemcpyDeviceToHost);

	// imprimimos los resultados
	// printf("%s\n", "");
	// printf("%s\n", "Resultados GPU:");
	// print_CPU_array(arreglo_C, cant_elem);

	solucion_CPU();

	// liberamos memoria
	free(arreglo_A);
	free(arreglo_B);
	free(arreglo_C);
	cudaFree (d_arreglo_A);
	cudaFree (d_arreglo_B);
	cudaFree (d_arreglo_C);

}
