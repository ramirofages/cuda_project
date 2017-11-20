#include "cuda.h"
#include "stdio.h"


//#include <sys/time.h>
//#include <sys/resource.h>


// double dwalltime(){
//         double sec;
//         struct timeval tv;

//         gettimeofday(&tv,NULL);
//         sec = tv.tv_sec + tv.tv_usec/1000000.0;
//         return sec;
// }


int N = 3;


// arreglos usados como matrices

int* arreglo_A;
int* arreglo_B;
int* arreglo_C;

int* suma_det;

int* d_arreglo_A;
int* d_arreglo_B;
int* d_arreglo_C;



void printi(int i){
	printf("%d\n", i);
}

void init_CPU_array(int* arreglo_b, int n){
	for(int i=0; i< N; i++)
	{
		arreglo_b[(i*16) + 0] = 1;
		arreglo_b[(i*16) + 1] = 2;
		arreglo_b[(i*16) + 2] = 0;
		arreglo_b[(i*16) + 3] = 0;
		arreglo_b[(i*16) + 4] = 0;
		arreglo_b[(i*16) + 5] = 1;
		arreglo_b[(i*16) + 6] = 0;
		arreglo_b[(i*16) + 7] = 0;
		arreglo_b[(i*16) + 8] = 0;
		arreglo_b[(i*16) + 9] = 0;
		arreglo_b[(i*16) + 10] = 1;
		arreglo_b[(i*16) + 11] = 0;
		arreglo_b[(i*16) + 12] = 0;
		arreglo_b[(i*16) + 13] = 0;
		arreglo_b[(i*16) + 14] = 0;
		arreglo_b[(i*16) + 15] = 1;
	}

}

void print_CPU_array(int array[], int n){
	for(int i = 0; i < n; i++) {
		printi(array[i]);
	}
}

void print_CPU_array_double(double array[], int n){
	for(int i = 0; i < n; i++) {
		printf("%f\n", array[i]);
	}
}




__global__ void determinanteador(int* arreglo_b, int* arreglo_a, int N){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	

	int mat[9];
	mat[0] = arreglo_b[(tid * 16) + 5];
	mat[1] = arreglo_b[(tid * 16) + 6];
	mat[2] = arreglo_b[(tid * 16) + 7];
	mat[3] = arreglo_b[(tid * 16) + 9];
	mat[4] = arreglo_b[(tid * 16) + 10];
	mat[5] = arreglo_b[(tid * 16) + 11];
	mat[6] = arreglo_b[(tid * 16) + 13];
	mat[7] = arreglo_b[(tid * 16) + 14];
	mat[8] = arreglo_b[(tid * 16) + 15];
	
	float det0 = mat[0]*(mat[4]*mat[8] - mat[5]*mat[7]);
	float det1 = mat[1]*(mat[3]*mat[8] - mat[5]*mat[6]);
	float det2 = mat[2]*(mat[3]*mat[7] - mat[4]*mat[6]);
	
	float result0 = det0 - det1 + det2;
	result0 *= arreglo_b[0];	

	mat[0] = arreglo_b[(tid * 16) + 4];
	mat[1] = arreglo_b[(tid * 16) + 6];
	mat[2] = arreglo_b[(tid * 16) + 7];
	mat[3] = arreglo_b[(tid * 16) + 8];
	mat[4] = arreglo_b[(tid * 16) + 10];
	mat[5] = arreglo_b[(tid * 16) + 11];
	mat[6] = arreglo_b[(tid * 16) + 12];
	mat[7] = arreglo_b[(tid * 16) + 14];
	mat[8] = arreglo_b[(tid * 16) + 15];
	
	det0 = mat[0]*(mat[4]*mat[8] - mat[5]*mat[7]);
	det1 = mat[1]*(mat[3]*mat[8] - mat[5]*mat[6]);
	det2 = mat[2]*(mat[3]*mat[7] - mat[4]*mat[6]);
	
	float result1 = det0 - det1 + det2;
	result1 *= arreglo_b[1];

	mat[0] = arreglo_b[(tid * 16) + 4];
	mat[1] = arreglo_b[(tid * 16) + 5];
	mat[2] = arreglo_b[(tid * 16) + 7];
	mat[3] = arreglo_b[(tid * 16) + 8];
	mat[4] = arreglo_b[(tid * 16) + 9];
	mat[5] = arreglo_b[(tid * 16) + 11];
	mat[6] = arreglo_b[(tid * 16) + 12];
	mat[7] = arreglo_b[(tid * 16) + 13];
	mat[8] = arreglo_b[(tid * 16) + 15];
	
	det0 = mat[0]*(mat[4]*mat[8] - mat[5]*mat[7]);
	det1 = mat[1]*(mat[3]*mat[8] - mat[5]*mat[6]);
	det2 = mat[2]*(mat[3]*mat[7] - mat[4]*mat[6]);
	
	float result2 = det0 - det1 + det2;
	result2 *= arreglo_b[2];

	mat[0] = arreglo_b[(tid * 16) + 4];
	mat[1] = arreglo_b[(tid * 16) + 5];
	mat[2] = arreglo_b[(tid * 16) + 6];
	mat[3] = arreglo_b[(tid * 16) + 8];
	mat[4] = arreglo_b[(tid * 16) + 9];
	mat[5] = arreglo_b[(tid * 16) + 10];
	mat[6] = arreglo_b[(tid * 16) + 12];
	mat[7] = arreglo_b[(tid * 16) + 13];
	mat[8] = arreglo_b[(tid * 16) + 14];
	
	det0 = mat[0]*(mat[4]*mat[8] - mat[5]*mat[7]);
	det1 = mat[1]*(mat[3]*mat[8] - mat[5]*mat[6]);
	det2 = mat[2]*(mat[3]*mat[7] - mat[4]*mat[6]);
	
	float result3 = det0 - det1 + det2;
	result3 *= arreglo_b[3];

	float result_total = result0 - result1 + result2 - result3;
	arreglo_a[tid] = result_total;
	
}

// realiza la suma total de forma paralela, aumentando el offset en cada ejecucion
__global__ void sumador(int* arreglo, int offset, int N){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < N){

		if( (tid & ( (offset * 2) -1)) == 0 && ( (tid+offset) < N))
		{

			arreglo[tid] = arreglo[tid] + arreglo[tid + offset];

		}


	}

}


// realiza la suma de n matrices
__global__ void sumador_2(int* arreglo, int offset, int N){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < N*16){

		if( (( (tid / 16) & (offset*2)-1 ) == 0) && ( (tid+offset*16) < N*16))
		{
			arreglo[tid] = arreglo[tid] + arreglo[tid + offset * 16];
		}
	}
}


void multiplicar(double num, int* mat, double mat_res[16])
{
	for(int i=0; i< 16; i++)
	{
		mat_res[i] = mat[i] * num;
	}
}

int main(int argc, char** argv){
	int numBytes = sizeof(int) * N; //bytes a alocar
	int num = sizeof(int)*16;

	suma_det = (int *) malloc(sizeof(int)); 
	arreglo_A = (int *) malloc(numBytes);
	arreglo_B = (int *) malloc(numBytes * 16);
	arreglo_C = (int *) malloc(num);
	
	//double timetick;

	// llenamos los arreglos

	init_CPU_array(arreglo_B, N);


	// allocamos memoria en la gpu

	cudaMalloc(&d_arreglo_B, numBytes * 16);
	cudaMalloc(&d_arreglo_C, numBytes);
	cudaMalloc(&d_arreglo_A, numBytes);



	// copiamos los datos de la cpu a la gpu

	cudaMemcpy(d_arreglo_B, arreglo_B, numBytes * 16, cudaMemcpyHostToDevice);

	dim3 miGrid1D(1,1);
	dim3 miBloque1D(16 * N,1);
	dim3 miBloque1D_determinanteador(N,1);

	// timetick = dwalltime();


	// si tenemos 10 matrises, 10 determinants
	determinanteador<<<miGrid1D, miBloque1D_determinanteador>>>(d_arreglo_B, d_arreglo_A, N);

	for(int i=1; i < N; i*= 2){
		sumador<<<miGrid1D, miBloque1D>>>(d_arreglo_A, i, N);
		cudaThreadSynchronize();


	}



	cudaMemcpy(suma_det, d_arreglo_A, sizeof(int), cudaMemcpyDeviceToHost);

	double promedio_det = (*suma_det) / N;

	// Sumamos todos los elementos para el promedio, el resultado queda almacenado en la primer posicion
	for(int i=1; i < N; i*= 2){
		sumador_2<<<miGrid1D, miBloque1D>>>(d_arreglo_B, i, N);
		cudaThreadSynchronize();


	}
	
	// nos traemos los resultados de la gpu a la cpu
	cudaMemcpy(arreglo_C, d_arreglo_B, num, cudaMemcpyDeviceToHost);

	// printf("-> Tiempo transcurrido en la GPU %f\n", dwalltime() - timetick);


	double mat_res[16];
	multiplicar(promedio_det, arreglo_C, mat_res);

	//imprimimos los resultados
	printf("%s\n", "");
	printf("%s\n", "Promedio determinante:");

	printf("%lf\n", promedio_det);

	

	printf("%s\n", "MATRIZ RESULTANTE: ");
	print_CPU_array_double(mat_res, 16);



	
	
	free(arreglo_B);
	free(arreglo_C);
	

	cudaFree (d_arreglo_B);
	cudaFree (d_arreglo_C);

}