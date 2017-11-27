#include "cuda.h"
#include "stdio.h"

#define threads_per_block 512
#define threads_per_block_matrix 32 // 32*16 = 512


//#include <sys/time.h>
//#include <sys/resource.h>


// double dwalltime(){
//         double sec;
//         struct timeval tv;

//         gettimeofday(&tv,NULL);
//         sec = tv.tv_sec + tv.tv_usec/1000000.0;
//         return sec;
// }




int* arreglo_result;
int* mat_result;
int* suma_det;



int* arreglo_A;
int* d_arreglo_A;

int* arreglo_B;
int* d_arreglo_B;

int* mat_A;
int* d_mat_A;

int* mat_B;
int* d_mat_B;

int* arreglo_determinantes;
int* d_arreglo_determinantes;


void printi(int i){
	printf("%d\n", i);
}

void init_CPU_matrices_array(int* arreglo, int n){
	for(int i=0; i< n; i++)
	{
		//int valor = 1;
		arreglo[(i*16) + 0] = 1;
		arreglo[(i*16) + 1] = 0;
		arreglo[(i*16) + 2] = 0;
		arreglo[(i*16) + 3] = 0;
		arreglo[(i*16) + 4] = 0;
		arreglo[(i*16) + 5] = 1;
		arreglo[(i*16) + 6] = 0;
		arreglo[(i*16) + 7] = 0;
		arreglo[(i*16) + 8] = 0;
		arreglo[(i*16) + 9] = 0;
		arreglo[(i*16) + 10] = 1;
		arreglo[(i*16) + 11] = 0;
		arreglo[(i*16) + 12] = 0;
		arreglo[(i*16) + 13] = 0;
		arreglo[(i*16) + 14] = 0;
		arreglo[(i*16) + 15] = 1;

	}

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

void print_CPU_matrix(int array[], int n){
    for(int i = 0; i < n; i++) {
        if(i % 16 == 0)
            printf("%s\n", "");

        printf("%d ", array[i]);
    }
}



__global__ void determinanteador(int* arreglo_b, int* arreglo_a, int N){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(tid > N)
		return;


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

// realiza la suma de determinantes
__global__ void sumador_determinantes(int* arreglo, int* result, float N)
{
	__shared__ int compartida[threads_per_block];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid > N)
		return;
	
	compartida[threadIdx.x] = arreglo[tid];
	__syncthreads();
	for(int i=1; pow((float)2,(float)i-1) < threads_per_block; i++)
	{
		int acceso = pow((float)2,(float)i);
		int offset = pow((float)2, (float)i-1);
		if(threadIdx.x < ((float)threads_per_block/acceso) && (threadIdx.x * acceso + offset) < (N - blockIdx.x * blockDim.x))
		{
				compartida[threadIdx.x * acceso] = compartida[threadIdx.x * acceso] + compartida[threadIdx.x * acceso + offset];
				compartida[threadIdx.x * acceso + offset] = 0;
		}

	}

	//el primer thread de cada grupo guarda el resultado
	if(threadIdx.x == 0)
		result[blockIdx.x] = compartida[0];

}

__global__ void sumador_matrices(int* arreglo, int* result, float N)
{
	__shared__ int compartida[threads_per_block_matrix * 16];

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

		if(t_id < ((float)threads_per_block_matrix*16/acceso) && (new_offset  < (threads_per_block_matrix*16)))
		{
				

				compartida[new_access] = compartida[new_access] + compartida[new_offset];
				compartida[new_offset] = 0;
				// printf("GRUPO: %d - ITERACION: %d - TID %d - ACCESO: %d - OFFSET %d - REMAINING: %d \n", blockIdx.x,
				// 														i, 			tid, 	new_access , new_offset, threadIdx.x * acceso + offset);
		}


	}

	//el primer thread de cada grupo guarda el resultado
	if(threadIdx.x < 16)
		result[blockIdx.x * 16 + threadIdx.x] = compartida[threadIdx.x];

}

int main(int argc, char** argv){

	int N = 256;

	int numBytesMatrices = sizeof(int) * N * 16; //bytes a alocar
	int numBytesDeterminantes = sizeof(int) * N; //bytes a alocar

	//##################################################################################
	//############################## INICIALIZACION ####################################

	suma_det = (int *) malloc(sizeof(int)); 

	arreglo_A = (int *) malloc(numBytesDeterminantes);	
	arreglo_B = (int *) malloc(numBytesDeterminantes);	


	mat_A = (int *) malloc(numBytesMatrices);
	mat_B = (int *) malloc(numBytesMatrices);

	arreglo_determinantes = (int*) malloc(numBytesDeterminantes);

	cudaMalloc(&d_arreglo_determinantes, numBytesDeterminantes);
	cudaMalloc(&d_arreglo_A, numBytesDeterminantes);
	cudaMalloc(&d_arreglo_B, numBytesDeterminantes);

	cudaMalloc(&d_mat_A, numBytesMatrices);
	cudaMalloc(&d_mat_B, numBytesMatrices);

	init_CPU_matrices_array(mat_A, N);
	cudaMemcpy(d_mat_A, mat_A, numBytesMatrices, cudaMemcpyHostToDevice);


	//##################################################################################
	//################################ EJECUCIONES #####################################


	//################################ DETERMINANTE ####################################
	dim3 miGrid1D_determinanteador(ceil((float)N/threads_per_block),1);
	dim3 miBloque1D_determinanteador(threads_per_block,1);

	determinanteador<<<miGrid1D_determinanteador,miBloque1D_determinanteador>>>(d_mat_A, d_arreglo_A, N);
	cudaThreadSynchronize();
	// printf("ERROR %s\n", cudaGetErrorString(cudaGetLastError()));
	// cudaMemcpy(arreglo_determinantes, d_arreglo_A, numBytesDeterminantes, cudaMemcpyDeviceToHost);
	// print_CPU_matrix(arreglo_determinantes, N);



	//############################# SUMADOR DETERMINANTE ###############################

	dim3 miBloque1D_sumador(threads_per_block,1);
	for(int i=1; pow(threads_per_block, i-1) < N; i++)
	{
		int remaining_elements = ceil((float)N/pow(threads_per_block, i-1));
		dim3 miGrid1D_sumador(remaining_elements,1);
		sumador_determinantes<<<miGrid1D_sumador, miBloque1D_sumador>>>(d_arreglo_A, d_arreglo_B, remaining_elements);
		cudaThreadSynchronize();
		// printf("ERROR: %s\n", cudaGetErrorString(cudaGetLastError()));

		int* tmp = d_arreglo_A;
		d_arreglo_A = d_arreglo_B;
		d_arreglo_B = tmp;
	}

	// cudaMemcpy(arreglo_determinantes, d_arreglo_A, sizeof(int) * N, cudaMemcpyDeviceToHost);
	// print_CPU_matrix(arreglo_determinantes, N);

	//############################## SUMADOR MATRICES ##################################

	dim3 miBloque1D_sumador_mat(threads_per_block_matrix *16,1);
	for(int i=1; pow(threads_per_block_matrix, i-1) < N; i++)
	{
		int remaining_elements = ceil((float)N/pow(threads_per_block_matrix, i-1));
		int block_count = ceil((float)N/pow(threads_per_block_matrix * 16, i-1));

		dim3 miGrid1D_sumador_mat(remaining_elements,1);
		sumador_matrices<<<miGrid1D_sumador_mat, miBloque1D_sumador_mat>>>(d_mat_A, d_mat_B, remaining_elements);
		cudaThreadSynchronize();
		// printf("ERROR %s\n", cudaGetErrorString(cudaGetLastError()));
		int* tmp = d_mat_A;
		d_mat_A = d_mat_B;
		d_mat_B = tmp;

	}

	//############################### READ BACK ########################################

	// PROMEDIO
	cudaMemcpy(suma_det, d_arreglo_A, sizeof(int), cudaMemcpyDeviceToHost);
	double promedio_det = (float)(*suma_det) / N;
	printf("PROMEDIO: %lf\n", promedio_det);
	
	// SUMA DE MATRICES
	cudaMemcpy(mat_B, d_mat_A, 16 * sizeof(int), cudaMemcpyDeviceToHost);


	for(int i=0; i< 16; i++)
		mat_B[i] *= (int)promedio_det;


	printf("%s\n", "RESULTADO:");
	print_CPU_matrix(mat_B, 16);



	free(arreglo_determinantes);
	free(suma_det);

	free(arreglo_A);
	free(arreglo_B);

	free(mat_A);
	free(mat_B);
	

	cudaFree (d_arreglo_A);
	cudaFree (d_arreglo_B);
	cudaFree (d_arreglo_determinantes);

	cudaFree (d_mat_A);
	cudaFree (d_mat_B);


}


