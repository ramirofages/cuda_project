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
		arreglo[(i*16) + 0] = 5;
		arreglo[(i*16) + 1] = 5;
		arreglo[(i*16) + 2] = 3;
		arreglo[(i*16) + 3] = 4;
		arreglo[(i*16) + 4] = 12;
		arreglo[(i*16) + 5] = 3;
		arreglo[(i*16) + 6] = 4;
		arreglo[(i*16) + 7] = 5;
		arreglo[(i*16) + 8] = 6;
		arreglo[(i*16) + 9] = 7;
		arreglo[(i*16) + 10] = 8;
		arreglo[(i*16) + 11] = 9;
		arreglo[(i*16) + 12] = 10;
		arreglo[(i*16) + 13] = 1;
		arreglo[(i*16) + 14] = 2;
		arreglo[(i*16) + 15] = 3;

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




int main(int argc, char** argv){

	int N = 520;
	int threads_per_block_determinanteador = 512;


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
	dim3 miGrid1D_suma_determinantes(ceil((float)N/threads_per_block_determinanteador),1);
	dim3 miBloque1D_suma_determinantes(threads_per_block_determinanteador,1);

	determinanteador<<<miGrid1D_suma_determinantes,miBloque1D_suma_determinantes>>>(d_mat_A, d_arreglo_determinantes, N);
	cudaThreadSynchronize();
	// printf("ERROR %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(arreglo_determinantes, d_arreglo_determinantes, numBytesDeterminantes, cudaMemcpyDeviceToHost);
	print_CPU_array(arreglo_determinantes, N);








	// //double timetick;

	// // llenamos los arreglos

	// init_CPU_matrices_array(arreglo_B, N);


	// // allocamos memoria en la gpu

	// cudaMalloc(&d_arreglo_B, numBytes * 16);
	// cudaMalloc(&d_arreglo_C, numBytes);
	// cudaMalloc(&d_arreglo_A, numBytes);



	// // copiamos los datos de la cpu a la gpu

	// cudaMemcpy(d_arreglo_B, arreglo_B, numBytes * 16, cudaMemcpyHostToDevice);

	// dim3 miGrid1D(1,1);
	// dim3 miBloque1D(16 * N,1);
	// dim3 miBloque1D_determinanteador(N,1);

	// // timetick = dwalltime();


	// // si tenemos 10 matrises, 10 determinants
	// determinanteador<<<miGrid1D, miBloque1D_determinanteador>>>(d_arreglo_B, d_arreglo_A, N);

	// for(int i=1; i < N; i*= 2){
	// 	sumador<<<miGrid1D, miBloque1D>>>(d_arreglo_A, i, N);
	// 	cudaThreadSynchronize();


	// }



	// cudaMemcpy(suma_det, d_arreglo_A, sizeof(int), cudaMemcpyDeviceToHost);

	// double promedio_det = (*suma_det) / N;

	// // Sumamos todos los elementos para el promedio, el resultado queda almacenado en la primer posicion
	// for(int i=1; i < N; i*= 2){
	// 	sumador_2<<<miGrid1D, miBloque1D>>>(d_arreglo_B, i, N);
	// 	cudaThreadSynchronize();


	// }
	
	// // nos traemos los resultados de la gpu a la cpu
	// cudaMemcpy(arreglo_C, d_arreglo_B, num, cudaMemcpyDeviceToHost);

	// // printf("-> Tiempo transcurrido en la GPU %f\n", dwalltime() - timetick);


	// double mat_res[16];
	// multiplicar(promedio_det, arreglo_C, mat_res);

	// //imprimimos los resultados
	// // printf("%s\n", "");
	// // printf("%s\n", "Promedio determinante:");

	// // printf("%lf\n", promedio_det);

	

	// // printf("%s\n", "MATRIZ RESULTANTE: ");
	// // print_CPU_array_double(mat_res, 16);

	
	// // Sumamos todos los elementos para el promedio, el resultado queda almacenado en la primer posicion









	// int txb=10;

	// dim3 miGrid1D_suma_4(2,1);
	// dim3 miBloque1D_suma_4(txb,1);
	// for(int i=1; i <= cant_elem/txb; i*= 2){
	// 	sumador_4<<<miGrid1D_suma_4, miBloque1D_suma_4>>>(d_arreglo_suma1, d_arreglo_suma2, i, cant_elem, txb);
	// 	cudaThreadSynchronize();

	// 	int* temp = d_arreglo_suma1;
	// 	d_arreglo_suma1 = d_arreglo_suma2;
	// 	d_arreglo_suma2 = tmp;
	// }


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


