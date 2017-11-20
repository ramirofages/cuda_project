#include "cuda.h"
#include "stdio.h"
#include <math.h>

#include <sys/time.h>
#include <sys/resource.h>


double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}



double* arreglo;
double* suma_total;

double* d_arreglo_suma;
double* d_arreglo;
double* d_arreglo_2;

int cant_elem = 2048000;

void init_CPU_array_float(double* v, int n){
	for(int i = 0; i < n; i++) {
		v[i] = (double)i;
	}
}

// realiza la suma total de forma paralela, aumentando el offset en cada ejecucion
__global__ void sumador(double* arreglo, int offset, int N){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < N){

		if( (tid & ( (offset * 2) -1)) == 0 && ( (tid+offset) < N))
		{

			arreglo[tid] = arreglo[tid] + arreglo[tid + offset];

		}


	}

}

// (V[i] +/- promedio)^2 --- multiplicador es 1 o -1 dependiendo de si se quiere sumar o restar
__global__ void suma_prom(double* arreglo, int multiplicador, double promedio, int N){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < N)
	{
		double num = arreglo[tid] + (multiplicador * promedio);
		arreglo[tid] = num * num;
	}
}

double solucion_CPU(){

	double prom = 0;

	// sumamos todos los elementos para calcular el promedio
	for (int i = 0; i < cant_elem; ++i)
	{
		prom = prom + arreglo[i];
	}
	prom /= cant_elem;

	double dividendo = 0;
	double divisor = 0;

	// realizamos la sumatoria del dividendo y divisor
	for (int i = 0; i < cant_elem; ++i)
	{
		double num = arreglo[i] - prom;
		double num2 = arreglo[i] + prom;

		dividendo +=  (num*num);
		divisor += (num2*num2);
	}

	divisor = divisor + 1;

	return sqrt(dividendo/divisor);
}


int main(int argc, char** argv){



	double timetick;
	int numBytes = sizeof(double) * cant_elem; //bytes a alocar

	arreglo = (double*) malloc(numBytes);
	suma_total = (double *) malloc(sizeof(double));  // usado para traer el primer elemento resultado de la suma paralela

	init_CPU_array_float(arreglo, cant_elem);



	cudaMalloc(&d_arreglo, numBytes);
	cudaMalloc(&d_arreglo_2, numBytes);
	cudaMalloc(&d_arreglo_suma, numBytes);


	cudaMemcpy(d_arreglo, arreglo, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_arreglo_2, arreglo, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_arreglo_suma, arreglo, numBytes, cudaMemcpyHostToDevice);

	dim3 miGrid1D(4000,1);
	dim3 miBloque1D(512,1);




	timetick = dwalltime();

	cudaError_t error;
	// Sumamos todos los elementos para el promedio, el resultado queda almacenado en la primer posicion
	for(int i=1; i < cant_elem; i*= 2){
		sumador<<<miGrid1D, miBloque1D>>>(d_arreglo_suma, i, cant_elem);
		cudaThreadSynchronize();

	}



	// Esperamos a que termine la ejecucion
 	
 	//printf("%d\n", error);

 	// Traemos el primer elemento de d_arreglo_suma el cual posee el resultado
	cudaMemcpy(suma_total, d_arreglo_suma, sizeof(double), cudaMemcpyDeviceToHost);


	double promedio = (*suma_total) / cant_elem;


	// ############################################
	// ############################################
	// Dividendo
	suma_prom<<<miGrid1D, miBloque1D>>>(d_arreglo_2, -1, promedio, cant_elem);

	for(int i=1; i < cant_elem; i*=2){
		sumador<<<miGrid1D, miBloque1D>>>(d_arreglo_2, i, cant_elem);
	}

	error = cudaThreadSynchronize();
 	//printf("%d\n", error);
	cudaMemcpy(suma_total, d_arreglo_2, sizeof(double), cudaMemcpyDeviceToHost);

	double dividendo = (*suma_total);

	// ############################################
	// ############################################
	// Divisor
	suma_prom<<<miGrid1D, miBloque1D>>>(d_arreglo, 1, promedio, cant_elem);

	for(int i=1; i < cant_elem; i*=2){
		sumador<<<miGrid1D, miBloque1D>>>(d_arreglo, i, cant_elem);
	}

	error = cudaThreadSynchronize();
 	//printf("%d\n", error);
	printf("-> Tiempo transcurrido en la GPU %f\n", dwalltime() - timetick);

	cudaMemcpy(suma_total, d_arreglo, sizeof(double), cudaMemcpyDeviceToHost);

	double divisor = *suma_total + 1;
	// ############################################

	double division = dividendo / divisor;
	double resultado = sqrt(division);

	printf("Resultado GPU: %f\n", resultado);

	timetick = dwalltime();

	double cpu_result = solucion_CPU();

	printf("-> Tiempo transcurrido en la CPU %f\n", dwalltime() - timetick);
	printf("Resultado CPU: %f\n", cpu_result);


	free(arreglo);
	free(suma_total);
	cudaFree (d_arreglo);
	cudaFree (d_arreglo_2);
	cudaFree (d_arreglo_suma);

}
