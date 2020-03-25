
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "time.h" 
#include <stdio.h>
#include "Particles.h"
#include "Turbulences.h"


using namespace std;

#define D 3
#define N 200
#define K 512
#define Nt 20
#define Rt 0.1f
#define c 0.001f
#define ct 0.0001f

__global__ void NextQTur(float* Qt, float* Pt) {
	int i = threadIdx.x;
	Qt[i + 0] += Pt[i + 0] * ct;
	Qt[i + 1] += Pt[i + 1] * ct;
	Qt[i + 2] += Pt[i + 2] * ct;
}

__global__ void Sqrt(float* Q, float* P, float* Qt, float* Pt, float* Eg, float* Epg) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	int i = x * K * D + y * D;
	//int z = threadIdx.z;
	//printf("I = %i \n", x);
	for (int j = 0; j < 3; j++) {
		Q[i + j] = 0.01;
		Qt[i + j] = 0.6;
		P[i + j] = 0.3;
		Pt[i + j] = 0.5;
		Epg[i / D ] = 100000;
		Eg[i / D ] = 0.5;
	}
}

__global__ void addcuda(float* Q, float* P, float* Qt, float* Pt, float* Eg, float* Epg) {
	for (int j = 0; j < 10; j++) {
		int x = blockIdx.x;
		int y = threadIdx.x;
		int i = x * K * D + y * D;

		float Px = P[i + 0];
		float Py = P[i + 1];
		float Pz = P[i + 2];
		float E = Eg[i/3];
		float Ep = Epg[i/3];

		float Qx = Q[i + 0];
		float Qy = Q[i + 1];
		float Qz = Q[i + 2];

		float nQx = Q[i + 0] + c * P[i + 0];
		float nQy = Q[i + 1] + c * P[i + 1];
		float nQz = Q[i + 2] + c * P[i + 2];

		// Отражение от стенок области

		if ((nQx > 1) || (nQx < 0)) {
			Px = (-1) * Px;
		}
		if ((nQy > 1) || (nQy < 0)) {
			Py = (-1) * Py;
		}
		if ((nQz > 1) || (nQz < 0)) {
			Pz = (-1) * Pz;
		}

		// Отражение от турбулентностей

		for (int nt = 0; nt < Nt; nt += 1) {
			float Range = (sqrt(pow(Qx - Qt[nt + 0], 2) + pow(Qy - Qt[nt + 1], 2) + pow(Qz - Qt[nt + 2], 2)));
			float nRange = (sqrt(pow(nQx - Qt[nt + 0], 2) + pow(nQy - Qt[nt + 1], 2) + pow(nQz - Qt[nt + 2], 2)));

			if((Range > Rt) && (nRange < Rt)) {
				float DirX = (nQx - Qt[nt + 0]) / Range;
				float DirY = (nQy - Qt[nt + 1]) / Range;
				float DirZ = (nQz - Qt[nt + 2]) / Range;
				float PnormKoe = ((Px * DirX) + (Py * DirY) + (Pz * DirZ));
				float Pnormt = ((Pt[nt + 0] * DirX) + (Pt[nt + 1] * DirY) + (Pt[nt + 2] * DirZ));
				E -= (ct / c) * (PnormKoe * PnormKoe) * (Pnormt * abs(Pnormt));
				Px -= 2 * DirX;
				Py -= 2 * DirY;
				Pz -= 2 * DirZ;
			}
		}
		// Частица вылетает из области, записывается ее энергия и сбрасывается до начального значения. 
		// Частица продолжает двигаться по траектории
		// Ep случайная величина линейно зависящая от энергии
		if ((nQz > 1) && (E > Ep)) {
			E = 100.0f;
		}
		// Адиабатическое охлаждение
		if (nQz > 0.5) {
			E -= 0.0001f;
		}
		//Приращение энергии при пересечении центра
		if (((nQz > 0.5f) && (Qz < 0.5f)) || ((Qz > 0.5f) && (nQz < 0.5f))) {
			E += 1.0f;
		}

		// Запись в память
		Q[i + 0] = nQx;
		Q[i + 1] = nQy;
		Q[i + 2] = nQz;

		P[i + 0] = Px;
		P[i + 1] = Py;
		P[i + 2] = Pz;
		Eg[i/3] = E;
	}
}

int main() {
	// определяем указатель на файл
	FILE* fout;
	// открываем файл на чтение
	fout = fopen("DataE.txt", "w");

	float* Q = 0;
	cudaMalloc((void**)&Q, N * K * D * sizeof(float));
	float* P = 0;
	cudaMalloc((void**)&P, N * K * D * sizeof(float));
	float* Qt = 0;
	cudaMalloc((void**)&Qt, N * K * D * sizeof(float));
	float* Pt = 0;
	cudaMalloc((void**)&Pt, N * K * D * sizeof(float));
	float* E = 0;
	cudaMalloc((void**)&E, N * K * sizeof(float));
	float* Ep = 0;
	cudaMalloc((void**)&Ep, N * K * sizeof(float));
	 
	unsigned int start_time = clock();

	

	RandomGenQu(Q, N * K * 3, 0, 9);
	RandomGenPu(P, N * K * 3, 0, 23);
	RandomGenEpu(Ep, N * K, 0, 5, 1000.0f , 10000000.0f);
	ConstEu << <N, K >> > (E);

	RandomGenQ(Qt, Nt * 3, 0, 7);
	RandomGenP(Pt, Nt * 3, 0, 8);
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// asynchronously issue work to the GPU (all to stream 0)
	cudaEventRecord(start, 0);
	//Sqrt <<< N, K >>> (Q, P, Qt, Pt, E, Ep);
	float* f = new float[N * K];
	for (int k = 0; k < 100; k++) {
		cudaDeviceSynchronize();
		if (k == 99) {
			cudaMemcpy(f, E, N * K * sizeof(float), cudaMemcpyDeviceToHost);
			for (int j = 0; j < K * N; j++) {
				fprintf(fout, "%f  ", f[j]);
			}
			fprintf(fout, "\n");
		}
		for (int i = 0; i < 50; i++) {
			addcuda << < N, K >> > (Q, P, Qt, Pt, E, Ep);
			NextQTur << < 1, Nt >> > (Qt, Pt);
		}
	}
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	// print the cpu and gpu times
	printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime);

	
	cudaDeviceSynchronize();
	unsigned int end_time = clock(); // конечное время
	unsigned int search_time = end_time - start_time; // искомое время
	
	
	printf("\n Time %i ms", search_time);

	return 0;
}