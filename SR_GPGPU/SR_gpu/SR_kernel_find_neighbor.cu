
texture<int, 2, cudaReadModeElementType> TIR; // ww*hh
texture<int, 2, cudaReadModeElementType> TIG;
texture<int, 2, cudaReadModeElementType> TIB;

texture<int, 2, cudaReadModeElementType> TLR; // w*h
texture<int, 2, cudaReadModeElementType> TLG;
texture<int, 2, cudaReadModeElementType> TLB;

texture<int, 2, cudaReadModeElementType> THR; // w*h
texture<int, 2, cudaReadModeElementType> THG;
texture<int, 2, cudaReadModeElementType> THB;

#include <stdio.h>
#include <time.h>

__device__ int calc_dist(
		int x, int y, int low_x, int low_y, int w, int ww, int min, int *pos,
		texture<int, 2, cudaReadModeElementType> TI, texture<int, 2, cudaReadModeElementType> TL){
	int dist=0;
	
	for(int j=0; j<3; ++j){
		for(int i=0; i<3; ++i){
			dist+=( tex2D(TI, x+i, y+j)-tex2D(TL, low_x+i, low_y+j) )*( tex2D(TI, x+i, y+j)-tex2D(TL, low_x+i, low_y+j) );
		}
	}
	
	if(dist<min){
		pos[0]=low_x;
		pos[1]=low_y;
		return dist;
	}
	else return min;
}

__global__ void find_neighbor(int round, int *dans_R, int *dans_G, int *dans_B, int w, int h, int ww, int hh){
	int tid=blockDim.x * blockIdx.x + threadIdx.x;
	//int tidy=blockDim.y * blockIdx.y + threadIdx.y;
	
	if( round+tid<(ww/3)*(hh/3) ){
		int x=((round+tid)%(ww/3))*3;
		int y=((round+tid)/(ww/3))*3;
		
		int low_x=x*2/3;
		int low_y=y*2/3;
		int min_R=585526; // 255*255*9=585525
		int min_G=585526;
		int min_B=585526;
		int min_pos_R[2];
		int min_pos_G[2];
		int min_pos_B[2];
		
		for(int j=-1; j<=1; ++j){ // find neighbor in 3*3 block
			for(int i=-1; i<=1; ++i){
				if( low_x+i>=0 && low_x+i<=w-3 && low_y+j>=0 && low_y+j<=h-3 ){
					min_R=calc_dist(x, y, low_x+i, low_y+j, w, ww, min_R, min_pos_R, TIR, TLR);
					min_G=calc_dist(x, y, low_x+i, low_y+j, w, ww, min_G, min_pos_G, TIG, TLG);
					min_B=calc_dist(x, y, low_x+i, low_y+j, w, ww, min_B, min_pos_B, TIB, TLB);
				}
			}
		}
		
		for(int j=0; j<3; ++j){
			for(int i=0; i<3; ++i){
				dans_R[(y+j)*ww +x+i]=tex2D(THR, min_pos_R[0]+i, min_pos_R[1]+j);
				dans_G[(y+j)*ww +x+i]=tex2D(THG, min_pos_G[0]+i, min_pos_G[1]+j);
				dans_B[(y+j)*ww +x+i]=tex2D(THB, min_pos_B[0]+i, min_pos_B[1]+j);
			}
		}
	}
	
	__syncthreads();
}

void SR_kernel_find_neighbor(
	int *I_R, int *I_G, int *I_B,
	int *L_R, int *L_G, int *L_B,
	int *H_R, int *H_G, int *H_B,
	int *ans_R, int *ans_G, int *ans_B,
	int w, int h, int ww, int hh){
	
	//int *d_IR, *d_IG, *d_IB; // img(up)
	//int *d_LR, *d_LG, *d_LB; // img(up(down))
	//int *d_HR, *d_HG, *d_HB; // img(original) - img(up(down))
	int *d_ansR, *d_ansG, *d_ansB;
	cudaMalloc((void**)&d_ansR, ww*hh*sizeof(int));
	cudaMalloc((void**)&d_ansG, ww*hh*sizeof(int));
	cudaMalloc((void**)&d_ansB, ww*hh*sizeof(int));

	cudaChannelFormatDesc Desc=cudaCreateChannelDesc<int>();

	cudaArray *d_IR, *d_IG, *d_IB, *d_LR, *d_LG, *d_LB, *d_HR, *d_HG, *d_HB;
	cudaMallocArray(&d_IR, &Desc, ww, hh);
	cudaMallocArray(&d_IG, &Desc, ww, hh);
	cudaMallocArray(&d_IB, &Desc, ww, hh);
	cudaMallocArray(&d_LR, &Desc, w, h);
	cudaMallocArray(&d_LG, &Desc, w, h);
	cudaMallocArray(&d_LB, &Desc, w, h);
	cudaMallocArray(&d_HR, &Desc, w, h);
	cudaMallocArray(&d_HG, &Desc, w, h);
	cudaMallocArray(&d_HB, &Desc, w, h);
	
	cudaMemcpyToArray(d_IR, 0, 0, I_R, ww*hh*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_IG, 0, 0, I_G, ww*hh*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_IB, 0, 0, I_B, ww*hh*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_LR, 0, 0, L_R, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_LG, 0, 0, L_G, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_LB, 0, 0, L_B, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_HR, 0, 0, H_R, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_HG, 0, 0, H_G, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_HB, 0, 0, H_B, w*h*sizeof(int), cudaMemcpyHostToDevice);

	cudaBindTextureToArray(TIR, d_IR);
	cudaBindTextureToArray(TIG, d_IG);
	cudaBindTextureToArray(TIB, d_IB);
	cudaBindTextureToArray(TLR, d_LR);
	cudaBindTextureToArray(TLG, d_LG);
	cudaBindTextureToArray(TLB, d_LB);
	cudaBindTextureToArray(THR, d_HR);
	cudaBindTextureToArray(THG, d_HG);
	cudaBindTextureToArray(THB, d_HB);

	int threads=1024;
	int blocks=64;
	for(int i=0; i<((ww/3)*(hh/3)-1)/(threads*blocks) +1; ++i){
		find_neighbor<<<blocks, threads>>>(i*threads*blocks, d_ansR, d_ansG, d_ansB, w, h, ww, hh);
		
		//printf("error1: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		//printf("error2: %s\n", cudaGetErrorString(cudaThreadSynchronize()));
	}

	cudaMemcpy(ans_R, d_ansR, ww*hh*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(ans_G, d_ansG, ww*hh*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(ans_B, d_ansB, ww*hh*sizeof(int), cudaMemcpyDeviceToHost);

	cudaUnbindTexture(TIR);
	cudaUnbindTexture(TIG);
	cudaUnbindTexture(TIB);
	cudaUnbindTexture(TLR);
	cudaUnbindTexture(TLG);
	cudaUnbindTexture(TLB);
	cudaUnbindTexture(THR);
	cudaUnbindTexture(THG);
	cudaUnbindTexture(THB);

	cudaFreeArray(d_IR);
	cudaFreeArray(d_IG);
	cudaFreeArray(d_IB);
	cudaFreeArray(d_LR);
	cudaFreeArray(d_LG);
	cudaFreeArray(d_LB);
	cudaFreeArray(d_HR);
	cudaFreeArray(d_HG);
	cudaFreeArray(d_HB);
	cudaFree(d_ansR);
	cudaFree(d_ansG);
	cudaFree(d_ansB);
}
