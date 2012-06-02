//#include "SR_kernel_start.cu"
#include <stdio.h>
#include <time.h>

texture<int, 1, cudaReadModeElementType> TR;
texture<int, 1, cudaReadModeElementType> TG;
texture<int, 1, cudaReadModeElementType> TB;
texture<int ,1, cudaReadModeElementType> TansR;
texture<int ,1, cudaReadModeElementType> TansG;
texture<int ,1, cudaReadModeElementType> TansB;

//extern __shared__ int row[];
__constant__ float d_u0[5];
__constant__ float d_u1[5];

extern "C" void set_filter_up(float *u0, float *u1){
	cudaMemcpyToSymbol(d_u0, u0, 5 * sizeof(float));
	cudaMemcpyToSymbol(d_u1, u1, 5 * sizeof(float));
}

__device__ int convolusion_col(int index, int ww, int hh, int *ans, int *row0, int *row1){
	int e_aft=0;
	int temp[2];
	// i==0
	temp[0]=(int)(d_u0[2]*row0[0]+d_u0[3]*row0[1]+d_u0[4]*row0[2]);
	temp[1]=(int)(d_u1[2]*row1[0]+d_u1[3]*row1[1]+d_u1[4]*row1[2]);
	ans[index]=temp[0]+temp[1];
	e_aft+=(temp[0]+temp[1]);
	// i==1
	temp[0]=(int)(d_u0[1]*row0[0]+d_u0[2]*row0[1]+d_u0[3]*row0[2]+d_u0[4]*row0[3]);
	temp[1]=(int)(d_u1[1]*row1[0]+d_u1[2]*row1[1]+d_u1[3]*row1[2]+d_u1[4]*row1[3]);
	ans[ww +index]=temp[0]+temp[1];
	e_aft+=(temp[0]+temp[1]);
	// i==hh-2
	temp[0]=(int)(d_u0[0]*row0[hh-4]+d_u0[1]*row0[hh-3]+d_u0[2]*row0[hh-2]+d_u0[3]*row0[hh-1]);
	temp[1]=(int)(d_u1[0]*row1[hh-4]+d_u1[1]*row1[hh-3]+d_u1[2]*row1[hh-2]+d_u1[3]*row1[hh-1]);
	ans[(hh-2)*ww +index]=temp[0]+temp[1];
	e_aft+=(temp[0]+temp[1]);
	// i==hh-1
	temp[0]=(int)(d_u0[0]*row0[hh-3]+d_u0[1]*row0[hh-2]+d_u0[2]*row0[hh-1]);
	temp[1]=(int)(d_u1[0]*row1[hh-3]+d_u1[1]*row1[hh-2]+d_u1[2]*row1[hh-1]);
	ans[(hh-1)*ww +index]=temp[0]+temp[1];
	e_aft+=(temp[0]+temp[1]);
	//#pragma unroll
	for(int i=2; i<hh-2; ++i){
		temp[0]=(int)(d_u0[0]*row0[i-2]+d_u0[1]*row0[i-1]+d_u0[2]*row0[i]+d_u0[3]*row0[i+1]+d_u0[4]*row0[i+2]);
		temp[1]=(int)(d_u1[0]*row1[i-2]+d_u1[1]*row1[i-1]+d_u1[2]*row1[i]+d_u1[3]*row1[i+1]+d_u1[4]*row1[i+2]);
		
		ans[i*ww +index]=temp[0]+temp[1];
		e_aft+=(temp[0]+temp[1]);
	}
	return e_aft;
}

__device__ int convolusion_row(int a_index, int w, int ww, int *ans, int index, int *row0, int *row1){
	int e_aft=0;
	int temp[2];
	
	// i==0
	temp[0]=(int)(d_u0[2]*row0[0]+d_u0[3]*row0[1]+d_u0[4]*row0[2]);
	temp[1]=(int)(d_u1[2]*row1[0]+d_u1[3]*row1[1]+d_u1[4]*row1[2]);
	ans[a_index*ww]=temp[0]+temp[1];
	e_aft+=(temp[0]+temp[1]);
	// i==1
	temp[0]=(int)(d_u0[1]*row0[0]+d_u0[2]*row0[1]+d_u0[3]*row0[2]+d_u0[4]*row0[3]);
	temp[1]=(int)(d_u1[1]*row1[0]+d_u1[2]*row1[1]+d_u1[3]*row1[2]+d_u1[4]*row1[3]);
	ans[a_index*ww +1]=temp[0]+temp[1];
	e_aft+=(temp[0]+temp[1]);
	// i==ww-2
	temp[0]=(int)(d_u0[0]*row0[ww-4]+d_u0[1]*row0[ww-3]+d_u0[2]*row0[ww-2]+d_u0[3]*row0[ww-1]);
	temp[1]=(int)(d_u1[0]*row1[ww-4]+d_u1[1]*row1[ww-3]+d_u1[2]*row1[ww-2]+d_u1[3]*row1[ww-1]);
	ans[a_index*ww +ww-2]=temp[0]+temp[1];
	e_aft+=(temp[0]+temp[1]);
	// i==ww-1
	temp[0]=(int)(d_u0[0]*row0[ww-3]+d_u0[1]*row0[ww-2]+d_u0[2]*row0[ww-1]);
	temp[1]=(int)(d_u1[0]*row1[ww-3]+d_u1[1]*row1[ww-2]+d_u1[2]*row1[ww-1]);
	ans[a_index*ww +ww-1]=temp[0]+temp[1];
	e_aft+=(temp[0]+temp[1]);
	
	//#pragma unroll
	for(int i=2; i<ww-2; ++i){
		temp[0]=(int)(d_u0[0]*row0[i-2]+d_u0[1]*row0[i-1]+d_u0[2]*row0[i]+d_u0[3]*row0[i+1]+d_u0[4]*row0[i+2]);
		temp[1]=(int)(d_u1[0]*row1[i-2]+d_u1[1]*row1[i-1]+d_u1[2]*row1[i]+d_u1[3]*row1[i+1]+d_u1[4]*row1[i+2]);
		ans[a_index*ww +i]=temp[0]+temp[1];
		e_aft+=(temp[0]+temp[1]);
	}
	return e_aft;
}

__global__ void run_cuda_col(int round, int *ans_R, int *ans_G, int *ans_B, int w, int h, int ww, int hh){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(round+tid<ww){
		int R_ori=0, G_ori=0, B_ori=0; // store weight of original img
		int e_aft;
		float R_rate, G_rate, B_rate;
		int index=(round+tid)*h;
		//#pragma unroll
		for(int i=0; i<h; ++i){ // compute weight
			R_ori+=tex1Dfetch(TansR, index +i);
			G_ori+=tex1Dfetch(TansG, index +i);
			B_ori+=tex1Dfetch(TansB, index +i);
			/*
			R_ori+=ans_R[i*ww +index];
			G_ori+=ans_G[i*ww +index];
			B_ori+=ans_B[i*ww +index];
			*/
		}
		int row0[1080];
		int row1[1080];
		// red
		//#pragma unroll
		for(int i=0; i<hh; ++i){
			
			if(i%3==0) row0[i]=tex1Dfetch(TansR, index +i*2/3);
			else row0[i]=0;

			if(i%3==2) row1[i]=tex1Dfetch(TansR, index +(i-2)*2/3+1);
			else row1[i]=0;
			
			/*
			if(i%3==0) row0[i]=ans_R[(i*2/3)*ww +index];
			else row0[i]=0;

			if(i%3==2) row1[i]=ans_R[((i-2)*2/3+1)*ww +index];
			else row1[i]=0;
			*/
		}
		e_aft=convolusion_col(round+tid, ww, hh, ans_R, row0, row1);
		R_rate=(float)e_aft/(float)(R_ori*3/2);
		// green
		//#pragma unroll
		for(int i=0; i<hh; ++i){
			
			if(i%3==0) row0[i]=tex1Dfetch(TansG, index +i*2/3);
			else row0[i]=0;

			if(i%3==2) row1[i]=tex1Dfetch(TansG, index +(i-2)*2/3+1);
			else row1[i]=0;
			
			/*
			if(i%3==0) row0[i]=ans_G[(i*2/3)*ww +index];
			else row0[i]=0;

			if(i%3==2) row1[i]=ans_G[((i-2)*2/3+1)*ww +index];
			else row1[i]=0;
			*/
		}
		e_aft=convolusion_col(round+tid, ww, hh, ans_G, row0, row1);
		G_rate=(float)e_aft/(float)(G_ori*3/2);
		// blue
		//#pragma unroll
		for(int i=0; i<hh; ++i){
			
			if(i%3==0) row0[i]=tex1Dfetch(TansB, index +i*2/3);
			else row0[i]=0;

			if(i%3==2) row1[i]=tex1Dfetch(TansB, index +(i-2)*2/3+1);
			else row1[i]=0;
			
			/*
			if(i%3==0) row0[i]=ans_B[(i*2/3)*ww +index];
			else row0[i]=0;

			if(i%3==2) row1[i]=ans_B[((i-2)*2/3+1)*ww +index];
			else row1[i]=0;
			*/
		}
		e_aft=convolusion_col(round+tid, ww, hh, ans_B, row0, row1);
		B_rate=(float)e_aft/(float)(B_ori*3/2);
		
		index=round+tid;
		//#pragma unroll
		for(int i=0; i<hh; ++i){
			ans_R[i*ww +index]=(int)((float)ans_R[i*ww +index]/R_rate);
			ans_G[i*ww +index]=(int)((float)ans_G[i*ww +index]/G_rate);
			ans_B[i*ww +index]=(int)((float)ans_B[i*ww +index]/B_rate);
			/*
			if(ans_R[i*ww +round+tid]>255) ans_R[i*ww +round+tid]=255;
			else if(ans_R[i*ww +round+tid]<0) ans_R[i*ww +round+tid]=0;

			if(ans_G[i*ww +round+tid]>255) ans_G[i*ww +round+tid]=255;
			else if(ans_G[i*ww +round+tid]<0) ans_G[i*ww +round+tid]=0;

			if(ans_B[i*ww +round+tid]>255) ans_B[i*ww +round+tid]=255;
			else if(ans_B[i*ww +round+tid]<0) ans_B[i*ww +round+tid]=0;
			*/
		}
		
	}
}

__global__ void run_cuda_row(int round, int *ans_R, int *ans_G, int *ans_B, int w, int h, int ww, int hh, int *temp_R, int *temp_G, int *temp_B){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	//__shared__ int row[540*2*8];
	if(round+tid<h){
		//test[0]=1139;
		int R_ori=0, G_ori=0, B_ori=0; // store weight of original img
		int e_aft;
		float R_rate, G_rate, B_rate;
		int index=(round+tid)*w;
		//#pragma unroll
		for(int i=0; i<w; ++i){ // compute weight
			R_ori+=tex1Dfetch(TR, index +i);
			G_ori+=tex1Dfetch(TG, index +i);
			B_ori+=tex1Dfetch(TB, index +i);
		}
		
		int row0[1920];
		int row1[1920];
		
		// red
		//#pragma unroll
		for(int i=0; i<ww; ++i){ // setup row
			/*
			if(i%3==0) row[threadIdx.x*ww*2 +i]=tex1Dfetch(TR, index +i*2/3);
			else row[threadIdx.x*ww*2 +i]=0;

			if(i%3==2) row[threadIdx.x*ww*2 +ww+i]=tex1Dfetch(TR, index +(i-2)*2/3+1);
			else row[threadIdx.x*ww*2 +ww+i]=0;
			*/
			if(i%3==0) row0[i]=tex1Dfetch(TR, index +i*2/3);
			else row0[i]=0;

			if(i%3==2) row1[i]=tex1Dfetch(TR, index +(i-2)*2/3+1);
			else row1[i]=0;
			
		}
		e_aft=convolusion_row(round+tid, w, ww, ans_R, threadIdx.x*ww*2, row0, row1);
		R_rate=(float)e_aft/(float)(R_ori*3/2);

		// green
		//#pragma unroll
		for(int i=0; i<ww; ++i){ // setup row
			/*
			if(i%3==0) row[threadIdx.x*ww*2 +i]=tex1Dfetch(TG, index +i*2/3);
			else row[threadIdx.x*ww*2 +i]=0;

			if(i%3==2) row[threadIdx.x*ww*2 +ww+i]=tex1Dfetch(TG, index +(i-2)*2/3+1);
			else row[threadIdx.x*ww*2 +ww+i]=0;
			*/
			
			if(i%3==0) row0[i]=tex1Dfetch(TG, index +i*2/3);
			else row0[i]=0;

			if(i%3==2) row1[i]=tex1Dfetch(TG, index +(i-2)*2/3+1);
			else row1[i]=0;
			
		}
		e_aft=convolusion_row(round+tid, w, ww, ans_G, threadIdx.x*ww*2, row0, row1);
		G_rate=(float)e_aft/(float)(G_ori*3/2);

		// blue
		//#pragma unroll
		for(int i=0; i<ww; ++i){ // setup row
			/*
			if(i%3==0) row[threadIdx.x*ww*2 +i]=tex1Dfetch(TB, index +i*2/3);
			else row[threadIdx.x*ww*2 +i]=0;

			if(i%3==2) row[threadIdx.x*ww*2 +ww+i]=tex1Dfetch(TB, index +(i-2)*2/3+1);
			else row[threadIdx.x*ww*2 +ww+i]=0;
			*/
			
			if(i%3==0) row0[i]=tex1Dfetch(TB, index +i*2/3);
			else row0[i]=0;

			if(i%3==2) row1[i]=tex1Dfetch(TB, index +(i-2)*2/3+1);
			else row1[i]=0;
			
		}
		e_aft=convolusion_row(round+tid, w, ww, ans_B, threadIdx.x*ww*2, row0, row1);
		B_rate=(float)e_aft/(float)(B_ori*3/2);
		
		index=(round+tid)*ww;
		//#pragma unroll
		for(int i=0; i<ww; ++i){
			temp_R[i*h +round+tid]=ans_R[index +i]=(int)((float)ans_R[index +i]/R_rate);
			temp_G[i*h +round+tid]=ans_G[index +i]=(int)((float)ans_G[index +i]/G_rate);
			temp_B[i*h +round+tid]=ans_B[index +i]=(int)((float)ans_B[index +i]/B_rate);

			/*
			if(ans_R[(round+tid)*ww +i]>255) ans_R[(round+tid)*ww +i]=255;
			else if(ans_R[(round+tid)*ww +i]<0) ans_R[(round+tid)*ww +i]=0;

			if(ans_G[(round+tid)*ww +i]>255) ans_G[(round+tid)*ww +i]=255;
			else if(ans_G[(round+tid)*ww +i]<0) ans_G[(round+tid)*ww +i]=0;

			if(ans_B[(round+tid)*ww +i]>255) ans_B[(round+tid)*ww +i]=255;
			else if(ans_B[(round+tid)*ww +i]<0) ans_B[(round+tid)*ww +i]=0;
			*/
		}
	}

	__syncthreads();
}

void SR_kernel_up(int *ori_R, int *ori_G, int *ori_B, int *aft_R, int *aft_G, int *aft_B, int w, int h, int ww, int hh){
	float u0[5]={-0.047, 0.6, 0.927, 0.119, -0.1};
	float u1[5]={-0.1, 0.119, 0.927, 0.6, -0.047};
	
	int *d_ori_R, *d_ori_G, *d_ori_B;
	int *d_ans_R, *d_ans_G, *d_ans_B;
	int *temp_R, *temp_G, *temp_B;

	//printf("in up, w=%d, h=%d, ww=%d, hh=%d\n", w, h, ww, hh);
	cudaMalloc((void**)&d_ori_R, w*h*sizeof(int));
	cudaMalloc((void**)&d_ori_G, w*h*sizeof(int));
	cudaMalloc((void**)&d_ori_B, w*h*sizeof(int));
	cudaMalloc((void**)&temp_R, ww*h*sizeof(int));
	cudaMalloc((void**)&temp_G, ww*h*sizeof(int));
	cudaMalloc((void**)&temp_B, ww*h*sizeof(int));
	cudaMalloc((void**)&d_ans_R, ww*hh*sizeof(int));
	cudaMalloc((void**)&d_ans_G, ww*hh*sizeof(int));
	cudaMalloc((void**)&d_ans_B, ww*hh*sizeof(int));
	
	cudaMemcpy(d_ori_R, ori_R, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ori_G, ori_G, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ori_B, ori_B, w*h*sizeof(int), cudaMemcpyHostToDevice);

	cudaBindTexture(0, TR, d_ori_R);
	cudaBindTexture(0, TG, d_ori_G);
	cudaBindTexture(0, TB, d_ori_B);
	set_filter_up(u0, u1);

	int threads=64;
	int blocks=64;
	for(int i=0; i<(h-1)/(threads*blocks) +1; ++i) // a thread do a row
		run_cuda_row<<<blocks, threads>>>(i*threads*blocks, d_ans_R, d_ans_G, d_ans_B, w, h, ww, hh, temp_R, temp_G, temp_B);
		//run_cuda_row<<<blocks, threads, threads*sizeof(int)*ww*2>>>(i*threads*blocks, d_ans_R, d_ans_G, d_ans_B, w, h, ww, hh, temp_R, temp_G, temp_B);
	
	cudaBindTexture(0, TansR, temp_R);
	cudaBindTexture(0, TansG, temp_G);
	cudaBindTexture(0, TansB, temp_B);

	for(int i=0; i<(ww-1)/(threads*blocks) +1; ++i) // a thread do a column
		run_cuda_col<<<blocks, threads>>>(i*threads*blocks, d_ans_R, d_ans_G, d_ans_B, w, h, ww, hh);
	
	cudaMemcpy(aft_R, d_ans_R, ww*hh*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(aft_G, d_ans_G, ww*hh*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(aft_B, d_ans_B, ww*hh*sizeof(int), cudaMemcpyDeviceToHost);

	
	cudaUnbindTexture(TR);
	cudaUnbindTexture(TG);
	cudaUnbindTexture(TB);
	cudaUnbindTexture(TansR);
	cudaUnbindTexture(TansG);
	cudaUnbindTexture(TansB);
	
	cudaFree(d_ori_R);
	cudaFree(d_ori_G);
	cudaFree(d_ori_B);
	cudaFree(d_ans_R);
	cudaFree(d_ans_G);
	cudaFree(d_ans_B);
	cudaFree(temp_R);
	cudaFree(temp_G);
	cudaFree(temp_B);
}
