//#include "SR_kernel_start.cu"
#include <stdio.h>
#include <time.h>

texture<unsigned char, 1, cudaReadModeElementType> TR;
texture<unsigned char, 1, cudaReadModeElementType> TG;
texture<unsigned char, 1, cudaReadModeElementType> TB;
texture<unsigned char ,1, cudaReadModeElementType> TansR;
texture<unsigned char ,1, cudaReadModeElementType> TansG;
texture<unsigned char ,1, cudaReadModeElementType> TansB;

//extern __shared__ int row[];
__constant__ float d_u0[5];
__constant__ float d_u1[5];

extern "C" void set_filter_up(float *u0, float *u1){
	cudaMemcpyToSymbol(d_u0, u0, 5*sizeof(float));
	cudaMemcpyToSymbol(d_u1, u1, 5*sizeof(float));
}

__device__ int convolusion_col(int index, int ww, int hh, unsigned char *ans, unsigned char *row0, unsigned char *row1){
	int e_aft=0;
	int temp[2];
	int sum;
	// i==0
	temp[0]=(int)(d_u0[2]*(int)row0[0]+d_u0[3]*(int)row0[1]+d_u0[4]*(int)row0[2]);
	temp[1]=(int)(d_u1[2]*(int)row1[0]+d_u1[3]*(int)row1[1]+d_u1[4]*(int)row1[2]);
	sum=temp[0]+temp[1];
	if(sum>255) sum=255;
	else if(sum<0) sum=0;
	ans[index]=(unsigned char)sum;
	e_aft+=sum;
	// i==1
	temp[0]=(int)(d_u0[1]*(int)row0[0]+d_u0[2]*(int)row0[1]+d_u0[3]*(int)row0[2]+d_u0[4]*(int)row0[3]);
	temp[1]=(int)(d_u1[1]*(int)row1[0]+d_u1[2]*(int)row1[1]+d_u1[3]*(int)row1[2]+d_u1[4]*(int)row1[3]);
	sum=temp[0]+temp[1];
	if(sum>255) sum=255;
	else if(sum<0) sum=0;
	ans[ww +index]=(unsigned char)sum;
	e_aft+=sum;
	// i==hh-2
	temp[0]=(int)(d_u0[0]*(int)row0[hh-4]+d_u0[1]*(int)row0[hh-3]+d_u0[2]*(int)row0[hh-2]+d_u0[3]*(int)row0[hh-1]);
	temp[1]=(int)(d_u1[0]*(int)row1[hh-4]+d_u1[1]*(int)row1[hh-3]+d_u1[2]*(int)row1[hh-2]+d_u1[3]*(int)row1[hh-1]);
	sum=temp[0]+temp[1];
	if(sum>255) sum=255;
	else if(sum<0) sum=0;
	ans[(hh-2)*ww +index]=(unsigned char)sum;
	e_aft+=sum;
	// i==hh-1
	temp[0]=(int)(d_u0[0]*(int)row0[hh-3]+d_u0[1]*(int)row0[hh-2]+d_u0[2]*(int)row0[hh-1]);
	temp[1]=(int)(d_u1[0]*(int)row1[hh-3]+d_u1[1]*(int)row1[hh-2]+d_u1[2]*(int)row1[hh-1]);
	sum=temp[0]+temp[1];
	if(sum>255) sum=255;
	else if(sum<0) sum=0;
	ans[(hh-1)*ww +index]=(unsigned char)sum;
	e_aft+=sum;
	//#pragma unroll
	for(int i=2; i<hh-2; ++i){
		temp[0]=(int)(d_u0[0]*(int)row0[i-2]+d_u0[1]*(int)row0[i-1]+d_u0[2]*(int)row0[i]+d_u0[3]*(int)row0[i+1]+d_u0[4]*(int)row0[i+2]);
		temp[1]=(int)(d_u1[0]*(int)row1[i-2]+d_u1[1]*(int)row1[i-1]+d_u1[2]*(int)row1[i]+d_u1[3]*(int)row1[i+1]+d_u1[4]*(int)row1[i+2]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans[i*ww +index]=(unsigned char)sum;
		e_aft+=sum;
	}
	return e_aft;
}

__device__ int convolusion_row(int a_index, int w, int ww, unsigned char *ans, int index, unsigned char *row0, unsigned char *row1){
	int e_aft=0;
	int temp[2];
	int sum;
	
	// i==0
	temp[0]=(int)(d_u0[2]*(int)row0[0]+d_u0[3]*(int)row0[1]+d_u0[4]*(int)row0[2]);
	temp[1]=(int)(d_u1[2]*(int)row1[0]+d_u1[3]*(int)row1[1]+d_u1[4]*(int)row1[2]);
	sum=temp[0]+temp[1];
	if(sum>255) sum=255;
	else if(sum<0) sum=0;
	ans[a_index*ww]=(unsigned char)sum;
	e_aft+=sum;
	// i==1
	temp[0]=(int)(d_u0[1]*(int)row0[0]+d_u0[2]*(int)row0[1]+d_u0[3]*(int)row0[2]+d_u0[4]*(int)row0[3]);
	temp[1]=(int)(d_u1[1]*(int)row1[0]+d_u1[2]*(int)row1[1]+d_u1[3]*(int)row1[2]+d_u1[4]*(int)row1[3]);
	sum=temp[0]+temp[1];
	if(sum>255) sum=255;
	else if(sum<0) sum=0;
	ans[a_index*ww +1]=(unsigned char)sum;
	e_aft+=sum;
	// i==ww-2
	temp[0]=(int)(d_u0[0]*(int)row0[ww-4]+d_u0[1]*(int)row0[ww-3]+d_u0[2]*(int)row0[ww-2]+d_u0[3]*(int)row0[ww-1]);
	temp[1]=(int)(d_u1[0]*(int)row1[ww-4]+d_u1[1]*(int)row1[ww-3]+d_u1[2]*(int)row1[ww-2]+d_u1[3]*(int)row1[ww-1]);
	sum=temp[0]+temp[1];
	if(sum>255) sum=255;
	else if(sum<0) sum=0;
	ans[a_index*ww +ww-2]=(unsigned char)sum;
	e_aft+=sum;
	// i==ww-1
	temp[0]=(int)(d_u0[0]*(int)row0[ww-3]+d_u0[1]*(int)row0[ww-2]+d_u0[2]*(int)row0[ww-1]);
	temp[1]=(int)(d_u1[0]*(int)row1[ww-3]+d_u1[1]*(int)row1[ww-2]+d_u1[2]*(int)row1[ww-1]);
	sum=temp[0]+temp[1];
	if(sum>255) sum=255;
	else if(sum<0) sum=0;
	ans[a_index*ww +ww-1]=(unsigned char)sum;
	e_aft+=sum;
	
	//#pragma unroll
	for(int i=2; i<ww-2; ++i){
		temp[0]=(int)(d_u0[0]*(int)row0[i-2]+d_u0[1]*(int)row0[i-1]+d_u0[2]*(int)row0[i]+d_u0[3]*(int)row0[i+1]+d_u0[4]*(int)row0[i+2]);
		temp[1]=(int)(d_u1[0]*(int)row1[i-2]+d_u1[1]*(int)row1[i-1]+d_u1[2]*(int)row1[i]+d_u1[3]*(int)row1[i+1]+d_u1[4]*(int)row1[i+2]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans[a_index*ww +i]=(unsigned char)sum;
		e_aft+=sum;
	}
	return e_aft;
}

__global__ void run_cuda_col(int round, unsigned char *ans_R, unsigned char *ans_G, unsigned char *ans_B, int w, int h, int ww, int hh){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(round+tid<ww){
		int R_ori=0, G_ori=0, B_ori=0; // store weight of original img
		int e_aft;
		float R_rate, G_rate, B_rate;
		int index=(round+tid)*h;
		//#pragma unroll
		for(int i=0; i<h; ++i){ // compute weight
			R_ori+=(int)tex1Dfetch(TansR, index +i);
			G_ori+=(int)tex1Dfetch(TansG, index +i);
			B_ori+=(int)tex1Dfetch(TansB, index +i);
			/*
			R_ori+=ans_R[i*ww +index];
			G_ori+=ans_G[i*ww +index];
			B_ori+=ans_B[i*ww +index];
			*/
		}
		unsigned char row0[1080];
		unsigned char row1[1080];
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
		//e_aft=convolusion_col(round+tid, ww, hh, ans_R, row0, row1);
		int temp[2];
		int sum;
		sum=0;
		e_aft=0;	
		index=round+tid;
		// i==0
		temp[0]=(int)(d_u0[2]*(int)row0[0]+d_u0[3]*(int)row0[1]+d_u0[4]*(int)row0[2]);
		temp[1]=(int)(d_u1[2]*(int)row1[0]+d_u1[3]*(int)row1[1]+d_u1[4]*(int)row1[2]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_R[index]=(unsigned char)sum;
		e_aft+=sum;
		// i==1
		temp[0]=(int)(d_u0[1]*(int)row0[0]+d_u0[2]*(int)row0[1]+d_u0[3]*(int)row0[2]+d_u0[4]*(int)row0[3]);
		temp[1]=(int)(d_u1[1]*(int)row1[0]+d_u1[2]*(int)row1[1]+d_u1[3]*(int)row1[2]+d_u1[4]*(int)row1[3]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_R[ww +index]=(unsigned char)sum;
		e_aft+=sum;
		// i==hh-2
		temp[0]=(int)(d_u0[0]*(int)row0[hh-4]+d_u0[1]*(int)row0[hh-3]+d_u0[2]*(int)row0[hh-2]+d_u0[3]*(int)row0[hh-1]);
		temp[1]=(int)(d_u1[0]*(int)row1[hh-4]+d_u1[1]*(int)row1[hh-3]+d_u1[2]*(int)row1[hh-2]+d_u1[3]*(int)row1[hh-1]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_R[(hh-2)*ww +index]=(unsigned char)sum;
		e_aft+=sum;
		// i==hh-1
		temp[0]=(int)(d_u0[0]*(int)row0[hh-3]+d_u0[1]*(int)row0[hh-2]+d_u0[2]*(int)row0[hh-1]);
		temp[1]=(int)(d_u1[0]*(int)row1[hh-3]+d_u1[1]*(int)row1[hh-2]+d_u1[2]*(int)row1[hh-1]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_R[(hh-1)*ww +index]=(unsigned char)sum;
		e_aft+=sum;
		//#pragma unroll
		for(int i=2; i<hh-2; ++i){
			temp[0]=(int)(d_u0[0]*(int)row0[i-2]+d_u0[1]*(int)row0[i-1]+d_u0[2]*(int)row0[i]+d_u0[3]*(int)row0[i+1]+d_u0[4]*(int)row0[i+2]);
			temp[1]=(int)(d_u1[0]*(int)row1[i-2]+d_u1[1]*(int)row1[i-1]+d_u1[2]*(int)row1[i]+d_u1[3]*(int)row1[i+1]+d_u1[4]*(int)row1[i+2]);
			sum=temp[0]+temp[1];
			if(sum>255) sum=255;
			else if(sum<0) sum=0;
			ans_R[i*ww +index]=(unsigned char)sum;
			e_aft+=sum;
		}
		// convolution finish

		
		R_rate=(float)e_aft/(float)(R_ori*3/2);
		// green
		//#pragma unroll
		index=(round+tid)*h;
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
		//e_aft=convolusion_col(round+tid, ww, hh, ans_G, row0, row1);
		sum=0;
		e_aft=0;
		index=round+tid;
		// i==0
		temp[0]=(int)(d_u0[2]*(int)row0[0]+d_u0[3]*(int)row0[1]+d_u0[4]*(int)row0[2]);
		temp[1]=(int)(d_u1[2]*(int)row1[0]+d_u1[3]*(int)row1[1]+d_u1[4]*(int)row1[2]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_G[index]=(unsigned char)sum;
		e_aft+=sum;
		// i==1
		temp[0]=(int)(d_u0[1]*(int)row0[0]+d_u0[2]*(int)row0[1]+d_u0[3]*(int)row0[2]+d_u0[4]*(int)row0[3]);
		temp[1]=(int)(d_u1[1]*(int)row1[0]+d_u1[2]*(int)row1[1]+d_u1[3]*(int)row1[2]+d_u1[4]*(int)row1[3]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_G[ww +index]=(unsigned char)sum;
		e_aft+=sum;
		// i==hh-2
		temp[0]=(int)(d_u0[0]*(int)row0[hh-4]+d_u0[1]*(int)row0[hh-3]+d_u0[2]*(int)row0[hh-2]+d_u0[3]*(int)row0[hh-1]);
		temp[1]=(int)(d_u1[0]*(int)row1[hh-4]+d_u1[1]*(int)row1[hh-3]+d_u1[2]*(int)row1[hh-2]+d_u1[3]*(int)row1[hh-1]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_G[(hh-2)*ww +index]=(unsigned char)sum;
		e_aft+=sum;
		// i==hh-1
		temp[0]=(int)(d_u0[0]*(int)row0[hh-3]+d_u0[1]*(int)row0[hh-2]+d_u0[2]*(int)row0[hh-1]);
		temp[1]=(int)(d_u1[0]*(int)row1[hh-3]+d_u1[1]*(int)row1[hh-2]+d_u1[2]*(int)row1[hh-1]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_G[(hh-1)*ww +index]=(unsigned char)sum;
		e_aft+=sum;
		//#pragma unroll
		for(int i=2; i<hh-2; ++i){
			temp[0]=(int)(d_u0[0]*(int)row0[i-2]+d_u0[1]*(int)row0[i-1]+d_u0[2]*(int)row0[i]+d_u0[3]*(int)row0[i+1]+d_u0[4]*(int)row0[i+2]);
			temp[1]=(int)(d_u1[0]*(int)row1[i-2]+d_u1[1]*(int)row1[i-1]+d_u1[2]*(int)row1[i]+d_u1[3]*(int)row1[i+1]+d_u1[4]*(int)row1[i+2]);
			sum=temp[0]+temp[1];
			if(sum>255) sum=255;
			else if(sum<0) sum=0;
			ans_G[i*ww +index]=(unsigned char)sum;
			e_aft+=sum;
		}
		// convolution finish

		G_rate=(float)e_aft/(float)(G_ori*3/2);
		// blue
		//#pragma unroll
		index=(round+tid)*h;
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
		//e_aft=convolusion_col(round+tid, ww, hh, ans_B, row0, row1);
		sum=0;
		e_aft=0;
		index=round+tid;
		// i==0
		temp[0]=(int)(d_u0[2]*(int)row0[0]+d_u0[3]*(int)row0[1]+d_u0[4]*(int)row0[2]);
		temp[1]=(int)(d_u1[2]*(int)row1[0]+d_u1[3]*(int)row1[1]+d_u1[4]*(int)row1[2]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_B[index]=(unsigned char)sum;
		e_aft+=sum;
		// i==1
		temp[0]=(int)(d_u0[1]*(int)row0[0]+d_u0[2]*(int)row0[1]+d_u0[3]*(int)row0[2]+d_u0[4]*(int)row0[3]);
		temp[1]=(int)(d_u1[1]*(int)row1[0]+d_u1[2]*(int)row1[1]+d_u1[3]*(int)row1[2]+d_u1[4]*(int)row1[3]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_B[ww +index]=(unsigned char)sum;
		e_aft+=sum;
		// i==hh-2
		temp[0]=(int)(d_u0[0]*(int)row0[hh-4]+d_u0[1]*(int)row0[hh-3]+d_u0[2]*(int)row0[hh-2]+d_u0[3]*(int)row0[hh-1]);
		temp[1]=(int)(d_u1[0]*(int)row1[hh-4]+d_u1[1]*(int)row1[hh-3]+d_u1[2]*(int)row1[hh-2]+d_u1[3]*(int)row1[hh-1]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_B[(hh-2)*ww +index]=(unsigned char)sum;
		e_aft+=sum;
		// i==hh-1
		temp[0]=(int)(d_u0[0]*(int)row0[hh-3]+d_u0[1]*(int)row0[hh-2]+d_u0[2]*(int)row0[hh-1]);
		temp[1]=(int)(d_u1[0]*(int)row1[hh-3]+d_u1[1]*(int)row1[hh-2]+d_u1[2]*(int)row1[hh-1]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_B[(hh-1)*ww +index]=(unsigned char)sum;
		e_aft+=sum;
		//#pragma unroll
		for(int i=2; i<hh-2; ++i){
			temp[0]=(int)(d_u0[0]*(int)row0[i-2]+d_u0[1]*(int)row0[i-1]+d_u0[2]*(int)row0[i]+d_u0[3]*(int)row0[i+1]+d_u0[4]*(int)row0[i+2]);
			temp[1]=(int)(d_u1[0]*(int)row1[i-2]+d_u1[1]*(int)row1[i-1]+d_u1[2]*(int)row1[i]+d_u1[3]*(int)row1[i+1]+d_u1[4]*(int)row1[i+2]);
			sum=temp[0]+temp[1];
			if(sum>255) sum=255;
			else if(sum<0) sum=0;
			ans_B[i*ww +index]=(unsigned char)sum;
			e_aft+=sum;
		}
		// convolution finish
		B_rate=(float)e_aft/(float)(B_ori*3/2);
		
		index=round+tid;
		//#pragma unroll
		for(int i=0; i<hh; ++i){
			temp[0]=(int)ans_R[i*ww +index];
			temp[0]=(int)((float)temp[0]/R_rate);
			if(temp[0]>255) temp[0]=255;
			else if(temp[0]<0) temp[0]=0;
			ans_R[i*ww +index]=(unsigned char)temp[0];

			temp[0]=(int)ans_G[i*ww +index];
			temp[0]=(int)((float)temp[0]/G_rate);
			if(temp[0]>255) temp[0]=255;
			else if(temp[0]<0) temp[0]=0;
			ans_G[i*ww +index]=(unsigned char)temp[0];

			temp[0]=(int)ans_B[i*ww +index];
			temp[0]=(int)((float)temp[0]/B_rate);
			if(temp[0]>255) temp[0]=255;
			else if(temp[0]<0) temp[0]=0;
			ans_B[i*ww +index]=(unsigned char)temp[0];
			/*
			ans_R[i*ww +index]=(unsigned char)(int)((float)ans_R[i*ww +index]/R_rate);
			ans_G[i*ww +index]=(unsigned char)(int)((float)ans_G[i*ww +index]/G_rate);
			ans_B[i*ww +index]=(unsigned char)(int)((float)ans_B[i*ww +index]/B_rate);
			*/
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

__global__ void run_cuda_row(
	int round,
	unsigned char *ans_R, unsigned char *ans_G, unsigned char *ans_B,
	int w, int h, int ww, int hh,
	unsigned char *temp_R, unsigned char *temp_G, unsigned char *temp_B){
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
			R_ori+=(int)tex1Dfetch(TR, index +i);
			G_ori+=(int)tex1Dfetch(TG, index +i);
			B_ori+=(int)tex1Dfetch(TB, index +i);
		}
		
		unsigned char row0[1920];
		unsigned char row1[1920];
		
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
		//e_aft=convolusion_row(round+tid, w, ww, ans_R, threadIdx.x*ww*2, row0, row1);

		int temp[2];
		int sum;
		e_aft=0;
		index=round+tid;
		// i==0
		temp[0]=(int)(d_u0[2]*(int)row0[0]+d_u0[3]*(int)row0[1]+d_u0[4]*(int)row0[2]);
		temp[1]=(int)(d_u1[2]*(int)row1[0]+d_u1[3]*(int)row1[1]+d_u1[4]*(int)row1[2]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_R[index*ww]=(unsigned char)sum;
		e_aft+=sum;
		// i==1
		temp[0]=(int)(d_u0[1]*(int)row0[0]+d_u0[2]*(int)row0[1]+d_u0[3]*(int)row0[2]+d_u0[4]*(int)row0[3]);
		temp[1]=(int)(d_u1[1]*(int)row1[0]+d_u1[2]*(int)row1[1]+d_u1[3]*(int)row1[2]+d_u1[4]*(int)row1[3]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_R[index*ww +1]=(unsigned char)sum;
		e_aft+=sum;
		// i==ww-2
		temp[0]=(int)(d_u0[0]*(int)row0[ww-4]+d_u0[1]*(int)row0[ww-3]+d_u0[2]*(int)row0[ww-2]+d_u0[3]*(int)row0[ww-1]);
		temp[1]=(int)(d_u1[0]*(int)row1[ww-4]+d_u1[1]*(int)row1[ww-3]+d_u1[2]*(int)row1[ww-2]+d_u1[3]*(int)row1[ww-1]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_R[index*ww +ww-2]=(unsigned char)sum;
		e_aft+=sum;
		// i==ww-1
		temp[0]=(int)(d_u0[0]*(int)row0[ww-3]+d_u0[1]*(int)row0[ww-2]+d_u0[2]*(int)row0[ww-1]);
		temp[1]=(int)(d_u1[0]*(int)row1[ww-3]+d_u1[1]*(int)row1[ww-2]+d_u1[2]*(int)row1[ww-1]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_R[index*ww +ww-1]=(unsigned char)sum;
		e_aft+=sum;
		
		//#pragma unroll
		for(int i=2; i<ww-2; ++i){
			temp[0]=(int)(d_u0[0]*(int)row0[i-2]+d_u0[1]*(int)row0[i-1]+d_u0[2]*(int)row0[i]+d_u0[3]*(int)row0[i+1]+d_u0[4]*(int)row0[i+2]);
			temp[1]=(int)(d_u1[0]*(int)row1[i-2]+d_u1[1]*(int)row1[i-1]+d_u1[2]*(int)row1[i]+d_u1[3]*(int)row1[i+1]+d_u1[4]*(int)row1[i+2]);
			sum=temp[0]+temp[1];
			if(sum>255) sum=255;
			else if(sum<0) sum=0;
			ans_R[index*ww +i]=(unsigned char)sum;
			e_aft+=sum;
		}
		// convolution finish

		R_rate=(float)e_aft/(float)(R_ori*3/2);

		// green
		//#pragma unroll
		index=(round+tid)*w;
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
		//e_aft=convolusion_row(round+tid, w, ww, ans_G, threadIdx.x*ww*2, row0, row1);

		e_aft=0;
		index=round+tid;
		// i==0
		temp[0]=(int)(d_u0[2]*(int)row0[0]+d_u0[3]*(int)row0[1]+d_u0[4]*(int)row0[2]);
		temp[1]=(int)(d_u1[2]*(int)row1[0]+d_u1[3]*(int)row1[1]+d_u1[4]*(int)row1[2]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_G[index*ww]=(unsigned char)sum;
		e_aft+=sum;
		// i==1
		temp[0]=(int)(d_u0[1]*(int)row0[0]+d_u0[2]*(int)row0[1]+d_u0[3]*(int)row0[2]+d_u0[4]*(int)row0[3]);
		temp[1]=(int)(d_u1[1]*(int)row1[0]+d_u1[2]*(int)row1[1]+d_u1[3]*(int)row1[2]+d_u1[4]*(int)row1[3]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_G[index*ww +1]=(unsigned char)sum;
		e_aft+=sum;
		// i==ww-2
		temp[0]=(int)(d_u0[0]*(int)row0[ww-4]+d_u0[1]*(int)row0[ww-3]+d_u0[2]*(int)row0[ww-2]+d_u0[3]*(int)row0[ww-1]);
		temp[1]=(int)(d_u1[0]*(int)row1[ww-4]+d_u1[1]*(int)row1[ww-3]+d_u1[2]*(int)row1[ww-2]+d_u1[3]*(int)row1[ww-1]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_G[index*ww +ww-2]=(unsigned char)sum;
		e_aft+=sum;
		// i==ww-1
		temp[0]=(int)(d_u0[0]*(int)row0[ww-3]+d_u0[1]*(int)row0[ww-2]+d_u0[2]*(int)row0[ww-1]);
		temp[1]=(int)(d_u1[0]*(int)row1[ww-3]+d_u1[1]*(int)row1[ww-2]+d_u1[2]*(int)row1[ww-1]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_G[index*ww +ww-1]=(unsigned char)sum;
		e_aft+=sum;
	
		//#pragma unroll
		for(int i=2; i<ww-2; ++i){
			temp[0]=(int)(d_u0[0]*(int)row0[i-2]+d_u0[1]*(int)row0[i-1]+d_u0[2]*(int)row0[i]+d_u0[3]*(int)row0[i+1]+d_u0[4]*(int)row0[i+2]);
			temp[1]=(int)(d_u1[0]*(int)row1[i-2]+d_u1[1]*(int)row1[i-1]+d_u1[2]*(int)row1[i]+d_u1[3]*(int)row1[i+1]+d_u1[4]*(int)row1[i+2]);
			sum=temp[0]+temp[1];
			if(sum>255) sum=255;
			else if(sum<0) sum=0;
			ans_G[index*ww +i]=(unsigned char)sum;
			e_aft+=sum;
		}
		// convolution finish
		G_rate=(float)e_aft/(float)(G_ori*3/2);

		// blue
		//#pragma unroll
		index=(round+tid)*w;
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
		//e_aft=convolusion_row(round+tid, w, ww, ans_B, threadIdx.x*ww*2, row0, row1);

		e_aft=0;
		index=round+tid;
		// i==0
		temp[0]=(int)(d_u0[2]*(int)row0[0]+d_u0[3]*(int)row0[1]+d_u0[4]*(int)row0[2]);
		temp[1]=(int)(d_u1[2]*(int)row1[0]+d_u1[3]*(int)row1[1]+d_u1[4]*(int)row1[2]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_B[index*ww]=(unsigned char)sum;
		e_aft+=sum;
		// i==1
		temp[0]=(int)(d_u0[1]*(int)row0[0]+d_u0[2]*(int)row0[1]+d_u0[3]*(int)row0[2]+d_u0[4]*(int)row0[3]);	
		temp[1]=(int)(d_u1[1]*(int)row1[0]+d_u1[2]*(int)row1[1]+d_u1[3]*(int)row1[2]+d_u1[4]*(int)row1[3]);	
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_B[index*ww +1]=(unsigned char)sum;
		e_aft+=sum;
		// i==ww-2
		temp[0]=(int)(d_u0[0]*(int)row0[ww-4]+d_u0[1]*(int)row0[ww-3]+d_u0[2]*(int)row0[ww-2]+d_u0[3]*(int)row0[ww-1]);
		temp[1]=(int)(d_u1[0]*(int)row1[ww-4]+d_u1[1]*(int)row1[ww-3]+d_u1[2]*(int)row1[ww-2]+d_u1[3]*(int)row1[ww-1]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_B[index*ww +ww-2]=(unsigned char)sum;
		e_aft+=sum;
		// i==ww-1
		temp[0]=(int)(d_u0[0]*(int)row0[ww-3]+d_u0[1]*(int)row0[ww-2]+d_u0[2]*(int)row0[ww-1]);
		temp[1]=(int)(d_u1[0]*(int)row1[ww-3]+d_u1[1]*(int)row1[ww-2]+d_u1[2]*(int)row1[ww-1]);
		sum=temp[0]+temp[1];
		if(sum>255) sum=255;
		else if(sum<0) sum=0;
		ans_B[index*ww +ww-1]=(unsigned char)sum;
		e_aft+=sum;
		
		//#pragma unroll
		for(int i=2; i<ww-2; ++i){
			temp[0]=(int)(d_u0[0]*(int)row0[i-2]+d_u0[1]*(int)row0[i-1]+d_u0[2]*(int)row0[i]+d_u0[3]*(int)row0[i+1]+d_u0[4]*(int)row0[i+2]);
			temp[1]=(int)(d_u1[0]*(int)row1[i-2]+d_u1[1]*(int)row1[i-1]+d_u1[2]*(int)row1[i]+d_u1[3]*(int)row1[i+1]+d_u1[4]*(int)row1[i+2]);
			sum=temp[0]+temp[1];
			if(sum>255) sum=255;
			else if(sum<0) sum=0;
			ans_B[index*ww +i]=(unsigned char)sum;
			e_aft+=sum;
		}
		// convolution finish

		B_rate=(float)e_aft/(float)(B_ori*3/2);
		
		index=(round+tid)*ww;
		//#pragma unroll
		for(int i=0; i<ww; ++i){
			temp[0]=(int)ans_R[index +i];
			temp[0]=(int)((float)temp[0]/R_rate);
			if(temp[0]>255) temp[0]=255;
			else if(temp[0]<0) temp[0]=0;
			temp_R[i*h +round+tid]=ans_R[index +i]=(unsigned char)temp[0];

			temp[0]=(int)ans_G[index +i];
			temp[0]=(int)((float)temp[0]/G_rate);
			if(temp[0]>255) temp[0]=255;
			else if(temp[0]<0) temp[0]=0;
			temp_G[i*h +round+tid]=ans_G[index +i]=(unsigned char)temp[0];

			temp[0]=(int)ans_B[index +i];
			temp[0]=(int)((float)temp[0]/B_rate);
			if(temp[0]>255) temp[0]=255;
			else if(temp[0]<0) temp[0]=0;
			temp_B[i*h +round+tid]=ans_B[index +i]=(unsigned char)temp[0];

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

void SR_kernel_up(
	unsigned char *ori_R, unsigned char *ori_G, unsigned char *ori_B,
	unsigned char *aft_R, unsigned char *aft_G, unsigned char *aft_B,
	int w, int h, int ww, int hh){
	float u0[5]={-0.047, 0.6, 0.927, 0.119, -0.1};
	float u1[5]={-0.1, 0.119, 0.927, 0.6, -0.047};
	
	unsigned char *d_ori_R, *d_ori_G, *d_ori_B;
	unsigned char *d_ans_R, *d_ans_G, *d_ans_B;
	unsigned char *temp_R, *temp_G, *temp_B;

	//printf("in up, w=%d, h=%d, ww=%d, hh=%d\n", w, h, ww, hh);
	cudaMalloc((void**)&d_ori_R, w*h*sizeof(unsigned char));
	cudaMalloc((void**)&d_ori_G, w*h*sizeof(unsigned char));
	cudaMalloc((void**)&d_ori_B, w*h*sizeof(unsigned char));
	cudaMalloc((void**)&temp_R, ww*h*sizeof(unsigned char));
	cudaMalloc((void**)&temp_G, ww*h*sizeof(unsigned char));
	cudaMalloc((void**)&temp_B, ww*h*sizeof(unsigned char));
	cudaMalloc((void**)&d_ans_R, ww*hh*sizeof(unsigned char));
	cudaMalloc((void**)&d_ans_G, ww*hh*sizeof(unsigned char));
	cudaMalloc((void**)&d_ans_B, ww*hh*sizeof(unsigned char));
	
	cudaMemcpy(d_ori_R, ori_R, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ori_G, ori_G, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ori_B, ori_B, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);

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
	
	cudaMemcpy(aft_R, d_ans_R, ww*hh*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(aft_G, d_ans_G, ww*hh*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(aft_B, d_ans_B, ww*hh*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	
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
