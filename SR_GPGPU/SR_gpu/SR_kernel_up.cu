//#include "SR_kernel_start.cu"
#include <stdio.h>
#include <time.h>

texture<unsigned char, 1, cudaReadModeElementType> TR;
texture<unsigned char, 1, cudaReadModeElementType> TG;
texture<unsigned char, 1, cudaReadModeElementType> TB;
texture<unsigned char ,1, cudaReadModeElementType> TansR;
texture<unsigned char ,1, cudaReadModeElementType> TansG;
texture<unsigned char ,1, cudaReadModeElementType> TansB;

//__shared__ unsigned char share_mem[1024];

__constant__ float d_u0[5];
__constant__ float d_u1[5];

extern "C" void set_filter_up(float *u0, float *u1){
	cudaMemcpyToSymbol(d_u0, u0, 5*sizeof(float));
	cudaMemcpyToSymbol(d_u1, u1, 5*sizeof(float));
}

__device__ unsigned char up_clamp(int value){
	if(value > 255) return (unsigned char)255;
	else if(value < 0) return (unsigned char)0;
	else return value;
}

__global__ void run_cuda_col(int round, unsigned char *ans_R, unsigned char *ans_G, unsigned char *ans_B, int w, int h, int ww, int hh, uchar4* tex_trivial){
	//int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	if(round+bid<ww && tid<hh){
		//int R_ori=0, G_ori=0, B_ori=0; // store weight of original img
		//int e_aft;
		//float R_rate, G_rate, B_rate;
		int index=(round+bid)*h;
		int a_index=round+bid;
		unsigned char window1[5]={0, 0, 0, 0 ,0};
		unsigned char window2[5]={0, 0, 0, 0, 0};
		unsigned char* tex_trivial_p = (unsigned char*)tex_trivial;
		//unsigned char window1[6]={0, 0, 0, 0 ,0, 0};
		//unsigned char window2[6]={0, 0, 0, 0 ,0, 0};
		__shared__ unsigned char share_mem[1024];


		int temp[2];
		unsigned char sum;
		int bi;
		//int mod;

		if(tid<h)
			share_mem[tid]=tex1Dfetch(TansR, index +tid);
		__syncthreads();

		// red
		if(tid==0){
			window1[2]=share_mem[0];
			window2[4]=share_mem[1];
		}
		else if(tid==1){
			window1[1]=share_mem[0];
			window1[4]=share_mem[2];
			window2[3]=share_mem[1];
		}
		else if(tid%3==0){
			bi=tid*2/3;
			window1[2]=share_mem[bi];
			window2[1]=share_mem[bi-1];
			window2[4]=share_mem[bi+1];
		}
		else if(tid%3==1){
			bi=(tid-1)*2/3;
			window1[1]=share_mem[bi];
			window1[4]=share_mem[bi+2];
			window2[0]=share_mem[bi-1];
			window2[3]=share_mem[bi+1];
		}
		else{
			bi=(tid-2)*2/3;
			window1[0]=share_mem[bi];
			window1[3]=share_mem[bi+2];
			window2[2]=share_mem[bi+1];
		}

		temp[0]=(int)(d_u0[0]*(int)window1[0]+d_u0[1]*(int)window1[1]+d_u0[2]*(int)window1[2]+d_u0[3]*(int)window1[3]+d_u0[4]*(int)window1[4]);
		temp[1]=(int)(d_u1[0]*(int)window2[0]+d_u1[1]*(int)window2[1]+d_u1[2]*(int)window2[2]+d_u1[3]*(int)window2[3]+d_u1[4]*(int)window2[4]);
		sum=up_clamp(temp[0]+temp[1]);
		ans_R[tid*ww +a_index]=(unsigned char)sum;
		tex_trivial_p[(tid*ww + a_index)*4] = sum;

		//e_aft+=(int)sum;
		//R_rate=(float)e_aft/(float)(R_ori*3/2);

		// green
		window1[0]=window1[1]=window1[2]=window1[3]=window1[4]=0;
		window2[0]=window2[1]=window2[2]=window2[3]=window2[4]=0;
		
		if(tid<h)
			share_mem[tid]=tex1Dfetch(TansG, index +tid);
		__syncthreads();

		// green
		if(tid==0){
			window1[2]=share_mem[0];
			window2[4]=share_mem[1];
		}
		else if(tid==1){
			window1[1]=share_mem[0];
			window1[4]=share_mem[2];
			window2[3]=share_mem[1];
		}
		else if(tid%3==0){
			bi=tid*2/3;
			window1[2]=share_mem[bi];
			window2[1]=share_mem[bi-1];
			window2[4]=share_mem[bi+1];
		}
		else if(tid%3==1){
			bi=(tid-1)*2/3;
			window1[1]=share_mem[bi];
			window1[4]=share_mem[bi+2];
			window2[0]=share_mem[bi-1];
			window2[3]=share_mem[bi+1];
		}
		else{
			bi=(tid-2)*2/3;
			window1[0]=share_mem[bi];
			window1[3]=share_mem[bi+2];
			window2[2]=share_mem[bi+1];
		}
		temp[0]=(int)(d_u0[0]*(int)window1[0]+d_u0[1]*(int)window1[1]+d_u0[2]*(int)window1[2]+d_u0[3]*(int)window1[3]+d_u0[4]*(int)window1[4]);
		temp[1]=(int)(d_u1[0]*(int)window2[0]+d_u1[1]*(int)window2[1]+d_u1[2]*(int)window2[2]+d_u1[3]*(int)window2[3]+d_u1[4]*(int)window2[4]);
		sum=up_clamp(temp[0]+temp[1]);
		ans_G[tid*ww +a_index]=(unsigned char)sum;
		tex_trivial_p[(tid*ww + a_index)*4+1] = sum;
		//G_rate=(float)e_aft/(float)(G_ori*3/2);

		// blue
		window1[0]=window1[1]=window1[2]=window1[3]=window1[4]=0;
		window2[0]=window2[1]=window2[2]=window2[3]=window2[4]=0;
		
		if(tid<h)
			share_mem[tid]=tex1Dfetch(TansB, index +tid);
		__syncthreads();

		// blue
		if(tid==0){
			window1[2]=share_mem[0];
			window2[4]=share_mem[1];
		}
		else if(tid==1){
			window1[1]=share_mem[0];
			window1[4]=share_mem[2];
			window2[3]=share_mem[1];
		}
		else if(tid%3==0){
			bi=tid*2/3;
			window1[2]=share_mem[bi];
			window2[1]=share_mem[bi-1];
			window2[4]=share_mem[bi+1];
		}
		else if(tid%3==1){
			bi=(tid-1)*2/3;
			window1[1]=share_mem[bi];
			window1[4]=share_mem[bi+2];
			window2[0]=share_mem[bi-1];
			window2[3]=share_mem[bi+1];
		}
		else{
			bi=(tid-2)*2/3;
			window1[0]=share_mem[bi];
			window1[3]=share_mem[bi+2];
			window2[2]=share_mem[bi+1];
		}
		temp[0]=(int)(d_u0[0]*(int)window1[0]+d_u0[1]*(int)window1[1]+d_u0[2]*(int)window1[2]+d_u0[3]*(int)window1[3]+d_u0[4]*(int)window1[4]);
		temp[1]=(int)(d_u1[0]*(int)window2[0]+d_u1[1]*(int)window2[1]+d_u1[2]*(int)window2[2]+d_u1[3]*(int)window2[3]+d_u1[4]*(int)window2[4]);
		sum=up_clamp(temp[0]+temp[1]);
		ans_B[tid*ww +a_index]=(unsigned char)sum;
		tex_trivial_p[(tid*ww + a_index)*4+2] = sum;
		//B_rate=(float)e_aft/(float)(B_ori*3/2);
		

		tex_trivial_p[(tid*ww + a_index)*4+3] = 255;
		/*
		#pragma unroll
		for(int i=0; i<hh; ++i){
			temp[0]=(int)ans_R[i*ww +a_index];
			temp[0]=(int)((float)temp[0]/R_rate);
			if(temp[0]>255) temp[0]=255;
			else if(temp[0]<0) temp[0]=0;
			ans_R[i*ww +a_index]=(unsigned char)temp[0];

			temp[0]=(int)ans_G[i*ww +a_index];
			temp[0]=(int)((float)temp[0]/G_rate);
			if(temp[0]>255) temp[0]=255;
			else if(temp[0]<0) temp[0]=0;
			ans_G[i*ww +a_index]=(unsigned char)temp[0];

			temp[0]=(int)ans_B[i*ww +a_index];
			temp[0]=(int)((float)temp[0]/B_rate);
			if(temp[0]>255) temp[0]=255;
			else if(temp[0]<0) temp[0]=0;
			ans_B[i*ww +a_index]=(unsigned char)temp[0];
		}*/
	}
}

__global__ void run_cuda_row(
	int round,
	unsigned char *ans_R, unsigned char *ans_G, unsigned char *ans_B,
	int w, int h, int ww, int hh,
	unsigned char *temp_R, unsigned char *temp_G, unsigned char *temp_B){

	//int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	__shared__ unsigned char share_mem[1024];

	if(round+bid<h && tid<ww){
		//__shared__ int R_ori=0, G_ori=0, B_ori=0; // store weight of original img
		//__shared__ int e_aft;
		//__shared__ float R_rate, G_rate, B_rate;
		int index=(round+bid)*w;
		int a_index=(round+bid)*ww;
		int bi;
		int temp[2];

		unsigned char sum;
		unsigned char window1[5]={0, 0, 0, 0 ,0};
		unsigned char window2[5]={0, 0, 0, 0, 0};
		//unsigned char window1[6]={0, 0, 0, 0 ,0, 0};
		//unsigned char window2[6]={0, 0, 0, 0 ,0, 0};

		if(tid<w)
			share_mem[tid]=tex1Dfetch(TR, index +tid);
		__syncthreads();

		// red

		//int ori_index1=2;
		//int ori_index2=1;
		//e_aft=0;
		//R_ori+=window1[3]=tex1Dfetch(TR, index);
		if(tid==0){
			window1[2]=share_mem[0];
			window2[4]=share_mem[1];
		}
		else if(tid==1){
			window1[1]=share_mem[0];
			window1[4]=share_mem[2];
			window2[3]=share_mem[1];
		}
		else if(tid%3==0){
			bi=tid*2/3;
			window1[2]=share_mem[bi];
			window2[1]=share_mem[bi-1];
			window2[4]=share_mem[bi+1];
		}
		else if(tid%3==1){
			bi=(tid-1)*2/3;
			window1[1]=share_mem[bi];
			window1[4]=share_mem[bi+2];
			window2[0]=share_mem[bi-1];
			window2[3]=share_mem[bi+1];
		}
		else{
			bi=(tid-2)*2/3;
			window1[0]=share_mem[bi];
			window1[3]=share_mem[bi+2];
			window2[2]=share_mem[bi+1];
		}
		/*
		if(tid==0){
			window1[2]=share_mem[0];
			window2[4]=share_mem[1];
		}
		else if(tid==1){
			window1[1]=share_mem[0];
			window1[4]=share_mem[2];
			window2[3]=share_mem[1];
		}
		else if(tid==2){
			window1[0]=share_mem[0];
			window1[3]=share_mem[2];
			window2[2]=share_mem[1];
		}
		else{
			mod=tid%3;
			bi=(tid-mod)*2/3;
			window1[2-mod]=share_mem[bi];
			window1[5-mod]=share_mem[bi+2];
			window2[(7-mod)%6]=share_mem[bi-1];
			window2[4-mod]=share_mem[bi+1];
		}*/
		temp[0]=(int)(d_u0[0]*(int)window1[0]+d_u0[1]*(int)window1[1]+d_u0[2]*(int)window1[2]+d_u0[3]*(int)window1[3]+d_u0[4]*(int)window1[4]);
		temp[1]=(int)(d_u1[0]*(int)window2[0]+d_u1[1]*(int)window2[1]+d_u1[2]*(int)window2[2]+d_u1[3]*(int)window2[3]+d_u1[4]*(int)window2[4]);
		sum=up_clamp(temp[0]+temp[1]);
		temp_R[tid*h +round+bid]=ans_R[a_index +tid]=(unsigned char)sum;
		//R_rate=(float)e_aft/(float)(R_ori*3/2);
		
		// green
		window1[0]=window1[1]=window1[2]=window1[3]=window1[4]=0;
		window2[0]=window2[1]=window2[2]=window2[3]=window2[4]=0;
		if(tid<w)
			share_mem[tid]=tex1Dfetch(TG, index +tid);
		__syncthreads();
		//ori_index1=2;
		//ori_index2=1;
		//e_aft=0;
		//G_ori+=window1[3]=tex1Dfetch(TG, index);
		
		if(tid==0){
			window1[2]=share_mem[0];
			window2[4]=share_mem[1];
		}
		else if(tid==1){
			window1[1]=share_mem[0];
			window1[4]=share_mem[2];
			window2[3]=share_mem[1];
		}
		else if(tid%3==0){
			bi=tid*2/3;
			window1[2]=share_mem[bi];
			window2[1]=share_mem[bi-1];
			window2[4]=share_mem[bi+1];
		}
		else if(tid%3==1){
			bi=(tid-1)*2/3;
			window1[1]=share_mem[bi];
			window1[4]=share_mem[bi+2];
			window2[0]=share_mem[bi-1];
			window2[3]=share_mem[bi+1];
		}
		else{
			bi=(tid-2)*2/3;
			window1[0]=share_mem[bi];
			window1[3]=share_mem[bi+2];
			window2[2]=share_mem[bi+1];
		}
		temp[0]=(int)(d_u0[0]*(int)window1[0]+d_u0[1]*(int)window1[1]+d_u0[2]*(int)window1[2]+d_u0[3]*(int)window1[3]+d_u0[4]*(int)window1[4]);
		temp[1]=(int)(d_u1[0]*(int)window2[0]+d_u1[1]*(int)window2[1]+d_u1[2]*(int)window2[2]+d_u1[3]*(int)window2[3]+d_u1[4]*(int)window2[4]);
		sum=up_clamp(temp[0]+temp[1]);
		temp_G[tid*h +round+bid]=ans_G[a_index +tid]=(unsigned char)sum;
		// convolution finish
		//G_rate=(float)e_aft/(float)(G_ori*3/2);

		// blue
		window1[0]=window1[1]=window1[2]=window1[3]=window1[4]=0;
		window2[0]=window2[1]=window2[2]=window2[3]=window2[4]=0;
		if(tid<w)
			share_mem[tid]=tex1Dfetch(TB, index +tid);
		__syncthreads();
		//ori_index1=2;
		//ori_index2=1;
		//e_aft=0;
		//B_ori+=window1[3]=tex1Dfetch(TB, index);
		
		if(tid==0){
			window1[2]=share_mem[0];
			window2[4]=share_mem[1];
		}
		else if(tid==1){
			window1[1]=share_mem[0];
			window1[4]=share_mem[2];
			window2[3]=share_mem[1];
		}
		else if(tid%3==0){
			bi=tid*2/3;
			window1[2]=share_mem[bi];
			window2[1]=share_mem[bi-1];
			window2[4]=share_mem[bi+1];
		}
		else if(tid%3==1){
			bi=(tid-1)*2/3;
			window1[1]=share_mem[bi];
			window1[4]=share_mem[bi+2];
			window2[0]=share_mem[bi-1];
			window2[3]=share_mem[bi+1];
		}
		else{
			bi=(tid-2)*2/3;
			window1[0]=share_mem[bi];
			window1[3]=share_mem[bi+2];
			window2[2]=share_mem[bi+1];
		}
		temp[0]=(int)(d_u0[0]*(int)window1[0]+d_u0[1]*(int)window1[1]+d_u0[2]*(int)window1[2]+d_u0[3]*(int)window1[3]+d_u0[4]*(int)window1[4]);
		temp[1]=(int)(d_u1[0]*(int)window2[0]+d_u1[1]*(int)window2[1]+d_u1[2]*(int)window2[2]+d_u1[3]*(int)window2[3]+d_u1[4]*(int)window2[4]);
		sum=up_clamp(temp[0]+temp[1]);
		temp_B[tid*h +round+bid]=ans_B[a_index +tid]=(unsigned char)sum;
		// convolution finish
		//B_rate=(float)e_aft/(float)(B_ori*3/2);
		
		/*
		#pragma unroll
		for(int i=0; i<ww; ++i){
			temp[0]=(int)ans_R[a_index +i];
			temp[0]=(int)((float)temp[0]/R_rate);
			if(temp[0]>255) temp[0]=255;
			else if(temp[0]<0) temp[0]=0;
			temp_R[i*h +round+tid]=ans_R[a_index +i]=(unsigned char)temp[0];

			temp[0]=(int)ans_G[a_index +i];
			temp[0]=(int)((float)temp[0]/G_rate);
			if(temp[0]>255) temp[0]=255;
			else if(temp[0]<0) temp[0]=0;
			temp_G[i*h +round+tid]=ans_G[a_index +i]=(unsigned char)temp[0];

			temp[0]=(int)ans_B[a_index +i];
			temp[0]=(int)((float)temp[0]/B_rate);
			if(temp[0]>255) temp[0]=255;
			else if(temp[0]<0) temp[0]=0;
			temp_B[i*h +round+tid]=ans_B[a_index +i]=(unsigned char)temp[0];
		}*/
	}

	//__syncthreads();
}

void SR_kernel_up(
	unsigned char *ori_R, unsigned char *ori_G, unsigned char *ori_B,
	unsigned char *aft_R, unsigned char *aft_G, unsigned char *aft_B,
	int w, int h, int ww, int hh, uchar4* tex_trivial){
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

	int threads=750;
	int blocks=256;
	for(int i=0; i<(h-1)/(blocks) +1; ++i) // a thread do a row
		run_cuda_row<<<blocks, threads>>>(i*blocks, d_ans_R, d_ans_G, d_ans_B, w, h, ww, hh, temp_R, temp_G, temp_B);
		//run_cuda_row<<<blocks, threads, threads*sizeof(int)*ww*2>>>(i*threads*blocks, d_ans_R, d_ans_G, d_ans_B, w, h, ww, hh, temp_R, temp_G, temp_B);
	
	cudaBindTexture(0, TansR, temp_R);
	cudaBindTexture(0, TansG, temp_G);
	cudaBindTexture(0, TansB, temp_B);

	for(int i=0; i<(ww-1)/(blocks) +1; ++i) // a thread do a column
		run_cuda_col<<<blocks, threads>>>(i*blocks, d_ans_R, d_ans_G, d_ans_B, w, h, ww, hh, tex_trivial);
	
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
