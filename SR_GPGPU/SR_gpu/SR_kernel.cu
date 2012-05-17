/*
#pragma comment(lib, "cv.lib")
#pragma comment(lib, "highgui.lib")
#include <cv.h>
#include <highgui.h>
*/

texture<int, 1, cudaReadModeElementType> TR;
texture<int, 1, cudaReadModeElementType> TG;
texture<int, 1, cudaReadModeElementType> TB;
texture<float, 1, cudaReadModeElementType> Tu0;
texture<float, 1, cudaReadModeElementType> Tu1;

__device__ int convolusion_col(int index, int ww, int hh, int *ans, int *row0, int *row1){
	int e_aft=0;
	int temp[2];
	for(int i=0; i<hh; ++i){
		if(i==0){
			temp[0]=(int)(tex1Dfetch(Tu0, 2)*row0[0]+tex1Dfetch(Tu0, 3)*row0[1]+tex1Dfetch(Tu0, 4)*row0[2]);
			temp[1]=(int)(tex1Dfetch(Tu1, 2)*row1[0]+tex1Dfetch(Tu1, 3)*row1[1]+tex1Dfetch(Tu1, 4)*row1[2]);
		}
		else if(i==1){
			temp[0]=(int)(tex1Dfetch(Tu0, 1)*row0[0]+tex1Dfetch(Tu0, 2)*row0[1]+tex1Dfetch(Tu0, 3)*row0[2]+tex1Dfetch(Tu0, 4)*row0[3]);
			temp[1]=(int)(tex1Dfetch(Tu1, 1)*row1[0]+tex1Dfetch(Tu1, 2)*row1[1]+tex1Dfetch(Tu1, 3)*row1[2]+tex1Dfetch(Tu1, 4)*row1[3]);
		}
		else if(i==hh-2){
			temp[0]=(int)(tex1Dfetch(Tu0, 0)*row0[hh-4]+tex1Dfetch(Tu0, 1)*row0[hh-3]+tex1Dfetch(Tu0, 2)*row0[hh-2]+tex1Dfetch(Tu0, 3)*row0[hh-1]);
			temp[1]=(int)(tex1Dfetch(Tu1, 0)*row1[hh-4]+tex1Dfetch(Tu1, 1)*row1[hh-3]+tex1Dfetch(Tu1, 2)*row1[hh-2]+tex1Dfetch(Tu1, 3)*row1[hh-1]);
		}
		else if(i==hh-1){
			temp[0]=(int)(tex1Dfetch(Tu0, 0)*row0[hh-3]+tex1Dfetch(Tu0, 1)*row0[hh-2]+tex1Dfetch(Tu0, 2)*row0[hh-1]);
			temp[1]=(int)(tex1Dfetch(Tu1, 0)*row1[hh-3]+tex1Dfetch(Tu1, 1)*row1[hh-2]+tex1Dfetch(Tu1, 2)*row1[hh-1]);
		}
		else{
			temp[0]=(int)(tex1Dfetch(Tu0, 0)*row0[i-2]+tex1Dfetch(Tu0, 1)*row0[i-1]+tex1Dfetch(Tu0, 2)*row0[i]+tex1Dfetch(Tu0, 3)*row0[i+1]+tex1Dfetch(Tu0, 4)*row0[i+2]);
			temp[1]=(int)(tex1Dfetch(Tu1, 0)*row1[i-2]+tex1Dfetch(Tu1, 1)*row1[i-1]+tex1Dfetch(Tu1, 2)*row1[i]+tex1Dfetch(Tu1, 3)*row1[i+1]+tex1Dfetch(Tu1, 4)*row1[i+2]);
		}
		ans[i*ww +index]=temp[0]+temp[1];
		e_aft+=(temp[0]+temp[1]);
	}
	return e_aft;
}

__device__ int convolusion_row(int index, int w, int ww, int *ans, int *row0, int *row1){
	int e_aft=0;
	int temp[2];
	for(int i=0; i<ww; ++i){
		if(i==0){
			temp[0]=(int)(tex1Dfetch(Tu0, 2)*row0[0]+tex1Dfetch(Tu0, 3)*row0[1]+tex1Dfetch(Tu0, 4)*row0[2]);
			temp[1]=(int)(tex1Dfetch(Tu1, 2)*row1[0]+tex1Dfetch(Tu1, 3)*row1[1]+tex1Dfetch(Tu1, 4)*row1[2]);
		}
		else if(i==1){
			temp[0]=(int)(tex1Dfetch(Tu0, 1)*row0[0]+tex1Dfetch(Tu0, 2)*row0[1]+tex1Dfetch(Tu0, 3)*row0[2]+tex1Dfetch(Tu0, 4)*row0[3]);
			temp[1]=(int)(tex1Dfetch(Tu1, 1)*row1[0]+tex1Dfetch(Tu1, 2)*row1[1]+tex1Dfetch(Tu1, 3)*row1[2]+tex1Dfetch(Tu1, 4)*row1[3]);
		}
		else if(i==ww-2){
			temp[0]=(int)(tex1Dfetch(Tu0, 0)*row0[ww-4]+tex1Dfetch(Tu0, 1)*row0[ww-3]+tex1Dfetch(Tu0, 2)*row0[ww-2]+tex1Dfetch(Tu0, 3)*row0[ww-1]);
			temp[1]=(int)(tex1Dfetch(Tu1, 0)*row1[ww-4]+tex1Dfetch(Tu1, 1)*row1[ww-3]+tex1Dfetch(Tu1, 2)*row1[ww-2]+tex1Dfetch(Tu1, 3)*row1[ww-1]);
		}
		else if(i==ww-1){
			temp[0]=(int)(tex1Dfetch(Tu0, 0)*row0[ww-3]+tex1Dfetch(Tu0, 1)*row0[ww-2]+tex1Dfetch(Tu0, 2)*row0[ww-1]);
			temp[1]=(int)(tex1Dfetch(Tu1, 0)*row1[ww-3]+tex1Dfetch(Tu1, 1)*row1[ww-2]+tex1Dfetch(Tu1, 2)*row1[ww-1]);
		}
		else{
			temp[0]=(int)(tex1Dfetch(Tu0, 0)*row0[i-2]+tex1Dfetch(Tu0, 1)*row0[i-1]+tex1Dfetch(Tu0, 2)*row0[i]+tex1Dfetch(Tu0, 3)*row0[i+1]+tex1Dfetch(Tu0, 4)*row0[i+2]);
			temp[1]=(int)(tex1Dfetch(Tu1, 0)*row1[i-2]+tex1Dfetch(Tu1, 1)*row1[i-1]+tex1Dfetch(Tu1, 2)*row1[i]+tex1Dfetch(Tu1, 3)*row1[i+1]+tex1Dfetch(Tu1, 4)*row1[i+2]);
		}
		ans[index*ww +i]=temp[0]+temp[1];
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
		for(int i=0; i<h; ++i){ // compute weight
			R_ori+=ans_R[i*ww +round+tid];
			G_ori+=ans_G[i*ww +round+tid];
			B_ori+=ans_B[i*ww +round+tid];
		}
		int row0[405];
		int row1[405];
		// red
		for(int i=0; i<405; ++i){
			if(i%3==0) row0[i]=ans_R[(i*2/3)*ww +round+tid];
			else row0[i]=0;

			if(i%3==2) row1[i]=ans_R[((i-2)*2/3+1)*ww +round+tid];
			else row1[i]=0;
		}
		e_aft=convolusion_col(round+tid, ww, hh, ans_R, row0, row1);
		R_rate=(float)e_aft/(float)(R_ori*3/2);
		// green
		for(int i=0; i<405; ++i){
			if(i%3==0) row0[i]=ans_G[(i*2/3)*ww +round+tid];
			else row0[i]=0;

			if(i%3==2) row1[i]=ans_G[((i-2)*2/3+1)*ww +round+tid];
			else row1[i]=0;
		}
		e_aft=convolusion_col(round+tid, ww, hh, ans_G, row0, row1);
		G_rate=(float)e_aft/(float)(G_ori*3/2);
		// blue
		for(int i=0; i<405; ++i){
			if(i%3==0) row0[i]=ans_B[(i*2/3)*ww +round+tid];
			else row0[i]=0;

			if(i%3==2) row1[i]=ans_B[((i-2)*2/3+1)*ww +round+tid];
			else row1[i]=0;
		}
		e_aft=convolusion_col(round+tid, ww, hh, ans_B, row0, row1);
		B_rate=(float)e_aft/(float)(B_ori*3/2);
		
		for(int i=0; i<405; ++i){
			ans_R[i*ww +round+tid]=(int)((float)ans_R[i*ww +round+tid]/R_rate);
			ans_G[i*ww +round+tid]=(int)((float)ans_G[i*ww +round+tid]/G_rate);
			ans_B[i*ww +round+tid]=(int)((float)ans_B[i*ww +round+tid]/B_rate);

			if(ans_R[i*ww +round+tid]>255) ans_R[i*ww +round+tid]=255;
			else if(ans_R[i*ww +round+tid]<0) ans_R[i*ww +round+tid]=0;

			if(ans_G[i*ww +round+tid]>255) ans_G[i*ww +round+tid]=255;
			else if(ans_G[i*ww +round+tid]<0) ans_G[i*ww +round+tid]=0;

			if(ans_B[i*ww +round+tid]>255) ans_B[i*ww +round+tid]=255;
			else if(ans_B[i*ww +round+tid]<0) ans_B[i*ww +round+tid]=0;
		}
		
	}
}

__global__ void run_cuda_row(int round, int *ans_R, int *ans_G, int *ans_B, int w, int h, int ww, int hh){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(round+tid<h){
		int R_ori=0, G_ori=0, B_ori=0; // store weight of original img
		int e_aft;
		float R_rate, G_rate, B_rate;
		for(int i=0; i<w; ++i){ // compute weight
			/*
			ans_R[(round+tid)*w +i]=tex1Dfetch(TR, (round+tid)*w +i);
			ans_G[(round+tid)*w +i]=tex1Dfetch(TG, (round+tid)*w +i);
			ans_B[(round+tid)*w +i]=tex1Dfetch(TB, (round+tid)*w +i);
			*/
			
			R_ori+=tex1Dfetch(TR, (round+tid)*w +i);
			G_ori+=tex1Dfetch(TG, (round+tid)*w +i);
			B_ori+=tex1Dfetch(TB, (round+tid)*w +i);
			
		}
		int row0[540];
		int row1[540];
		// red
		for(int i=0; i<ww; ++i){ // setup row
			if(i%3==0) row0[i]=tex1Dfetch(TR, (round+tid)*w +i*2/3);
			else row0[i]=0;

			if(i%3==2) row1[i]=tex1Dfetch(TR, (round+tid)*w +(i-2)*2/3+1);
			else row1[i]=0;
		}
		e_aft=convolusion_row(round+tid, w, ww, ans_R, row0, row1);
		R_rate=(float)e_aft/(float)(R_ori*3/2);

		// green
		for(int i=0; i<ww; ++i){ // setup row
			if(i%3==0) row0[i]=tex1Dfetch(TG, (round+tid)*w +i*2/3);
			else row0[i]=0;

			if(i%3==2) row1[i]=tex1Dfetch(TG, (round+tid)*w +(i-2)*2/3+1);
			else row1[i]=0;
		}
		e_aft=convolusion_row(round+tid, w, ww, ans_G, row0, row1);
		G_rate=(float)e_aft/(float)(G_ori*3/2);

		// blue
		for(int i=0; i<ww; ++i){ // setup row
			if(i%3==0) row0[i]=tex1Dfetch(TB, (round+tid)*w +i*2/3);
			else row0[i]=0;

			if(i%3==2) row1[i]=tex1Dfetch(TB, (round+tid)*w +(i-2)*2/3+1);
			else row1[i]=0;
		}
		e_aft=convolusion_row(round+tid, w, ww, ans_B, row0, row1);
		B_rate=(float)e_aft/(float)(B_ori*3/2);
		
		for(int i=0; i<ww; ++i){
			ans_R[(round+tid)*ww +i]=(int)((float)ans_R[(round+tid)*ww +i]/R_rate);
			ans_G[(round+tid)*ww +i]=(int)((float)ans_G[(round+tid)*ww +i]/G_rate);
			ans_B[(round+tid)*ww +i]=(int)((float)ans_B[(round+tid)*ww +i]/B_rate);
			
			if(ans_R[(round+tid)*ww +i]>255) ans_R[(round+tid)*ww +i]=255;
			else if(ans_R[(round+tid)*ww +i]<0) ans_R[(round+tid)*ww +i]=0;

			if(ans_G[(round+tid)*ww +i]>255) ans_G[(round+tid)*ww +i]=255;
			else if(ans_G[(round+tid)*ww +i]<0) ans_G[(round+tid)*ww +i]=0;

			if(ans_B[(round+tid)*ww +i]>255) ans_B[(round+tid)*ww +i]=255;
			else if(ans_B[(round+tid)*ww +i]<0) ans_B[(round+tid)*ww +i]=0;
		}
	}
}

void SR_kernel(int *ori_R, int *ori_G, int *ori_B, int *aft_R, int *aft_G, int *aft_B, int w, int h){
	float u1[5]={-0.1, 0.119, 0.927, 0.6, -0.047};
	float u0[5]={-0.047, 0.6, 0.927, 0.119, -0.1};
	
	int *R;
	int *G;
	int *B;
	int *ans_R;
	int *ans_G;
	int *ans_B;
	int *d_u0;
	int *d_u1;

	int ww=w*3/2;
	int hh=h*3/2;

	cudaMalloc((void**)&R, w*h*sizeof(int));
	cudaMalloc((void**)&G, w*h*sizeof(int));
	cudaMalloc((void**)&B, w*h*sizeof(int));
	cudaMalloc((void**)&ans_R, w*h*sizeof(int)*9/4);
	cudaMalloc((void**)&ans_G, w*h*sizeof(int)*9/4);
	cudaMalloc((void**)&ans_B, w*h*sizeof(int)*9/4);
	cudaMalloc((void**)&d_u0, 5*sizeof(float));
	cudaMalloc((void**)&d_u1, 5*sizeof(float));
	
	cudaMemcpy(R, ori_R, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(G, ori_G, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B, ori_B, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u0, u0, 5*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u1, u1, 5*sizeof(float), cudaMemcpyHostToDevice);

	cudaBindTexture(0, TR, R);
	cudaBindTexture(0, TG, G);
	cudaBindTexture(0, TB, B);
	cudaBindTexture(0, Tu0, d_u0);
	cudaBindTexture(0, Tu1, d_u1);

	int threads=64;
	int blocks=64;
	
	for(int i=0; i<(h-1)/(threads*blocks) +1; ++i) // a thread do a row
		run_cuda_row<<<threads, blocks>>>(i*threads*blocks, ans_R, ans_G, ans_B, w, h, ww, hh);
	
	for(int i=0; i<(ww-1)/(threads*blocks) +1; ++i)
		run_cuda_col<<<threads, blocks>>>(i*threads*blocks, ans_R, ans_G, ans_B, w, h, ww, hh);
		
	cudaMemcpy(aft_R, ans_R, w*h*sizeof(int)*9/4, cudaMemcpyDeviceToHost);
	cudaMemcpy(aft_G, ans_G, w*h*sizeof(int)*9/4, cudaMemcpyDeviceToHost);
	cudaMemcpy(aft_B, ans_B, w*h*sizeof(int)*9/4, cudaMemcpyDeviceToHost);

	cudaUnbindTexture(TR);
	cudaUnbindTexture(TG);
	cudaUnbindTexture(TB);
	cudaFree(R);
	cudaFree(G);
	cudaFree(B);
}