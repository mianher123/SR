
texture<unsigned char, 2, cudaReadModeElementType> TIR; // ww*hh
texture<unsigned char, 2, cudaReadModeElementType> TIG;
texture<unsigned char, 2, cudaReadModeElementType> TIB;

texture<unsigned char, 2, cudaReadModeElementType> TLR; // w*h
texture<unsigned char, 2, cudaReadModeElementType> TLG;
texture<unsigned char, 2, cudaReadModeElementType> TLB;

texture<unsigned char, 2, cudaReadModeElementType> THR; // w*h
texture<unsigned char, 2, cudaReadModeElementType> THG;
texture<unsigned char, 2, cudaReadModeElementType> THB;

#include <stdio.h>
#include <time.h>

__device__ int calc_dist(
		int x, int y, int low_x, int low_y, int w, int ww, int min, int *pos,
		texture<unsigned char, 2, cudaReadModeElementType> TI, texture<unsigned char, 2, cudaReadModeElementType> TL){
	int dist=0;
	int dtex;

	for(int t=0; t<3; ++t){
		for(int k=0; k<3; ++k){
			dtex=(int)tex2D(TI, x+k, y+t)-(int)tex2D(TL, low_x+k, low_y+t);
			dist+=dtex*dtex;
		}
	}
	
	if(dist<min){
		pos[0]=low_x;
		pos[1]=low_y;
		return dist;
	}
	else return min;
}

__global__ void find_neighbor(
	int round,
	int w, int h, int ww, int hh, uchar4 *final_ans_ptr){
	char* final_ans = (char*)final_ans_ptr;
	//int tid=blockDim.x * blockIdx.x + threadIdx.x;
	//int tidy=blockDim.y * blockIdx.y + threadIdx.y;
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	if( round+bid<(hh/3) && tid<ww ){
		__shared__ unsigned char map_I[1024*3];
		__shared__ unsigned char map_L[1024*5];
		
		int y=(round+bid)*3;
		map_I[tid]=tex2D(TIR, tid, y);
		map_I[tid +ww]=tex2D(TIR, tid, y+1);
		map_I[tid +2*ww]=tex2D(TIR, tid, y+2);

		if(tid<w){
			if(y==0){
				map_L[tid +w]=tex2D(TLR, tid, 0);
				map_L[tid +2*w]=tex2D(TLR, tid, 1);
				map_L[tid +3*w]=tex2D(TLR, tid, 2);
				map_L[tid +4*w]=tex2D(TLR, tid, 3);
			}
			else{
				map_L[tid]=tex2D(TLR, tid, y*2/3-1);
				map_L[tid +w]=tex2D(TLR, tid, y*2/3);
				map_L[tid +2*w]=tex2D(TLR, tid, y*2/3+1);
				map_L[tid +3*w]=tex2D(TLR, tid, y*2/3+2);
				map_L[tid +4*w]=tex2D(TLR, tid, y*2/3+3);
			}
		}
		__syncthreads();

		if(tid<ww/3){
			int x=(tid)*3;	
			int low_x=x*2/3;
			int low_y=y*2/3;
			int min_R=585526; // 255*255*9=585525
			//int min_G=585526;
			//int min_B=585526;
			int min_pos_R[2];
			//int min_pos_G[2];
			//int min_pos_B[2];
			int dtex;
			int dist;
		
			for(int j=0; j<=2; ++j){ // find neighbor in 3*3 block
				for(int i=0; i<=2; ++i){
					if( low_x+i-1>=0 && low_x+i-1<=w-3 && low_y+j-1>=0 && low_y+j-1<=h-3 ){
						dist=0;
						for(int t=0; t<3; ++t){
							for(int k=0; k<3; ++k){
								dtex=(int)map_I[t*ww +k]-(int)map_L[(j+t)*w +i+k];
								dist+=dtex*dtex;
							}
						}

						if(dist<min_R){
							min_pos_R[0]=low_x+i-1;
							min_pos_R[1]=low_y+j-1;
							min_R=dist;
						}

						//min_R=calc_dist(x, y, low_x+i, low_y+j, w, ww, min_R, min_pos_R, TIR, TLR);
						//min_G=calc_dist(x, y, low_x+i, low_y+j, w, ww, min_G, min_pos_G, TIG, TLG);
						//min_B=calc_dist(x, y, low_x+i, low_y+j, w, ww, min_B, min_pos_B, TIB, TLB);
					}
				}
			}
		
			int mmm, nnn;
			for(int j=0; j<3; ++j){
				for(int i=0; i<3; ++i){
					mmm=(int)tex2D(THR, min_pos_R[0]+i, min_pos_R[1]+j);
					nnn=(int)tex2D(TIR, x+i, y+j);
					mmm+=nnn;
					if(mmm>255) mmm=255;
					else if(mmm<0) mmm=0;
					final_ans[((y+j)*ww +x+i)*4]=(unsigned char)mmm;
						
					mmm=(int)tex2D(THG, min_pos_R[0]+i, min_pos_R[1]+j);
					nnn=(int)tex2D(TIG, x+i, y+j);
					mmm+=nnn;
					if(mmm>255) mmm=255;
					else if(mmm<0) mmm=0;
					final_ans[((y+j)*ww +x+i)*4 +1]=(unsigned char)mmm;

					mmm=(int)tex2D(THB, min_pos_R[0]+i, min_pos_R[1]+j);
					nnn=(int)tex2D(TIB, x+i, y+j);
					mmm+=nnn;
					if(mmm>255) mmm=255;
					else if(mmm<0) mmm=0;
					final_ans[((y+j)*ww +x+i)*4 +2]=(unsigned char)mmm;

					final_ans[((y+j)*ww +x+i)*4 +3]=(unsigned char)255;
				}
			}
		}
	}
	
	//__syncthreads();
}

void SR_kernel_find_neighbor(
	unsigned char *I_R, unsigned char *I_G, unsigned char *I_B,
	unsigned char *L_R, unsigned char *L_G, unsigned char *L_B,
	unsigned char *H_R, unsigned char *H_G, unsigned char *H_B,
	int w, int h, int ww, int hh, uchar4* tex){
	
	//int *d_IR, *d_IG, *d_IB; // img(up)
	//int *d_LR, *d_LG, *d_LB; // img(up(down))
	//int *d_HR, *d_HG, *d_HB; // img(original) - img(up(down))
	//unsigned char *d_ansR, *d_ansG, *d_ansB;
	//cudaMalloc((void**)&d_ansR, ww*hh*sizeof(unsigned char));
	//cudaMalloc((void**)&d_ansG, ww*hh*sizeof(unsigned char));
	//cudaMalloc((void**)&d_ansB, ww*hh*sizeof(unsigned char));
	//cudaMalloc((void**)&final_ans, ww*hh*4*sizeof(char));
	cudaChannelFormatDesc Desc=cudaCreateChannelDesc<unsigned char>();
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
	
	cudaMemcpyToArray(d_IR, 0, 0, I_R, ww*hh*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_IG, 0, 0, I_G, ww*hh*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_IB, 0, 0, I_B, ww*hh*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_LR, 0, 0, L_R, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_LG, 0, 0, L_G, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_LB, 0, 0, L_B, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_HR, 0, 0, H_R, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_HG, 0, 0, H_G, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_HB, 0, 0, H_B, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);


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
	int blocks=256;
	//for(int i=0; i<((ww/3)*(hh/3)-1)/(threads*blocks) +1; ++i){
	for(int i=0; i<((hh/3)-1)/(blocks) +1; ++i){
		find_neighbor<<<blocks, threads>>>(i*blocks, w, h, ww, hh, tex);
		
		//printf("error1: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		//printf("error2: %s\n", cudaGetErrorString(cudaThreadSynchronize()));
	}

	

	//cudaMemcpy(ans_R, d_ansR, ww*hh*sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(ans_G, d_ansG, ww*hh*sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(ans_B, d_ansB, ww*hh*sizeof(int), cudaMemcpyDeviceToHost);

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
}
