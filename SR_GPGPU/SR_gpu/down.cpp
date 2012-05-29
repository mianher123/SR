#include <stdlib.h>

void setup_col(int *row0, int *row1, int *arr, float *d0, float *d1, int j, int w, int h){
	for(int i=0; i<h; ++i){
		if(i==0){
			row0[0]=(int)(d0[1]*arr[j]+d0[2]*arr[w +j]);
			row1[0]=(int)(d1[1]*arr[j]+d1[2]*arr[w +j]);
		}
		else if(i==h-1){
			row0[h-1]=(int)(d0[0]*arr[(i-1)*w +j]+d0[1]*arr[i*w +j]);
			row1[h-1]=(int)(d1[0]*arr[(i-1)*w +j]+d1[1]*arr[i*w +j]);
		}
		else{
			row0[i]=(int)(d0[0]*arr[(i-1)*w +j]+d0[1]*arr[i*w +j]+d0[2]*arr[(i+1)*w +j]);
			row1[i]=(int)(d1[0]*arr[(i-1)*w +j]+d1[1]*arr[i*w +j]+d1[2]*arr[(i+1)*w +j]);
		}
	}
}

void setup_row(int *row0, int *row1, int *arr, float *d0, float *d1, int j, int w, int h){
	for(int i=0; i<w; ++i){
		if(i==0){
			row0[0]=(int)(d0[1]*arr[j*w +0]+d0[2]*arr[j*w +1]);
			row1[0]=(int)(d1[1]*arr[j*w +0]+d1[2]*arr[j*w +1]);
		}
		else if(i==w-1){
			row0[w-1]=(int)(d0[0]*arr[j*w +i-1]+d0[1]*arr[j*w +i]);
			row1[w-1]=(int)(d1[0]*arr[j*w +i-1]+d1[1]*arr[j*w +i]);
		}
		else{
			row0[i]=(int)(d0[0]*arr[j*w +i-1]+d0[1]*arr[j*w +i]+d0[2]*arr[j*w +i+1]);
			row1[i]=(int)(d1[0]*arr[j*w +i-1]+d1[1]*arr[j*w +i]+d1[2]*arr[j*w +i+1]);
		}
	}
}

void down(
	int *ori_R, int *ori_G, int *ori_B,
	int *ans_R, int *ans_G, int *ans_B,
	int w, int h){

	float d0[3]={-0.022, 0.974, 0.227};
	float d1[3]={0.227, 0.974, -0.022};
	int R_ori, G_ori, B_ori;
	int e_aft;
	float R_rate, G_rate, B_rate;
	int *row0;
	int *row1;
	row0=(int*)malloc(sizeof(int)*w);
	row1=(int*)malloc(sizeof(int)*w);
	int ww=w*2/3;
	int hh=h*2/3;

	// row
	for(int j=0; j<h; ++j){
		R_ori=G_ori=B_ori=0;
		for(int i=0; i<w; ++i){
			R_ori+=ori_R[j*w +i];
			G_ori+=ori_G[j*w +i];
			B_ori+=ori_B[j*w +i];
		}
		// red
		setup_row(row0, row1, ori_R, d0, d1, j, w, h);
		e_aft=0;
		for(int i=0; i<ww; ++i){
			if(i%2==0) ans_R[j*ww +i]=row0[3*i/2];
			else ans_R[j*ww +i]=row1[3*(i-1)/2 +2];
			e_aft+=ans_R[j*ww +i];
		}
		R_rate=(float)e_aft/(float)(R_ori*2.0/3.0);
		for(int i=0; i<ww; ++i){
			ans_R[j*ww +i]=(int)((float)ans_R[j*ww +i]/R_rate);
		}
		// green
		setup_row(row0, row1, ori_G, d0, d1, j, w, h);
		e_aft=0;
		for(int i=0; i<ww; ++i){
			if(i%2==0) ans_G[j*ww +i]=row0[3*i/2];
			else ans_G[j*ww +i]=row1[3*(i-1)/2 +2];
			e_aft+=ans_G[j*ww +i];
		}
		G_rate=(float)e_aft/(float)(G_ori*2.0/3.0);
		for(int i=0; i<ww; ++i){
			ans_G[j*ww +i]=(int)((float)ans_G[j*ww +i]/G_rate);
		}
		// blue
		setup_row(row0, row1, ori_B, d0, d1, j, w, h);
		e_aft=0;
		for(int i=0; i<ww; ++i){
			if(i%2==0) ans_B[j*ww +i]=row0[3*i/2];
			else ans_B[j*ww +i]=row1[3*(i-1)/2 +2];
			e_aft+=ans_B[j*ww +i];
		}
		B_rate=(float)e_aft/(float)(B_ori*2.0/3.0);
		for(int i=0; i<ww; ++i){
			ans_B[j*ww +i]=(int)((float)ans_B[j*ww +i]/B_rate);
		}
	} // row finish
	
	// column
	for(int i=0; i<ww; ++i){
		R_ori=G_ori=B_ori=0;
		for(int j=0; j<h; ++j){
			R_ori+=ans_R[j*ww +i];
			G_ori+=ans_G[j*ww +i];
			B_ori+=ans_B[j*ww +i];
		}
		// red
		setup_col(row0, row1, ans_R, d0, d1, i, ww, h);
		e_aft=0;
		for(int j=0; j<hh; ++j){
			if(j%2==0) ans_R[j*ww +i]=row0[3*j/2];
			else ans_R[j*ww +i]=row1[3*(j-1)/2 +2];
			e_aft+=ans_R[j*ww +i];
		}
		R_rate=(float)e_aft/((float)R_ori*2.0/3.0);
		for(int j=0; j<hh; ++j){
			ans_R[j*ww +i]=(int)((float)ans_R[j*ww +i]/R_rate);
		}
		// green
		setup_col(row0, row1, ans_G, d0, d1, i, ww, h);
		e_aft=0;
		for(int j=0; j<hh; ++j){
			if(j%2==0) ans_G[j*ww +i]=row0[3*j/2];
			else ans_G[j*ww +i]=row1[3*(j-1)/2 +2];
			e_aft+=ans_G[j*ww +i];
		}
		G_rate=(float)e_aft/((float)G_ori*2.0/3.0);
		for(int j=0; j<hh; ++j){
			ans_G[j*ww +i]=(int)((float)ans_G[j*ww +i]/G_rate);
		}
		// blue
		setup_col(row0, row1, ans_B, d0, d1, i, ww, h);
		e_aft=0;
		for(int j=0; j<hh; ++j){
			if(j%2==0) ans_B[j*ww +i]=row0[3*j/2];
			else ans_B[j*ww +i]=row1[3*(j-1)/2 +2];
			e_aft+=ans_B[j*ww +i];
		}
		B_rate=(float)e_aft/((float)B_ori*2.0/3.0);
		for(int j=0; j<hh; ++j){
			ans_B[j*ww +i]=(int)((float)ans_B[j*ww +i]/B_rate);
		}
	}
}
