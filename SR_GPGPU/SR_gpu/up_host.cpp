#include <stdlib.h>
#include <stdio.h>

void up_setup_col(int *row0, int *row1, int w, int hh, int index, int *aft){
	for(int i=0; i<hh; i++){
		if(i%3==0) row0[i]=aft[(i*2/3)*w+index];
		else row0[i]=0;

		if(i%3==2) row1[i]=aft[((i-2)*2/3+1)*w+index];
		else row1[i]=0;
	}
}

void up_setup_row(int *row0, int *row1, int *arr, int w, int ww, int index){
	for(int i=0; i<ww; i++){ // setup row0 and row1
		if(i%3==0) row0[i]=arr[index*w+i*2/3];
		else row0[i]=0;
		
		if(i%3==2) row1[i]=arr[index*w+(i-2)*2/3+1];
		else row1[i]=0;
	}
	
}

int up_convolusion_col(float u0[5], float u1[5], int *row0, int *row1, int ww, int hh, int index, int *aft){
	int e_aft=0;
	int temp[2];
	// i==0
	temp[0]=(int)(u0[2]*row0[0] + u0[3]*row0[1] + u0[4]*row0[2]);
	temp[1]=(int)(u1[2]*row1[0] + u1[3]*row1[1] + u1[4]*row1[2]);
	aft[index]=temp[0]+temp[1];
	e_aft+=aft[index];
	// i==1
	temp[0]=(int)(u0[1]*row0[0] + u0[2]*row0[1] + u0[3]*row0[2] + u0[4]*row0[3]);
	temp[1]=(int)(u1[1]*row1[0] + u1[2]*row1[1] + u1[3]*row1[2] + u1[4]*row1[3]);
	aft[ww+index]=temp[0]+temp[1];
	e_aft+=aft[ww+index];
	// i==hh-2
	temp[0]=(int)(u0[0]*row0[hh-4] + u0[1]*row0[hh-3] + u0[2]*row0[hh-2] + u0[3]*row0[hh-1]);
	temp[1]=(int)(u1[0]*row1[hh-4] + u1[1]*row1[hh-3] + u1[2]*row1[hh-2] + u1[3]*row1[hh-1]);
	aft[(hh-2)*ww+index]=temp[0]+temp[1];
	e_aft+=aft[(hh-2)*ww+index];
	// i==hh-1
	temp[0]=(int)(u0[0]*row0[hh-3] + u0[1]*row0[hh-2] + u0[2]*row0[hh-1]);
	temp[1]=(int)(u1[0]*row1[hh-3] + u1[1]*row1[hh-2] + u1[2]*row1[hh-1]);
	aft[(hh-1)*ww+index]=temp[0]+temp[1];
	e_aft+=aft[(hh-1)*ww+index];
	for(int i=2; i<hh-2; i++){
		// convolusion
		temp[0]=(int)(u0[0]*row0[i-2] + u0[1]*row0[i-1] + u0[2]*row0[i] + u0[3]*row0[i+1] + u0[4]*row0[i+2]);
		temp[1]=(int)(u1[0]*row1[i-2] + u1[1]*row1[i-1] + u1[2]*row1[i] + u1[3]*row1[i+1] + u1[4]*row1[i+2]);
		
		// add result
		aft[i*ww+index]=temp[0]+temp[1];
		e_aft+=aft[i*ww+index];
	}

	return e_aft;
}

int up_convolusion_row(float u0[5], float u1[5], int *row0, int *row1, int ww, int index, int *aft){
	int e_aft=0;
	int temp[2];
	// i==0
	temp[0]=(int)(u0[2]*row0[0] + u0[3]*row0[1] + u0[4]*row0[2]);
	temp[1]=(int)(u1[2]*row1[0] + u1[3]*row1[1] + u1[4]*row1[2]);
	aft[index*ww]=temp[0]+temp[1];
	e_aft+=(temp[0]+temp[1]);
	// i==1
	temp[0]=(int)(u0[1]*row0[0] + u0[2]*row0[1] + u0[3]*row0[2] + u0[4]*row0[3]);
	temp[1]=(int)(u1[1]*row1[0] + u1[2]*row1[1] + u1[3]*row1[2] + u1[4]*row1[3]);
	aft[index*ww +1]=temp[0]+temp[1];
	e_aft+=(temp[0]+temp[1]);
	// i==ww-2
	temp[0]=(int)(u0[0]*row0[ww-4] + u0[1]*row0[ww-3] + u0[2]*row0[ww-2] + u0[3]*row0[ww-1]);
	temp[1]=(int)(u1[0]*row1[ww-4] + u1[1]*row1[ww-3] + u1[2]*row1[ww-2] + u1[3]*row1[ww-1]);
	aft[index*ww +ww-2]=temp[0]+temp[1];
	e_aft+=(temp[0]+temp[1]);
	// i==ww-1
	temp[0]=(int)(u0[0]*row0[ww-3] + u0[1]*row0[ww-2] + u0[2]*row0[ww-1]);
	temp[1]=(int)(u1[0]*row1[ww-3] + u1[1]*row1[ww-2] + u1[2]*row1[ww-1]);
	aft[index*ww +ww-1]=temp[0]+temp[1];
	e_aft+=(temp[0]+temp[1]);

	for(int i=2; i<ww-2; i++){ // ww=w*3/2
	// convolusion
		temp[0]=(int)(u0[0]*row0[i-2] + u0[1]*row0[i-1] + u0[2]*row0[i] + u0[3]*row0[i+1] + u0[4]*row0[i+2]);
		temp[1]=(int)(u1[0]*row1[i-2] + u1[1]*row1[i-1] + u1[2]*row1[i] + u1[3]*row1[i+1] + u1[4]*row1[i+2]);
		
		// add result
		aft[index*ww +i]=temp[0]+temp[1];
		e_aft+=(temp[0]+temp[1]);
	}

	return e_aft;
}

void up(int *ori_R, int *ori_G, int *ori_B, int *aft_R, int *aft_G, int *aft_B, int w, int h, int ww, int hh){
	float u1[5]={-0.1, 0.119, 0.927, 0.6, -0.047};
	float u0[5]={-0.047, 0.6, 0.927, 0.119, -0.1};
	float R_rate, G_rate, B_rate;
	int R_ori, G_ori, B_ori, e_aft;
	int *row0;
	int *row1;
	printf("in up: w=%d, h=%d, ww=%d, hh=%d\n", w, h, ww, hh);
	/*
	// column
	row0=(int*)malloc(sizeof(int)*hh);
	row1=(int*)malloc(sizeof(int)*hh);

	for(int i=0; i<w; i++){ // do how many time in column
		R_ori=G_ori=B_ori=0;
		for(int j=0; j<h; j++){
			R_ori+=ori_R[j*w+i]; // compute weight
			G_ori+=ori_G[j*w+i];
			B_ori+=ori_B[j*w+i];
		}
		// red
		up_setup_col(row0, row1, w, hh, i, ori_R);
		e_aft=up_convolusion_col(u0, u1, row0, row1, w, hh, i, aft_R);
		R_rate=(float)e_aft/(float)(R_ori*3.0/2.0);
		// green
		up_setup_col(row0, row1, w, hh, i, ori_G);
		e_aft=up_convolusion_col(u0, u1, row0, row1, w, hh, i, aft_G);
		G_rate=(float)e_aft/(float)(G_ori*3.0/2.0);
		// blue
		up_setup_col(row0, row1, w, hh, i, ori_B);
		e_aft=up_convolusion_col(u0, u1, row0, row1, w, hh, i, aft_B);
		B_rate=(float)e_aft/(float)(B_ori*3.0/2.0);
		for(int j=0; j<hh; j++){
			aft_R[j*w+i]=(int)((float)(aft_R[j*w+i])/R_rate);
			aft_G[j*w+i]=(int)((float)(aft_G[j*w+i])/G_rate);
			aft_B[j*w+i]=(int)((float)(aft_B[j*w+i])/B_rate);
		}
	} // column finish
	
	row0=(int*)malloc(sizeof(int)*ww);
	row1=(int*)malloc(sizeof(int)*ww);
	// row
	for(int j=0; j<hh; j++){ // do how many time in row
		R_ori=G_ori=B_ori=0;
		for(int i=0; i<w; i++){ // for each row in Img_ori(width)
			R_ori+=aft_R[j*w +i]; // compute weight
			G_ori+=aft_G[j*w +i];
			B_ori+=aft_B[j*w +i];
		}
		// red
		up_setup_row(row0, row1, aft_R, w, ww, j);
		e_aft=up_convolusion_row(u0, u1, row0, row1, ww, j, aft_R);
		R_rate=(float)e_aft/(float)(R_ori*3.0/2.0);
		// green
		up_setup_row(row0, row1, aft_G, w, ww, j);
		e_aft=up_convolusion_row(u0, u1, row0, row1, ww, j, aft_G);
		G_rate=(float)e_aft/(float)(G_ori*3.0/2.0);
		// blue
		up_setup_row(row0, row1, aft_B, w, ww, j);
		e_aft=up_convolusion_row(u0, u1, row0, row1, ww, j, aft_B);
		B_rate=(float)e_aft/(float)(B_ori*3.0/2.0);
		for(int i=0; i<ww; i++){
			aft_R[j*ww+i]=(int)((float)(aft_R[j*ww+i])/R_rate);
			aft_G[j*ww+i]=(int)((float)(aft_G[j*ww+i])/G_rate);
			aft_B[j*ww+i]=(int)((float)(aft_B[j*ww+i])/B_rate);
		}
	} // row finish
	*/
	
	row0=(int*)malloc(sizeof(int)*ww);
	row1=(int*)malloc(sizeof(int)*ww);
	// row
	for(int j=0; j<h; j++){ // do how many time in row
		R_ori=0;
		G_ori=0;
		B_ori=0;
		for(int i=0; i<w; i++){ // for each row in Img_ori(width)
			R_ori+=ori_R[j*w +i]; // compute weight
			G_ori+=ori_G[j*w +i];
			B_ori+=ori_B[j*w +i];
		}
		// red
		up_setup_row(row0, row1, ori_R, w, ww, j);
		e_aft=up_convolusion_row(u0, u1, row0, row1, ww, j, aft_R);
		R_rate=(float)e_aft/(float)(R_ori*3.0/2.0);
		// green
		up_setup_row(row0, row1, ori_G, w, ww, j);
		e_aft=up_convolusion_row(u0, u1, row0, row1, ww, j, aft_G);
		G_rate=(float)e_aft/(float)(G_ori*3.0/2.0);
		// blue
		up_setup_row(row0, row1, ori_B, w, ww, j);
		e_aft=up_convolusion_row(u0, u1, row0, row1, ww, j, aft_B);
		B_rate=(float)e_aft/(float)(B_ori*3.0/2.0);
		for(int i=0; i<ww; i++){
			aft_R[j*ww+i]=(int)((float)(aft_R[j*ww+i])/R_rate);
			aft_G[j*ww+i]=(int)((float)(aft_G[j*ww+i])/G_rate);
			aft_B[j*ww+i]=(int)((float)(aft_B[j*ww+i])/B_rate);
		}
	} // row finish
	
	// column
	/*
	row0=(int*)malloc(sizeof(int)*hh);
	row1=(int*)malloc(sizeof(int)*hh);

	for(int i=0; i<ww; i++){ // do how many time in column
		R_ori=0;
		G_ori=0;
		B_ori=0;
		for(int j=0; j<h; j++){
			R_ori+=aft_R[j*ww+i]; // compute weight
			G_ori+=aft_G[j*ww+i];
			B_ori+=aft_B[j*ww+i];
		}
		// red
		up_setup_col(row0, row1, ww, hh, i, aft_R);
		e_aft=up_convolusion_col(u0, u1, row0, row1, ww, hh, i, aft_R);
		R_rate=(float)e_aft/(float)(R_ori*3.0/2.0);
		// green
		up_setup_col(row0, row1, ww, hh, i, aft_G);
		e_aft=up_convolusion_col(u0, u1, row0, row1, ww, hh, i, aft_G);
		G_rate=(float)e_aft/(float)(G_ori*3.0/2.0);
		// blue
		up_setup_col(row0, row1, ww, hh, i, aft_B);
		e_aft=up_convolusion_col(u0, u1, row0, row1, ww, hh, i, aft_B);
		B_rate=(float)e_aft/(float)(B_ori*3.0/2.0);
		for(int j=0; j<hh; j++){
			aft_R[j*ww+i]=(int)((float)(aft_R[j*ww+i])/R_rate);
			aft_G[j*ww+i]=(int)((float)(aft_G[j*ww+i])/G_rate);
			aft_B[j*ww+i]=(int)((float)(aft_B[j*ww+i])/B_rate);
		}
	}*/
}