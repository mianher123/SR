#include "all_lib.h"

// OpenGL Globals
GLuint texture;
GLuint pbo;
unsigned char texture_map [256][256][4];
struct cudaGraphicsResource* cuda_resource;

CvCapture *capture;
int fps;

using namespace cv;

void renderScene(void);


int main(int argc, char** argv){
  
	int w=540;
	int h=405;

   
    // load the AVI file
    capture = cvCaptureFromAVI("C:\\Users\\dada\\Desktop\\GPU_Image\\SRvideo1.avi");

    // always check
    if( !capture ) return 1;    
   
    // get fps, needed to set the delay
    //fps = ( int )cvGetCaptureProperty( capture, CV_CAP_PROP_FPS );
   
    // display video
    //cvNamedWindow( "video", 0 );
   
    //while( key != 'q' ) {

    //}
   
    //free memory
    //cvReleaseCapture( &capture );
    //cvDestroyWindow( "video" );


	// init GLUT and create Window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(w,h);
	glutCreateWindow("Lighthouse3D - GLUT Tutorial");

	// set orthographic projection
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	// enable texture
	glEnable(GL_TEXTURE_2D);

	// generate a texture
	//passed_time = 0;

	// interoperate CUDA and OpenGL
	cudaGLSetGLDevice(0);
	glewInit();

	// bind texture and apply specify params
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	

	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, w*h*4, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&cuda_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

	// register callbacks
	glutDisplayFunc(renderScene);
	glutIdleFunc(renderScene);
	glutMainLoop();
}

void renderScene(void){
	//uchar4* tex;

	IplImage *pImg;
	/* get a frame */
    pImg = cvQueryFrame( capture );
       
    /* always check */
    if( !pImg ) printf("error\n");

	//char *ImageName = "C:\\Users\\dada\\Desktop\\GPU_Image\\Koala.jpg";
	//char *ImageName = "C:\\Users\\dada\\Desktop\\GPU_Image\\Flower.jpg";

	IplImage *pImg2;
	//pImg=cvLoadImage(ImageName, 1);
	//cvNamedWindow("ShowImage", 1); // create window
	//cvShowImage("ShowImage", pImg); // show image
	clock_t start, end, Up, find, Down;
	int ori_w=pImg->width;
	int ori_h=pImg->height;
	int bpp=pImg->nChannels;
	int depth=pImg->depth;

	int w=ori_w-(ori_w%6);
	int h=ori_h-(ori_h%6);
	
	int ww=w*3/2;
	int hh=h*3/2;
	printf("ori_w=%d, ori_h=%d, w=%d, h=%d, ww=%d, hh=%d\n", ori_w, ori_h, w, h, ww, hh);

	int *ori_R=(int*)malloc(sizeof(int)*w*h);
	int *ori_G=(int*)malloc(sizeof(int)*w*h);
	int *ori_B=(int*)malloc(sizeof(int)*w*h);

	

	
	int *DownUp_R=(int*)malloc(sizeof(int)*w*h);
	int *DownUp_G=(int*)malloc(sizeof(int)*w*h);
	int *DownUp_B=(int*)malloc(sizeof(int)*w*h);

	int *H_R=(int*)malloc(sizeof(int)*w*h);
	int *H_G=(int*)malloc(sizeof(int)*w*h);
	int *H_B=(int*)malloc(sizeof(int)*w*h);
	
	int *aft_R=(int*)malloc(sizeof(int)*ww*hh);
	int *aft_G=(int*)malloc(sizeof(int)*ww*hh);
	int *aft_B=(int*)malloc(sizeof(int)*ww*hh);
	
	int *ans_R=(int*)malloc(sizeof(int)*ww*hh);
	int *ans_G=(int*)malloc(sizeof(int)*ww*hh);
	int *ans_B=(int*)malloc(sizeof(int)*ww*hh);

	//int *test;
	uchar4* tex;
	size_t num_bytes;
	cudaGraphicsMapResources(1, &cuda_resource);
	cudaGraphicsResourceGetMappedPointer((void**)&tex, &num_bytes, cuda_resource);

	//SR_kernel_start(w, h, ww, hh, test);
	
		for(int j=0; j<h; ++j){ // copy the original RGB
			for(int i=0; i<w; ++i){
				ori_R[j*w +i]=(unsigned char)pImg->imageData[(j*ori_w +i)*bpp +2];
				ori_G[j*w +i]=(unsigned char)pImg->imageData[(j*ori_w +i)*bpp +1];
				ori_B[j*w +i]=(unsigned char)pImg->imageData[(j*ori_w +i)*bpp +0];
			}
		}

	start=clock();
	/******* run 1.5x down sample *******/
	//down(ori_R, ori_G, ori_B, aft_R, aft_G, aft_B, w, h);
	SR_kernel_down(ori_R, ori_G, ori_B, aft_R, aft_G, aft_B, w, h);
	Down=clock();
	SR_kernel_up(aft_R, aft_G, aft_B, DownUp_R, DownUp_G, DownUp_B, w*2/3, h*2/3, w, h);
	//up(aft_R, aft_G, aft_B, DownUp_R, DownUp_G, DownUp_B, w*2/3, h*2/3, w, h);
	
	/******* run 1.5x upsample *******/
	Up=clock();
	//up(ori_R, ori_G, ori_B, aft_R, aft_G, aft_B, w, h, ww, hh);
	SR_kernel_up(ori_R, ori_G, ori_B, aft_R, aft_G, aft_B, w, h, ww, hh);
	

	/*******************************************************
		ori_R store original image
		aft_R store 1.5x upsample image
		DownUp_R store down sample then upsample image
	*******************************************************/
	
	for(int j=0; j<h; ++j){
		for(int i=0; i<w; ++i){
			H_R[j*w +i]=ori_R[j*w +i]-DownUp_R[j*w +i];
			H_G[j*w +i]=ori_G[j*w +i]-DownUp_G[j*w +i];
			H_B[j*w +i]=ori_B[j*w +i]-DownUp_B[j*w +i];
		}
	}
	
	find=clock();
	SR_kernel_find_neighbor(aft_R, aft_G, aft_B,
							DownUp_R, DownUp_G, DownUp_B,
							H_R, H_G, H_B,
							ans_R, ans_G, ans_B,
							w, h, ww, hh, tex);
	end=clock();

	printf("error1: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	cudaGraphicsUnmapResources(1, &cuda_resource, 0);



	std::cout << std::dec << "Down time: " << 1000.0*(double)(Down-start)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	std::cout << std::dec << "DownUp time: " << 1000.0*(double)(Up-Down)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	std::cout << std::dec << "up time: " << 1000.0*(double)(find-Up)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	std::cout << std::dec << "find neighbor time: " << 1000.0*(double)(end-find)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	std::cout << std::dec << "process time: " << 1000.0*(double)(end-start)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	//printf("clock per sec=%d\n", CLOCKS_PER_SEC);
	//ww=w*2/3;
	//hh=h*2/3;
	//hh=h;
	//ww=w;
	printf("before show: ww=%d, hh=%d\n", ww, hh);
	//pImg2=cvCreateImage(cvSize(ww, hh), depth, bpp);

	/*
	for(int j=0; j<hh; ++j){ // copy the ans to pImg2
		for(int i=0; i<ww; ++i){
			
			aft_R[j*ww +i]+=ans_R[j*ww +i];
			aft_G[j*ww +i]+=ans_G[j*ww +i];
			aft_B[j*ww +i]+=ans_B[j*ww +i];
			
			if( aft_R[j*ww +i]>255 ) aft_R[j*ww +i]=255;
			else if( aft_R[j*ww +i]<0 ) aft_R[j*ww +i]=0;
			if( aft_G[j*ww +i]>255 ) aft_G[j*ww +i]=255;
			else if( aft_G[j*ww +i]<0 ) aft_G[j*ww +i]=0;
			if( aft_B[j*ww +i]>255 ) aft_B[j*ww +i]=255;
			else if( aft_B[j*ww +i]<0 ) aft_B[j*ww +i]=0;
			
			pImg2->imageData[(j*ww +i)*bpp +2]=aft_R[j*ww +i];
			pImg2->imageData[(j*ww +i)*bpp +1]=aft_G[j*ww +i];
			pImg2->imageData[(j*ww +i)*bpp +0]=aft_B[j*ww +i];
			
		}
	}
	*/
	//SR_kernel_end();
	
	//char *img_save="C:\\Users\\dada\\Desktop\\GPU_Image\\Koala_weard.jpg";
	//cvSaveImage(img_save, pImg2);
	//cvShowImage("ShowImage", pImg2);
	//waitKey(0);

	glBindTexture(GL_TEXTURE_2D, texture);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ww, hh, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	

	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glBindTexture(GL_TEXTURE_2D, texture);
		glBegin(GL_QUADS);
			glTexCoord2f(0.0, 1.0); glVertex2f(0.0, 0.0);
			glTexCoord2f(0.0, 0.0); glVertex2f(0.0, 1.0);
			glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0);
			glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 0.0);
		glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	glutSwapBuffers();
	//cvDestroyWindow("ShowImage");
	//cvReleaseImage(&pImg);
	//cvReleaseImage(&pImg2);
	free(ori_R);
	free(ori_G);
	free(ori_B);
	free(DownUp_R);
	free(DownUp_G);
	free(DownUp_B);
	free(H_R);
	free(H_G);
	free(H_B);
	free(aft_R);
	free(aft_G);
	free(aft_B);

	//cvWaitKey( 1000 / fps );
	return;
}
