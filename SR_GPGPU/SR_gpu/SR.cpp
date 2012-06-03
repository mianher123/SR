#include "all_lib.h"

// OpenGL Globals
GLuint texture;
GLuint pbo;

// CUDA Globals
struct cudaGraphicsResource* cuda_resource;

// OpenCV Globals
CvCapture *capture;
using namespace cv;

// GLUT Callback Function Prototypes
void renderScene(void);

// Main Function
int main(int argc, char** argv){

	// Declare output width and height of the video. Automate this later.
	//int w=540;
	//int h=405;
	int w=468;
	int h=324;


    // Load AVI file. This should be given by user input in the future.
	capture = cvCaptureFromAVI("D:\\close.avi");

    // Stop if the player cannot obtain the file.
    if( !capture ) return 1;

	// Initialize GLUT, GLEW and create a window.
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(w,h);
	glutCreateWindow("HD Koala Bears Super-Resolution Player");
	glewInit();

	// interoperate CUDA and OpenGL
	cudaGLSetGLDevice(0);

	// Set viewport to orthographic projection, since we only need to display 2D images.
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	// Enable OpenGL texture.
	glEnable(GL_TEXTURE_2D);

	// Bind, allocate and unbind the texture to be drawn onto the screen.
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	
	// Bind, allocate and unbind the pixel buffer object to be modified by CUDA.
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, w*h*4, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// Register the pixel buffer object to "cuda_resource"
	cudaGraphicsGLRegisterBuffer(&cuda_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

	// Assign GLUT callbacks.
	glutDisplayFunc(renderScene);
	glutIdleFunc(renderScene);

	// Start GLUT main loop.
	glutMainLoop();
}

void renderScene(void){

	// Get a frame from the video footage.
	IplImage *pImg;
    pImg = cvQueryFrame( capture );
       
    // Stop when the player cannot obtain a frame. 
    if( !pImg ) return;

	// Create timers as performance benchmarks.
	clock_t start, end, Up, find, Down;

	// Collect info about the frame.
	int ori_w=pImg->width;
	int ori_h=pImg->height;
	int bpp=pImg->nChannels;
	int depth=pImg->depth;

	// Calculate the "effective" input width (w) and height (h), which must be a multiple of 6.
	int w=ori_w-(ori_w%6);
	int h=ori_h-(ori_h%6);
	
	// Calculate the output width (ww) and height (hh).
	int ww=w*3/2;
	int hh=h*3/2;

	printf("ori_w=%d, ori_h=%d, w=%d, h=%d, ww=%d, hh=%d\n", ori_w, ori_h, w, h, ww, hh);

	// Allocate memory for the original input image.
	unsigned char *ori_R=(unsigned char*)malloc(sizeof(unsigned char)*w*h);
	unsigned char *ori_G=(unsigned char*)malloc(sizeof(unsigned char)*w*h);
	unsigned char *ori_B=(unsigned char*)malloc(sizeof(unsigned char)*w*h);
	
	// Allocate memory for the smoothed image.
	unsigned char *DownUp_R=(unsigned char*)malloc(sizeof(unsigned char)*w*h);
	unsigned char *DownUp_G=(unsigned char*)malloc(sizeof(unsigned char)*w*h);
	unsigned char *DownUp_B=(unsigned char*)malloc(sizeof(unsigned char)*w*h);

	// Allocate memory for the high frequency band of the image.
	unsigned char *H_R=(unsigned char*)malloc(sizeof(unsigned char)*w*h);
	unsigned char *H_G=(unsigned char*)malloc(sizeof(unsigned char)*w*h);
	unsigned char *H_B=(unsigned char*)malloc(sizeof(unsigned char)*w*h);
	
	// Allocate memory for the interpolative upscaled image.
	unsigned char *aft_R=(unsigned char*)malloc(sizeof(unsigned char)*ww*hh);
	unsigned char *aft_G=(unsigned char*)malloc(sizeof(unsigned char)*ww*hh);
	unsigned char *aft_B=(unsigned char*)malloc(sizeof(unsigned char)*ww*hh);

	// Map "cuda_resource" and obtain a pointer "tex" to the pixel buffer object.
	uchar4* tex;
	size_t num_bytes;
	cudaGraphicsMapResources(1, &cuda_resource);
	cudaGraphicsResourceGetMappedPointer((void**)&tex, &num_bytes, cuda_resource);
	
	// Move the original image from OpenCV space to memory where the player has full control.
	for(int j=0; j<h; ++j){
		for(int i=0; i<w; ++i){
			ori_R[j*w +i]=(unsigned char)pImg->imageData[(j*ori_w +i)*bpp +2];
			ori_G[j*w +i]=(unsigned char)pImg->imageData[(j*ori_w +i)*bpp +1];
			ori_B[j*w +i]=(unsigned char)pImg->imageData[(j*ori_w +i)*bpp +0];
		}
	}

	
	// Calculate smoothed image.
	start=clock();
	SR_kernel_down(ori_R, ori_G, ori_B, aft_R, aft_G, aft_B, w, h);
	Down=clock();
	SR_kernel_up(aft_R, aft_G, aft_B, DownUp_R, DownUp_G, DownUp_B, w*2/3, h*2/3, w, h);
	Up=clock();

	// Calculate upscaled image.
	SR_kernel_up(ori_R, ori_G, ori_B, aft_R, aft_G, aft_B, w, h, ww, hh);
	
	// RECAP
	//	"ori_R" stores original image
	//	"aft_R" stores 1.5x upsampled image
	//	"DownUp_R" stores down sampled then upsampled image

	// Calculate the high-frequency band of the original image.
	for(int j=0; j<h; ++j){
		for(int i=0; i<w; ++i){
			int hifreq;

			hifreq = (int)ori_R[j*w +i] - (int)DownUp_R[j*w +i];
			if (hifreq > 255) hifreq = 255;
			else if (hifreq < 0) hifreq = 0;
			H_R[j*w +i] = (unsigned char)hifreq;

			hifreq = (int)ori_G[j*w +i] - (int)DownUp_G[j*w +i];
			if (hifreq > 255) hifreq = 255;
			else if (hifreq < 0) hifreq = 0;
			H_G[j*w +i] = (unsigned char)hifreq;

			hifreq = (int)ori_B[j*w +i] - (int)DownUp_B[j*w +i];
			if (hifreq > 255) hifreq = 255;
			else if (hifreq < 0) hifreq = 0;
			H_B[j*w +i] = (unsigned char)hifreq;
		}
	}
	
	// Match patches between interpolative upscaled and smoothed imags.
	// Also, fill the high frequency band into the upscaled image and write into "tex".
	find=clock();
	SR_kernel_find_neighbor(aft_R, aft_G, aft_B,
							DownUp_R, DownUp_G, DownUp_B,
							H_R, H_G, H_B,
							w, h, ww, hh, tex);
	end=clock();

	// Debug usage only.
	printf("error1: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	// Unmap "cuda_resource" and return the responsibility of the pixel buffer object to OpenGL. 
	cudaGraphicsUnmapResources(1, &cuda_resource, 0);

	// Print performance benchmark.
	std::cout << std::dec << "Down time: " << 1000.0*(double)(Down-start)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	std::cout << std::dec << "DownUp time: " << 1000.0*(double)(Up-Down)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	std::cout << std::dec << "up time: " << 1000.0*(double)(find-Up)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	std::cout << std::dec << "find neighbor time: " << 1000.0*(double)(end-find)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	std::cout << std::dec << "process time: " << 1000.0*(double)(end-start)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	printf("before show: ww=%d, hh=%d\n", ww, hh);

	// Copy the pixel buffer object to the texture..
	glBindTexture(GL_TEXTURE_2D, texture);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ww, hh, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	
	// Draw the texture onto the display buffer.
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

	// Show the display buffer.
	glutSwapBuffers();

	// Free memory.
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

	return;
}
