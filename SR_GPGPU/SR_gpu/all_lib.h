#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Macros For Windows
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics Headers
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA runtime
#include <cuda.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>

#include "SR_kernel.h"