#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

struct timer_info {
  clock_t c_start;
  clock_t c_end;
  struct timespec t_start;
  struct timespec t_end;
  struct timeval v_start;
  struct timeval v_end;
};

struct timer_info timer;

void start_timer() {
  timer.c_start = clock();
  clock_gettime(CLOCK_MONOTONIC, &timer.t_start);
  gettimeofday(&timer.v_start, NULL);
}

void stop_timer() {
  timer.c_end = clock();
  clock_gettime(CLOCK_MONOTONIC, &timer.t_end);
  gettimeofday(&timer.v_end, NULL);
}

void print_results() {
  printf("%f, %f , %f\n",
         (double)(timer.c_end - timer.c_start) / (double)CLOCKS_PER_SEC,
         (double)(timer.t_end.tv_sec - timer.t_start.tv_sec) +
             (double)(timer.t_end.tv_nsec - timer.t_start.tv_nsec) /
                 1000000000.0,
         (double)(timer.v_end.tv_sec - timer.v_start.tv_sec) +
             (double)(timer.v_end.tv_usec - timer.v_start.tv_usec) / 1000000.0);
}

double c_x_min;
double c_x_max;
double c_y_min;
double c_y_max;

double pixel_width;
double pixel_height;

__device__ int iteration_max = 200;

int image_size;
unsigned char *image_buffer;
unsigned char *d_image_buffer;
unsigned char **image_buffer_formatted;

int i_x_max;
int i_y_max;
int image_buffer_size;
int i_o;
int grid_x;
int grid_y;
int block_x;
int block_y;

__device__ int gradient_size = 16;
__device__ int colors[17][3] = {
    {66, 30, 15},    {25, 7, 26},     {9, 1, 47},      {4, 4, 73},
    {0, 7, 100},     {12, 44, 138},   {24, 82, 177},   {57, 125, 209},
    {134, 181, 229}, {211, 236, 248}, {241, 233, 191}, {248, 201, 95},
    {255, 170, 0},   {204, 128, 0},   {153, 87, 0},    {106, 52, 3},
    {16, 16, 16},
};

void allocate_image_buffer() {
  // printf("OK");
  int rgb_size = 3;
  image_buffer_formatted =
      (unsigned char **)malloc(sizeof(unsigned char *) * image_buffer_size);
  image_buffer = (unsigned char *)malloc(sizeof(unsigned char) *
                                         image_buffer_size * rgb_size);
  cudaMalloc((void **)&d_image_buffer,
             sizeof(unsigned char) * image_buffer_size * rgb_size);

  for (int i = 0; i < image_buffer_size; i++) {
    image_buffer_formatted[i] =
        (unsigned char *)malloc(sizeof(unsigned char) * rgb_size);
  };
};

void init(int argc, char *argv[]) {
  if (argc < 10) {
    printf("usage: ./mandelbrot_cuda c_x_min c_x_max c_y_min c_y_max image_size grid_x grid_y block_x block_y\n");
    printf("examples with image_size = 11500:\n");
    printf("    Full Picture:         ./mandelbrot_cuda -2.5 1.5 -2.0 2.0 11500 8 8 8 8\n");
    printf("    Seahorse Valley:      ./mandelbrot_cuda -0.8 -0.7 0.05 0.15 11500 32 2 6 16\n");
    printf("    Elephant Valley:      ./mandelbrot_cuda 0.175 0.375 -0.1 0.1 11500 12 4 32 32\n");
    printf("    Triple Spiral Valley: ./mandelbrot_cuda -0.188 -0.012 0.554 0.754 11500 3 3 7 15\n");
    exit(0);
    exit(0);
  } else {
<<<<<<< HEAD
    sscanf(argv[1], "%lf", &c_x_min);
    sscanf(argv[2], "%lf", &c_x_max);
    sscanf(argv[3], "%lf", &c_y_min);
    sscanf(argv[4], "%lf", &c_y_max);
    sscanf(argv[5], "%d", &image_size);
    sscanf(argv[6], "%d", &grid_x);
    sscanf(argv[7], "%d", &grid_y);
    sscanf(argv[8], "%d", &block_x);
    sscanf(argv[9], "%d", &block_y);
=======
    sscanf(argv[1], "%d", &grid_x);
    sscanf(argv[2], "%d", &grid_y);
    sscanf(argv[3], "%d", &block_x);
    sscanf(argv[4], "%d", &block_y);

    c_x_min = -0.188;
    c_x_max = -0.012;
    c_y_min = 0.554;
    c_y_max = 0.754;

    image_size = 22000;
>>>>>>> 96fbca4fe4bd47a566ad9ad65dbeb7332bd90ef2

    i_x_max = image_size;
    i_y_max = image_size;

    image_buffer_size = image_size * image_size;

    pixel_width = (c_x_max - c_x_min) / i_x_max;
    pixel_height = (c_y_max - c_y_min) / i_y_max;
  };
};

void write_to_file() {
  FILE *file;
  char *filename = "output.ppm";
  char *comment = "# ";

  int max_color_component_value = 255;

  for (int i = 0; i < image_buffer_size; i++) {
    image_buffer_formatted[i][0] = image_buffer[i];
    image_buffer_formatted[i][1] = image_buffer[i + image_buffer_size];
    image_buffer_formatted[i][2] = image_buffer[i + 2 * image_buffer_size];
  };

  file = fopen(filename, "wb");

  fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment, i_x_max, i_y_max,
          max_color_component_value);

  for (int i = 0; i < image_buffer_size; i++) {
    fwrite(image_buffer_formatted[i], 1, 3, file);
  };

  fclose(file);
};

__global__ void calc(int image_size, double c_x_min, double c_y_min,
                     double pixel_height, double pixel_width, int i_x_max,
                     int i_y_max, int image_buffer_size,
                     unsigned char *image_buffer) {
  double z_x;
  double z_y;
  double z_x_squared;
  double z_y_squared;

  double escape_radius_squared = 4;

  int i_x;
  int i_y;
  int i_x_thread;
  int i_y_thread;
  int index;
  int iteration;

  int color;

  double c_x;
  double c_y;

  int n_tasks_max_x = blockDim.x * gridDim.x;
  int n_tasks_max_y = blockDim.y * gridDim.y;

  i_x_thread = blockIdx.x * blockDim.x + threadIdx.x;
  i_y_thread = blockIdx.y * blockDim.y + threadIdx.y;

  for (i_x = i_x_thread; i_x < i_x_max; i_x += n_tasks_max_x) {
    for (i_y = i_y_thread; i_y < i_y_max; i_y += n_tasks_max_y) {
      index = (i_y_max * i_y) + i_x;
      c_y = c_y_min + i_y * pixel_height;

      if (fabs(c_y) < pixel_height / 2) {
        c_y = 0.0;
      };

      c_x = c_x_min + i_x * pixel_width;

      z_x = 0.0;
      z_y = 0.0;

      z_x_squared = 0.0;
      z_y_squared = 0.0;

      for (iteration = 0; iteration < iteration_max &&
                          ((z_x_squared + z_y_squared) < escape_radius_squared);
           iteration++) {
        z_y = 2 * z_x * z_y + c_y;
        z_x = z_x_squared - z_y_squared + c_x;

        z_x_squared = z_x * z_x;
        z_y_squared = z_y * z_y;
      };

      if (iteration == iteration_max) {
        image_buffer[index] = colors[gradient_size][0];
        image_buffer[index + image_buffer_size] = colors[gradient_size][1];
        image_buffer[index + 2 * image_buffer_size] = colors[gradient_size][2];
      } else {
        color = iteration % gradient_size;

        image_buffer[index] = colors[color][0];
        image_buffer[index + image_buffer_size] = colors[color][1];
        image_buffer[index + 2 * image_buffer_size] = colors[color][2];
      };
    };
  };
}

void compute_mandelbrot() {

  dim3 blockDim(block_x, block_y, 1);
  dim3 gridDim(grid_x, grid_y, 1);

  calc<<<gridDim, blockDim, 0>>>(image_size, c_x_min, c_y_min, pixel_height,
                                 pixel_width, i_x_max, i_y_max,
                                 image_buffer_size, d_image_buffer);
  cudaMemcpy(image_buffer, d_image_buffer,
             sizeof(unsigned char) * image_buffer_size * 3,
             cudaMemcpyDeviceToHost);
};

int main(int argc, char *argv[]) {
  start_timer();
  init(argc, argv);
  allocate_image_buffer();
  compute_mandelbrot();
  write_to_file();
  cudaFree(d_image_buffer);
  free(image_buffer);
  free(image_buffer_formatted);
  stop_timer();
  print_results();
  return 0;
};
