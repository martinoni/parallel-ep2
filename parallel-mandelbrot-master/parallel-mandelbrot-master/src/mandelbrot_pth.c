#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<pthread.h>
#include <time.h>
#include <sys/time.h>

struct timer_info {
    clock_t c_start;
    clock_t c_end;
    struct timespec t_start;
    struct timespec t_end;
    struct timeval v_start;
    struct timeval v_end;
};

struct timer_info timer;

void start_timer(){
  timer.c_start = clock();
  clock_gettime(CLOCK_MONOTONIC, &timer.t_start);
  gettimeofday(&timer.v_start, NULL);
}

void stop_timer(){
  timer.c_end = clock();
  clock_gettime(CLOCK_MONOTONIC, &timer.t_end);
  gettimeofday(&timer.v_end, NULL);
}

void print_results(){
  printf("%f, %f , %f",
               (double) (timer.c_end - timer.c_start) / (double) CLOCKS_PER_SEC,
               (double) (timer.t_end.tv_sec - timer.t_start.tv_sec) +
               (double) (timer.t_end.tv_nsec - timer.t_start.tv_nsec) / 1000000000.0,
               (double) (timer.v_end.tv_sec - timer.v_start.tv_sec) +
               (double) (timer.v_end.tv_usec - timer.v_start.tv_usec) / 1000000.0);
}

struct thread_data{
  int thread_id;
  int task_size;
  int  thread_start;
  int thread_stop;
};


double c_x_min;
double c_x_max;
double c_y_min;
double c_y_max;

double pixel_width;
double pixel_height;


int iteration_max = 200;
int i_o;
int n_threads;
int image_size;
unsigned char **image_buffer;

int i_x_max;
int i_y_max;
int image_buffer_size;

int gradient_size = 16;
int colors[17][3] = {
                        {66, 30, 15},
                        {25, 7, 26},
                        {9, 1, 47},
                        {4, 4, 73},
                        {0, 7, 100},
                        {12, 44, 138},
                        {24, 82, 177},
                        {57, 125, 209},
                        {134, 181, 229},
                        {211, 236, 248},
                        {241, 233, 191},
                        {248, 201, 95},
                        {255, 170, 0},
                        {204, 128, 0},
                        {153, 87, 0},
                        {106, 52, 3},
                        {16, 16, 16},
                    };

void allocate_image_buffer(){
    int rgb_size = 3;
    image_buffer = (unsigned char **) malloc(sizeof(unsigned char *) * image_buffer_size);

    for(int i = 0; i < image_buffer_size; i++){
        image_buffer[i] = (unsigned char *) malloc(sizeof(unsigned char) * rgb_size);
    };
};

void init(int argc, char *argv[]){
  if(argc < 8){
        printf("usage: ./mandelbrot_pth c_x_min c_x_max c_y_min c_y_max image_size i_o n_threads\n");
        printf("examples with image_size = 11500:\n");
        printf("    Full Picture:         ./mandelbrot_pth -2.5 1.5 -2.0 2.0 11500 1 8\n");
        printf("    Seahorse Valley:      ./mandelbrot_pth -0.8 -0.7 0.05 0.15 11500 0 32\n");
        printf("    Elephant Valley:      ./mandelbrot_pth 0.175 0.375 -0.1 0.1 11500 1 2\n");
        printf("    Triple Spiral Valley: ./mandelbrot_pth -0.188 -0.012 0.554 0.754 11500 0 10\n");
        exit(0);
    }
    else{
        sscanf(argv[1], "%lf", &c_x_min);
        sscanf(argv[2], "%lf", &c_x_max);
        sscanf(argv[3], "%lf", &c_y_min);
        sscanf(argv[4], "%lf", &c_y_max);
        sscanf(argv[5], "%d", &image_size);
	sscanf(argv[6],"%d", &i_o);
        sscanf(argv[7], "%d", &n_threads);

	i_x_max           = image_size;
        i_y_max           = image_size;
        image_buffer_size = image_size * image_size;

        pixel_width       = (c_x_max - c_x_min) / i_x_max;
        pixel_height      = (c_y_max - c_y_min) / i_y_max;
    };
};

void update_rgb_buffer(int iteration, int x, int y){
    int color;

    if(iteration == iteration_max){
        image_buffer[(i_y_max * y) + x][0] = colors[gradient_size][0];
        image_buffer[(i_y_max * y) + x][1] = colors[gradient_size][1];
        image_buffer[(i_y_max * y) + x][2] = colors[gradient_size][2];
    }
    else{
        color = iteration % gradient_size;

        image_buffer[(i_y_max * y) + x][0] = colors[color][0];
        image_buffer[(i_y_max * y) + x][1] = colors[color][1];
        image_buffer[(i_y_max * y) + x][2] = colors[color][2];
    };
};

void write_to_file(){
    FILE * file;
    char * filename               = "output_pth.ppm";
    char * comment                = "# ";

    int max_color_component_value = 255;

    file = fopen(filename,"wb");
    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
            i_x_max, i_y_max, max_color_component_value);

    for(int i = 0; i < image_buffer_size; i++){
        fwrite(image_buffer[i], 1 , 3, file);
    };

    fclose(file);
};

void* compute_mandelbrot(void *args){
  //Data initialization

  int thread_id;
  int  thread_stop;
  int thread_start;

  
  struct thread_data *data;
  data = (struct thread_data *) args;
  thread_id=data->thread_id;
  thread_start= data->thread_start;
  thread_stop=data->thread_stop;

  // Previous routine
    double z_x;
    double z_y;
    double z_x_squared;
    double z_y_squared;
    double escape_radius_squared = 4;

    int iteration;
    int i;
    int line;
    int col;

    double c_x;
    double c_y;
     /* printf("Thread %d will calculate from %f to %f\n",thread_id, thread_start,thread_stop); */ 
    for(i = thread_start; i < thread_stop; i++){
      line = i/image_size;
      col = i%image_size;
	  c_y = c_y_min + line * pixel_height;
	  if(fabs(c_y) < pixel_height / 2){
	    c_y = 0.0;
	  };

            c_x         = c_x_min + col * pixel_width;
	    z_x         = 0.0;
            z_y         = 0.0;

            z_x_squared = 0.0;
            z_y_squared = 0.0;
	    
            for(iteration = 0;
                iteration < iteration_max && \
                ((z_x_squared + z_y_squared) < escape_radius_squared);
                iteration++){
                z_y         = 2 * z_x * z_y + c_y;
                z_x         = z_x_squared - z_y_squared + c_x;

                z_x_squared = z_x * z_x;
                z_y_squared = z_y * z_y;
            };

            update_rgb_buffer(iteration, col, line);
	    
    };
};

void compute_mandelbrot_pthreads(){
  // Create  variables
  pthread_t threads[n_threads];
  struct thread_data data[n_threads];
  int i=0;
  int error_code;
  int task_size ;
  int rest;
  // Divide Tasks for each thread
  task_size= image_buffer_size/n_threads;
  rest=image_buffer_size % n_threads;
  //Procedure
  for (i=0; i<n_threads;i++){
    data[i].thread_id=i+1;
    data[i].task_size = task_size;
    if(i==0){
    data[i].thread_start=0;
    data[i].thread_stop= task_size + rest;}
    else {
      data[i].thread_start=data[i-1].thread_stop + 1;
      data[i].thread_stop=data[i-1].thread_stop + task_size;
    };
    error_code = pthread_create(&threads[i],
                                NULL,
                                compute_mandelbrot,
                                (void *)&data[i]);
    if(error_code){
      printf("pthread_create returned error code %d for thread %d" ,
             data[i].thread_id,
             error_code);
      exit(-1);
    }
  }
  i=0;
  for(i=0;i<n_threads;i++){
    pthread_join(threads[i],NULL);
  }
}

int main(int argc, char *argv[]){
  init(argc,argv);
  if(i_o==1){
    start_timer();
    allocate_image_buffer();
    compute_mandelbrot_pthreads();
    write_to_file();
    stop_timer();
    print_results();   
  }
  else{
    allocate_image_buffer();
    start_timer();
    compute_mandelbrot_pthreads();
    stop_timer();
    print_results();
  }
  return 0;
};
