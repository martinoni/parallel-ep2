#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>

#define MASTER 0

struct timer_info
{
    clock_t c_start;
    clock_t c_end;
    struct timespec t_start;
    struct timespec t_end;
    struct timeval v_start;
    struct timeval v_end;
};

struct timer_info timer;

void start_timer()
{
    timer.c_start = clock();
    clock_gettime(CLOCK_MONOTONIC, &timer.t_start);
    gettimeofday(&timer.v_start, NULL);
}

void stop_timer()
{
    timer.c_end = clock();
    clock_gettime(CLOCK_MONOTONIC, &timer.t_end);
    gettimeofday(&timer.v_end, NULL);
}

void print_results()
{
    printf("%f, %f , %f\n",
           (double)(timer.c_end - timer.c_start) / (double)CLOCKS_PER_SEC,
           (double)(timer.t_end.tv_sec - timer.t_start.tv_sec) +
               (double)(timer.t_end.tv_nsec - timer.t_start.tv_nsec) / 1000000000.0,
           (double)(timer.v_end.tv_sec - timer.v_start.tv_sec) +
               (double)(timer.v_end.tv_usec - timer.v_start.tv_usec) / 1000000.0);
}

double c_x_min;
double c_x_max;
double c_y_min;
double c_y_max;

double pixel_width;
double pixel_height;

int iteration_max = 200;

int image_size;
int *i_xs;
int *i_ys;
int *iterations;
int *i_xs_b;
int *i_ys_b;
int *iterations_b;
unsigned char **image_buffer;
int size_max;

// MPI:
int numprocess, taskid, len;
int tag_iteration = 0;
int tag_x = 1;
int tag_y = 2;
char hostname[MPI_MAX_PROCESSOR_NAME];

int max_process_per_dim;

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

void allocate_image_buffer(int taskid)
{
    if (taskid == MASTER)
    {
        int rgb_size = 3;
        image_buffer = (unsigned char **)malloc(sizeof(unsigned char *) * image_buffer_size);

        for (int i = 0; i < image_buffer_size; i++)
        {
            image_buffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * rgb_size);
        };
    }

    iterations = (int *)malloc((size_max) * sizeof(int));
    i_xs = (int *)malloc((size_max) * sizeof(int));
    i_ys = (int *)malloc((size_max) * sizeof(int));

    iterations_b = (int *)malloc((size_max) * sizeof(int));
    i_xs_b = (int *)malloc((size_max) * sizeof(int));
    i_ys_b = (int *)malloc((size_max) * sizeof(int));

    for (int i = 0; i < size_max; i++)
    {
        iterations[i] = -1;
        i_xs[i] = -1;
        i_ys[i] = -1;

        iterations_b[i] = -1;
        i_xs_b[i] = -1;
        i_ys_b[i] = -1;
    }
};

void init(int argc, char *argv[])
{
      if(argc < 6){
        if(taskid == MASTER){
            printf("usage:  mpirun --quiet --host host:n_processes mandelbrot_ompi c_x_max c_y_min c_y_max image_size\n");
            printf("examples on localhost with image_size = 11500:\n");
            printf("    Full Picture:         mpirun --quiet --host localhost:2 mandelbrot_ompi  -2.5 1.5 -2.0 2.0 11500\n");
            printf("    Seahorse Valley:      mpirun --quiet --host localhost:32 mandelbrot_ompi  -0.8 -0.7 0.05 0.15 11500\n");
            printf("    Elephant Valley:      mpirun --quiet --host localhost:10 mandelbrot_ompi  0.175 0.375 -0.1 0.1 11500\n");
            printf("    Triple Spiral Valley: mpirun --quiet --host localhost:64 mandelbrot_ompi  -0.188 -0.012 0.554 0.754 11500\n");
            MPI_Finalize();
            exit(0);
        }
    } else{
        sscanf(argv[1], "%lf", &c_x_min);
        sscanf(argv[2], "%lf", &c_x_max);
        sscanf(argv[3], "%lf", &c_y_min);
        sscanf(argv[4], "%lf", &c_y_max);
        sscanf(argv[5], "%d", &image_size);

	    i_x_max           = image_size;
        i_y_max           = image_size;
        image_buffer_size = image_size * image_size;

        pixel_width       = (c_x_max - c_x_min) / i_x_max;
        pixel_height      = (c_y_max - c_y_min) / i_y_max;
    };

    max_process_per_dim = image_size/numprocess+1;

    i_x_max = image_size;
    i_y_max = image_size;
    image_buffer_size = image_size * image_size;

    pixel_width = (c_x_max - c_x_min) / i_x_max;
    pixel_height = (c_y_max - c_y_min) / i_y_max;

    size_max = max_process_per_dim * image_size;
};

void update_rgb_buffer(int iteration, int x, int y)
{
    int color;

    if (iteration == iteration_max)
    {
        image_buffer[(i_y_max * y) + x][0] = colors[gradient_size][0];
        image_buffer[(i_y_max * y) + x][1] = colors[gradient_size][1];
        image_buffer[(i_y_max * y) + x][2] = colors[gradient_size][2];
    }
    else
    {
        color = iteration % gradient_size;

        image_buffer[(i_y_max * y) + x][0] = colors[color][0];
        image_buffer[(i_y_max * y) + x][1] = colors[color][1];
        image_buffer[(i_y_max * y) + x][2] = colors[color][2];
    };
};

void write_to_file()
{
    FILE *file;
    char *filename = "output.ppm";
    char *comment = "# ";

    int max_color_component_value = 255;

    file = fopen(filename, "wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
            i_x_max, i_y_max, max_color_component_value);

    for (int i = 0; i < image_buffer_size; i++)
    {
        fwrite(image_buffer[i], 1, 3, file);
    };

    fclose(file);
};

void compute_mandelbrot(int numprocess, int taskid)
{
    double z_x;
    double z_y;
    double z_x_squared;
    double z_y_squared;
    double escape_radius_squared = 4;

    int iteration;
    int i_x;
    int i_y;
    int col;
    int line;
    int k = 0;

    double c_x;
    double c_y;

    MPI_Status status;

    for (int i = taskid; i < image_buffer_size; i += numprocess)
    {
        col = i/image_size;
        line = i%image_size;

        c_y = c_y_min + line * pixel_height;

        if (fabs(c_y) < pixel_height / 2)
        {
            c_y = 0.0;
        };

            c_x = c_x_min + col * pixel_width;

            z_x = 0.0;
            z_y = 0.0;

            z_x_squared = 0.0;
            z_y_squared = 0.0;

            for (iteration = 0;
                    iteration < iteration_max &&
                    ((z_x_squared + z_y_squared) < escape_radius_squared);
                    iteration++)
            {
                z_y = 2 * z_x * z_y + c_y;
                z_x = z_x_squared - z_y_squared + c_x;

                z_x_squared = z_x * z_x;
                z_y_squared = z_y * z_y;
            };

            iterations[k] = iteration;
            i_xs[k] = col;
            i_ys[k] = line;
            k += 1;
    };

    if (taskid != MASTER)
    {
        MPI_Send(iterations, size_max, MPI_INT, MASTER, tag_iteration, MPI_COMM_WORLD);
        MPI_Send(i_xs, size_max, MPI_INT, MASTER, tag_x, MPI_COMM_WORLD);
        MPI_Send(i_ys, size_max, MPI_INT, MASTER, tag_y, MPI_COMM_WORLD);
    }
    else
    {
        for (int l = 0; l < size_max; l++)
        {
            if (iterations[l] > 0)
            {
                update_rgb_buffer(iterations[l], i_xs[l], i_ys[l]);
            }
        }
        for (k = 1; k < numprocess; k++)
        {
            MPI_Recv(iterations_b, size_max, MPI_INT, k, tag_iteration, MPI_COMM_WORLD, &status);
            MPI_Recv(i_xs_b, size_max, MPI_INT, k, tag_x, MPI_COMM_WORLD, &status);
            MPI_Recv(i_ys_b, size_max, MPI_INT, k, tag_y, MPI_COMM_WORLD, &status);

            for (int l = 0; l < size_max; l++)
            {
                if (iterations_b[l] > 0)
                {
                    update_rgb_buffer(iterations_b[l], i_xs_b[l], i_ys_b[l]);
                }
            }
        }
    }
};

int main(int argc, char *argv[])
{
    start_timer();
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocess);

    init(argc, argv);

    allocate_image_buffer(taskid);

    compute_mandelbrot(numprocess, taskid);

    if (taskid == MASTER)
    {
        write_to_file();
        stop_timer();
        print_results();
    }

    MPI_Finalize();

    return 0;
};
