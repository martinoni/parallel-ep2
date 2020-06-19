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
int numtasks, taskid, len;
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
// printf("(x,y): (%d, %d)\n",  i_x, i_y);

_Bool isPerfectSquare(long double x)
{
    // Find floating point value of
    // square root of x.
    long double sr = sqrt(x);

    // If square root is an integer
    return ((sr - floor(sr)) == 0);
}

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
        // iterations[i] = (int *)malloc(sizeof(int));
        // i_xs[i] = (int *)malloc(sizeof(int));
        // i_ys[i] = (int *)malloc(sizeof(int));

        iterations[i] = -1;
        i_xs[i] = -1;
        i_ys[i] = -1;

        iterations_b[i] = -1;
        i_xs_b[i] = -1;
        i_ys_b[i] = -1;
    }
};

void init()
{
    c_x_min = -0.188;
    c_x_max = -0.012;
    c_y_min = 0.554;
    c_y_max = 0.754;

    // MUDAR ISSO
    image_size = 4096;

    max_process_per_dim = sqrt(numtasks);
    // printf("MAX = %d\n", max_process_per_dim);

    i_x_max = image_size;
    i_y_max = image_size;
    image_buffer_size = image_size * image_size;

    pixel_width = (c_x_max - c_x_min) / i_x_max;
    pixel_height = (c_y_max - c_y_min) / i_y_max;

    size_max = (image_size / max_process_per_dim + 1) * (image_size / max_process_per_dim + 1);
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

void compute_mandelbrot(int numtasks, int taskid)
{
    double z_x;
    double z_y;
    double z_x_squared;
    double z_y_squared;
    double escape_radius_squared = 4;

    int iteration;
    int i_x;
    int i_y;
    int i_x_thread;
    int i_y_thread;
    int k = 0;

    double c_x;
    double c_y;

    MPI_Status status;

    if (!isPerfectSquare(numtasks))
    {
        if (taskid == MASTER)
        {
            printf("Quitting. Need an perfect square number of tasks: numtasks=%d\n", numtasks);
        }
    }
    else
    {
        if (taskid == MASTER)
        {
            printf("MASTER: Number of MPI tasks is: %d\n", numtasks);
        }
        i_x_thread = taskid / max_process_per_dim;
        i_y_thread = taskid % max_process_per_dim;

        for (i_y = i_y_thread; i_y < i_y_max; i_y += max_process_per_dim)
        {
            c_y = c_y_min + i_y * pixel_height;

            if (fabs(c_y) < pixel_height / 2)
            {
                c_y = 0.0;
            };

            for (i_x = i_x_thread; i_x < i_x_max; i_x += max_process_per_dim)
            {
                c_x = c_x_min + i_x * pixel_width;

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
                // printf("Taskid: %d --> (%d, %d) --> Interação %d\n", taskid, i_x, i_y, iteration);

                iterations[k] = iteration;
                i_xs[k] = i_x;
                i_ys[k] = i_y;
                k += 1;
                // update_rgb_buffer(iteration, i_x, i_y);
                // printf("(%d, %d): %d\n", image_buffer[(i_y_max * i_y) + i_x][0], image_buffer[(i_y_max * i_y) + i_x][1], image_buffer[(i_y_max * i_y) + i_x][2]);
                // printf("ID:%d -> (%d, %d)\n", taskid, i_x, i_y);
                // if(taskid == MASTER){
                //     // printf("OK\n");
                //     update_rgb_buffer(iteration, i_x, i_y);
                //     for(k = 1; k < numtasks; k++){
                //         printf("OK_recebido: %d\n", k);
                //         MPI_Recv(image_buffer_unit, 3, MPI_INT, k, k, MPI_COMM_WORLD, &status);
                //         // update_rgb_buffer(image_buffer_unit[0], image_buffer_unit[1], image_buffer_unit[2]);
                //     }
                // } else{
                //     image_buffer_unit[0] = iteration;
                //     image_buffer_unit[1] = i_x;
                //     image_buffer_unit[2] = i_y;
                //     MPI_Send(image_buffer_unit, 3, MPI_INT, MASTER, k, MPI_COMM_WORLD);
                //     // printf("OK222:\n");
                // }
            }
        };
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
                // printf("Taskid: %d --> (%d, %d) --> Interação %d\n", 0, i_xs[l], i_ys[l], iterations[l]);
                update_rgb_buffer(iterations[l], i_xs[l], i_ys[l]);
            }
        }
        for (k = 1; k < numtasks; k++)
        {
            // printf("%d\n", iterations_b[0]);
            MPI_Recv(iterations_b, size_max, MPI_INT, k, tag_iteration, MPI_COMM_WORLD, &status);
            MPI_Recv(i_xs_b, size_max, MPI_INT, k, tag_x, MPI_COMM_WORLD, &status);
            MPI_Recv(i_ys_b, size_max, MPI_INT, k, tag_y, MPI_COMM_WORLD, &status);
            // printf("%d\n", iterations_b[0]);

            // printf("OK\n");
            for (int l = 0; l < size_max; l++)
            {
                if (iterations_b[l] > 0)
                {
                    update_rgb_buffer(iterations_b[l], i_xs_b[l], i_ys_b[l]);
                    // printf("Taskid: %d --> (%d, %d) --> Interação %d\n", k, i_xs_b[l], i_ys_b[l], iterations_b[l]);
                }
            }
        }
    }
};

int main(int argc, char *argv[])
{
    // MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    init();

    allocate_image_buffer(taskid);
    // printf("Memória alocada!\n");
    if (taskid == MASTER)
    {
        start_timer();
    }
    compute_mandelbrot(numtasks, taskid);
    stop_timer();
    // printf("OK!!!\n");

    if (taskid == MASTER)
    {
        // write_to_file();
        print_results();
    }

    MPI_Finalize();

    return 0;
};
