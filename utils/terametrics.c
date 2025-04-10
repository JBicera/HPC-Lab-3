#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <getopt.h>
#include "terarec.h"

extern void terasort(terarec_t *local_data, int  local_len, 
	     			 terarec_t **sorted_data, int* sorted_counts, long* sorted_displs);

extern int teravalidate(terarec_t *local_data, int local_len);

#define USAGE \
"usage:\n"                                                                     \
"  terametrics [options]\n"                                                       \
"options:\n"                                                                   \
"  -f [filename]       Filename (Default: data.dat)\n"                         \
"  -c [count]          Number of benchmark runs (Default: 100)\n"              \
"  -h                  Show this help message\n"

int main (int argc, char *argv[]){
	MPI_Init (&argc, &argv);
	int P, rank;
	MPI_Comm_size (MPI_COMM_WORLD, &P);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	char *data_filename = "data.dat";
	long runcount = 100;
	int c;

	while ((c = getopt (argc, argv, "hf:c:")) != -1){
		switch (c){
			case 'f':
				data_filename = optarg;
				break;
			case 'h':
				fprintf(stderr, USAGE);
				exit(EXIT_SUCCESS);
			case 'c':
				runcount = atol(optarg);
				break;
			default:
				fprintf(stderr, USAGE);
				exit(EXIT_FAILURE);
		}
	}

	//Registering the "tera" type with MPI
	teraMPICommitType();

	//Reading the local data from a file
	int local_len;
	terarec_t *local_data = readEqualShare(data_filename, &local_len);

	//Preparing the return values
	terarec_t *sorted_data;
	int *sorted_counts = (int*) malloc(sizeof(int) * P);
	long *sorted_displs = (long*) malloc(sizeof(long) * P);

	//Initialize items needed for metrics
	double runTimes[runcount];

	//Save values across iterations
	int backup_local_len = local_len;
	int *backup_sorted_counts = (int*) malloc(sizeof(int) * P);
	memcpy(backup_sorted_counts, sorted_counts, sizeof(int) * P);
	long *backup_sorted_displs = (long*) malloc(sizeof(long) * P);
	memcpy(backup_sorted_displs, sorted_displs, sizeof(long) * P);
	terarec_t *backup_local_data = (terarec_t*) malloc(backup_local_len * sizeof(terarec_t));
	memcpy(backup_local_data, local_data, backup_local_len * sizeof(terarec_t));

	long loopCnt = 0;

	for (loopCnt = 0; loopCnt < runcount; loopCnt++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		double start_time = MPI_Wtime();
		terasort(local_data, local_len, &sorted_data, sorted_counts, sorted_displs);
		double end_time = MPI_Wtime();
		double calc_diff = end_time - start_time;
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&calc_diff, runTimes+loopCnt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		int error_acc = teravalidate(sorted_data, sorted_counts[rank]);
		if (rank == 0 && error_acc)
		{
			fprintf(stderr, "Solution had invalid results on loop iteration: %ld\n", loopCnt);
			MPI_Abort(MPI_COMM_WORLD, MPI_ERR_KEYVAL);
		}
		//Reset
		memcpy(local_data, backup_local_data, backup_local_len * sizeof(terarec_t));
		local_len = backup_local_len;
		free(sorted_data);
		memcpy(sorted_counts, backup_sorted_counts, sizeof(int) * P);
		memcpy(sorted_displs, backup_sorted_displs, sizeof(long) * P);
	}
	if (rank == 0)
	{
		double temp = 0;

		for (int i = 0; i < runcount -1; i++){
			for(int j = i + 1; j < runcount; j++){
				if (runTimes[j] < runTimes[i]){
					temp = runTimes[i];
					runTimes[i] = runTimes[j];
					runTimes[j] = temp;
				}
			}
		}
		
		if(runcount%2==0) {
     			temp = (runTimes[runcount/2] + runTimes[runcount/2 - 1]) / 2.0;
    		} else {
        		temp = runTimes[runcount/2];
    		}
		
		fprintf(stdout, "Median time over %ld runs: %.4f\n", runcount, temp);
	}

	//Cleaning up
	free(local_data);
	free(backup_local_data);
	free(sorted_counts);
	free(backup_sorted_counts);
	free(sorted_displs);
	free(backup_sorted_displs);

	MPI_Finalize();

	return EXIT_SUCCESS;
}
