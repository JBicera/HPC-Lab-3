#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <getopt.h>
#include "terarec.h"

extern int teravalidate(terarec_t *local_data, int local_len);

#define USAGE \
"usage:\n"                                                                     \
"  teravalidate [options]\n"                                                       \
"options:\n"                                                                   \
"  -f [filename]       Filename (Default: data.dat)\n"                         \
"  -h                  Show this help message\n"

int main (int argc, char *argv[]){
	int P, rank;
	MPI_Init (&argc, &argv);
	MPI_Comm_size (MPI_COMM_WORLD, &P);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	char * filename = "data.dat"; 
	int c;

	while ((c = getopt (argc, argv, "hf:")) != -1){
		switch (c){
			case 'f':
				filename = optarg;
				break;
			case 'h':
				fprintf(stderr, USAGE);
				exit(EXIT_SUCCESS);
			default:
				fprintf(stderr, USAGE);
				exit(EXIT_FAILURE);
		}
	}

	//Registering the terarec type
	teraMPICommitType();

	int  local_len;
	terarec_t *local_data = readEqualShare(filename, &local_len);

	int error_acc = teravalidate(local_data, local_len);

	if (rank == 0 && error_acc)
		fprintf(stderr, "Data not sorted.\n");

	free(local_data);

	MPI_Finalize();

	return EXIT_SUCCESS;
}
