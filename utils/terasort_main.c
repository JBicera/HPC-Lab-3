#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <getopt.h>
#include "terarec.h"

static void writeSortedData(char *filename, int  offset, int  len, terarec_t* data){
	int fd;
	if(0 > (fd = open(filename, O_WRONLY, S_IRUSR | S_IWUSR))){
		fprintf(stderr, "Failed to open file %s.", filename);
		exit(EXIT_FAILURE);
	}

	teraWriteToFile(fd, offset, len, data);

	close(fd);
}

extern void terasort(terarec_t *local_data, int  local_len, 
	     			 terarec_t **sorted_data, int* sorted_counts, long* sorted_displs);

#define USAGE \
"usage:\n"                                                                     \
"  terasort [options]\n"                                                       \
"options:\n"                                                                   \
"  -f [filename]       Filename (Default: data.dat)\n"                         \
"  -h                  Show this help message\n"

int main (int argc, char *argv[]){
	MPI_Init (&argc, &argv);
	int P, rank;
	MPI_Comm_size (MPI_COMM_WORLD, &P);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	char *data_filename = "data.dat";
	int c;
	int safeWriteCount;

	while ((c = getopt (argc, argv, "hf:")) != -1){
		switch (c){
			case 'f':
				data_filename = optarg;
				break;
			case 'h':
				fprintf(stderr, USAGE);
				exit(EXIT_SUCCESS);
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

	//Performing the sort
	terasort(local_data, local_len,
			&sorted_data, sorted_counts, sorted_displs);

	//Sleep so concurrent writes don't happen
	for (safeWriteCount=0; safeWriteCount < rank; safeWriteCount++)
		sleep(1);

	//Writing out the data
	writeSortedData(data_filename, sorted_displs[rank], sorted_counts[rank], sorted_data);

	//Cleaning up
	free(local_data);
	free(sorted_data);
	free(sorted_counts);
	free(sorted_displs);

	MPI_Finalize();

	return EXIT_SUCCESS;
}
