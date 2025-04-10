#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <sys/mman.h>
#include "terarec.h"

#define USAGE \
"usage:\n"                                                                     \
"  naivesort [options]\n"                                                       \
"options:\n"                                                                   \
"  -f [filename]       File to sort (Default: 'data.dat')\n"                         \
"  -h                  Show this help message\n"

int main (int argc, char *argv[]){
	MPI_Init (&argc, &argv);
	int P, rank;
	MPI_Comm_size (MPI_COMM_WORLD, &P);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	char* filename = "data.dat";

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

	teraMPICommitType();

	if(rank == 0){

		int fd;
		if(0 > (fd = open(filename, O_RDONLY))){
			fprintf(stderr, "Failed to open file %s.", filename);
			exit(EXIT_FAILURE);
		}

		int nrecs = teraRecsInFile(fd);
		size_t pagesize = sysconf(_SC_PAGESIZE);
		size_t len = (((nrecs * sizeof(terarec_t) - 1)/ pagesize) + 1) * pagesize;
		terarec_t *data = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);

		double start = MPI_Wtime();
		qsort(data, nrecs, sizeof(terarec_t), teraCompare);
		double elapsed = MPI_Wtime() - start;

		printf("elapsed time : %f\n", elapsed);

		munmap(data, len);
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}
