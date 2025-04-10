#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <getopt.h>
#include "terarec.h"

int teravalidate(terarec_t *data, int len){
	int P, rank;
	MPI_Comm_size (MPI_COMM_WORLD, &P);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	//Performing comparisons
	int found_error = 0;
	int error_acc = 0;
	for(long i = 0; i + 1 < len; i++)
		found_error |= (teraCompare(&data[i], &data[i+1]) > 0);

	MPI_Request req;
	if (rank != P - 1)
		MPI_Isend(&data[len-1], 1, mpi_tera_type, rank + 1, 0, MPI_COMM_WORLD, &req);

	if (rank != 0){
		terarec_t lb;
		MPI_Recv(&lb, 1, mpi_tera_type, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		found_error |= (teraCompare(&lb, &data[0]) > 0);
	}

	if (rank != P - 1)
		MPI_Wait(&req, MPI_STATUS_IGNORE);

	MPI_Reduce(&found_error, &error_acc, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);

	return error_acc;
}
